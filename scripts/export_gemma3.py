#!/usr/bin/env python3
"""
export_gemma3.py

Exports Google Gemma-3-4B-Instruct to an ExecuTorch .pte file with:
  - INT4 weight-only quantization (groupwise, group_size=128) — halves the
    weight footprint vs INT8; ~2.0 GB weights, fitting within the iPhone 14
    Pro's 6 GB RAM alongside the KV-cache (~400 MB fp32).
  - KV-cache enabled (static shape, max_cache_len = 2048 tokens) so that
    each decode step only processes one new token instead of the full prefix.
  - CoreML backend delegation for Apple Neural Engine acceleration.
  - SentencePiece tokenizer copied alongside the model.

Usage:
    pip install -r requirements.txt
    python export_gemma3.py --output_dir ./output

The produced artefacts:
    output/gemma3_4b_int4_coreml.pte   — ExecuTorch model bundle (~2.0 GB INT4)
    output/tokenizer.model             — SentencePiece vocabulary

Add both files to your Xcode project under "Copy Bundle Resources".
"""

import argparse
import shutil
from pathlib import Path

import coremltools as ct
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition.coreml_partitioner import (
    CoreMLPartitioner,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,  # fallback for ops CoreML cannot handle
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes.sym_shape_eval_pass import ConstrainedDynamicShapesPass
from torchao.quantization import quantize_, int4_weight_only
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache


# ---------------------------------------------------------------------------
# Gemma-3 Wrapper — exposes a pure torch.export-compatible forward signature
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# KV-cache constants
# ---------------------------------------------------------------------------

# Maximum total context length for the static KV-cache.
# Bumping this grows the cache tensors but not the weight bundle.
MAX_CACHE_LEN: int = 2048


class Gemma3ExportWrapper(torch.nn.Module):
    """
    Wraps Gemma3ForCausalLM with a StaticCache for efficient autoregressive
    decoding via torch.export.

    The forward signature is:

        (input_ids     [1, 1]  int64,
         cache_position [1]    int64)  — absolute position of the current token
        -> logits [1, 1, vocab_size]   float32

    KV-cache is maintained as mutable module state (StaticCache buffers).
    The C++ runtime does not pass or receive explicit KV tensors — it only
    provides input_ids and cache_position and reads the logits output.

    Resetting the cache between conversations is done on the C++ side by
    restarting cache_position from 0; the model overwrites existing cache
    entries at each position and its causal mask ensures that stale entries
    at positions beyond the current one are never attended to.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.static_cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=MAX_CACHE_LEN,
            device="cpu",
            dtype=torch.float32,
        )

    def forward(
        self,
        input_ids: torch.Tensor,       # [1, 1]  int64
        cache_position: torch.Tensor,  # [1]     int64  absolute position
    ) -> torch.Tensor:
        """Returns logits [1, 1, vocab_size]."""
        outputs = self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            past_key_values=self.static_cache,
            use_cache=True,
        )
        return outputs.logits


# ---------------------------------------------------------------------------
# INT4 weight-only quantization (groupwise)
# ---------------------------------------------------------------------------

def apply_int4_weight_quantization(model: torch.nn.Module) -> None:
    """
    Applies INT4 weight-only groupwise quantization via torchao.
    Group size 128 balances quality and compression for the 4B model.
    No calibration dataset is required because only weights are quantized
    (activations remain fp16 at runtime on the ANE).
    """
    quantize_(model, int4_weight_only(group_size=128))


# ---------------------------------------------------------------------------
# Main export routine
# ---------------------------------------------------------------------------

def export_gemma3(model_id: str, output_dir: Path, seq_len: int = 512) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading {model_id} in float32 …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # AutoModelForCausalLM selects Gemma3ForCausalLM automatically for
    # google/gemma-3-* checkpoints (requires transformers >= 4.49).
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # start in fp32 before quantisation
        low_cpu_mem_usage=True,
    )
    hf_model.eval()

    print("[2/6] Applying INT4 weight-only quantisation …")
    apply_int4_weight_quantization(hf_model)

    print("[3/6] Wrapping model for export (with StaticCache + KV-cache) …")
    wrapped = Gemma3ExportWrapper(hf_model)

    # Decode-step dummy inputs: one token at position `seq_len - 1`.
    # With StaticCache the KV buffers are part of the module state — no
    # explicit KV tensors appear in the forward signature.
    dummy_input_ids = torch.zeros(1, 1, dtype=torch.long)
    dummy_cache_pos = torch.tensor([seq_len - 1], dtype=torch.long)
    example_args    = (dummy_input_ids, dummy_cache_pos)

    # Both tensors have fixed shapes; no dynamic shapes are needed.
    dynamic_shapes = None

    print("[4/6] Running torch.export (this may take several minutes) …")
    exported_program = torch.export.export(
        wrapped,
        args=example_args,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    print("[5/6] Lowering to EdgeIR → CoreML delegate …")
    coreml_compile_specs = CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT16,       # ANE needs fp16 activations
        compute_unit=ct.ComputeUnit.ALL,              # allow CPU + GPU + ANE
        minimum_deployment_target=ct.target.iOS17,   # requires iOS 17 (matches Info.plist)
    )

    edge_program = to_edge_transform_and_lower(
        exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[
            CoreMLPartitioner(compile_specs=coreml_compile_specs),
            XnnpackPartitioner(),  # handles ops not supported by CoreML
        ],
        transform_passes=[ConstrainedDynamicShapesPass()],
    )

    print("[6/6] Serialising to .pte …")
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            passes=[],
            extract_delegate_segments=True,  # store CoreML blob inline
        )
    )

    pte_path = output_dir / "gemma3_4b_int4_coreml.pte"
    with open(pte_path, "wb") as f:
        f.write(executorch_program.buffer)
    print(f"    Saved: {pte_path}  ({pte_path.stat().st_size / 1e6:.1f} MB)")

    # Copy the SentencePiece tokenizer.model file.
    # AutoTokenizer may return GemmaTokenizerFast which lacks vocab_file;
    # save_pretrained + glob handles both slow and fast tokenizer variants.
    _tmp = output_dir / "_tokenizer_tmp"
    tokenizer.save_pretrained(str(_tmp))
    sp_candidates = list(_tmp.glob("*.model"))
    if not sp_candidates:
        raise FileNotFoundError(
            "tokenizer.model not found after save_pretrained — "
            "ensure sentencepiece is installed"
        )
    sp_model_dst = output_dir / "tokenizer.model"
    shutil.copy(sp_candidates[0], sp_model_dst)
    shutil.rmtree(_tmp)
    print(f"    Saved: {sp_model_dst}")

    print("\nDone.  Add both artefacts to Xcode under 'Copy Bundle Resources'.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Gemma-3-4B-Instruct to ExecuTorch .pte with INT4 + KV-cache + CoreML"
    )
    parser.add_argument(
        "--model_id",
        default="google/gemma-3-4b-it",
        help="HuggingFace model ID (default: google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        help="Directory to write gemma3_4b_int4_coreml.pte and tokenizer.model",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Initial cache position used for tracing the decode step (default: 512)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_gemma3(
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        seq_len=args.seq_len,
    )
