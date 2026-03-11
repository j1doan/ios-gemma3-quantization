#!/usr/bin/env python3
"""
export_gemma3.py

Exports Google Gemma-3-4B-Instruct to ExecuTorch .pte files with:
  - INT4 weight-only quantization (groupwise, group_size=128) — halves the
    weight footprint vs INT8; ~2.0 GB weights, fitting within the 6 GB
    unified memory of A16 Bionic devices alongside the KV-cache (~272 MB fp16).
  - KV-cache enabled (static shape, max_cache_len = 2048 tokens) so that
    each decode step only processes one new token instead of the full prefix.
  - CoreML backend delegation for Apple Neural Engine acceleration.
  - SentencePiece tokenizer copied alongside the model.

  In --multimodal mode, additionally exports:
  - A separate vision encoder .pte (SigLIP + multi-modal projector, fp16).
  - The text decoder wrapper accepts inputs_embeds + embed_mask so the C++
    runtime can inject projected image tokens during prefill.

Usage:
    pip install -r requirements.txt
    # Text-only
    python export_gemma3.py --output_dir ./output
    # Multimodal (includes vision encoder)
    python export_gemma3.py --output_dir ./output --multimodal

The produced artefacts (text-only):
    output/gemma3_4b_int4_coreml.pte   — ExecuTorch model bundle (~2.0 GB INT4)
    output/tokenizer.model             — SentencePiece vocabulary

Additional multimodal artefacts:
    output/gemma3_vision_encoder.pte   — Vision encoder + projector (~300 MB fp16)

Add all files to your Xcode project under "Copy Bundle Resources".
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
# Gemma-3 architecture constants (Section 1 of edge spec)
# ---------------------------------------------------------------------------

HIDDEN_DIM:      int = 3072
N_IMAGE_TOKENS:  int = 256    # projected patches from SigLIP → decoder space
VISION_INPUT_RES: int = 896   # static input resolution (896 × 896)

# ---------------------------------------------------------------------------
# KV-cache constants
# ---------------------------------------------------------------------------

# Maximum total context length for the static KV-cache.
# Bumping this grows the cache tensors but not the weight bundle.
MAX_CACHE_LEN: int = 2048


# ---------------------------------------------------------------------------
# Wrappers — expose a pure torch.export-compatible forward signature
# ---------------------------------------------------------------------------

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
            dtype=torch.float16,  # fp16: matches ANE native compute, saves ~272 MB vs fp32
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


class Gemma3MultimodalExportWrapper(torch.nn.Module):
    """
    Wraps Gemma3ForCausalLM with a 4-input forward signature that supports
    injecting pre-computed vision embeddings during prefill.

    Forward signature:

        (input_ids      [1, 1]         int64,
         cache_position [1]            int64,
         inputs_embeds  [1, 1, H]      float16,
         embed_mask     [1, 1, 1]      float16)  — 1.0 to use inputs_embeds,
                                                    0.0 to use token embedding
        -> logits [1, 1, vocab_size]   float32

    When embed_mask is 0.0, the model looks up input_ids in the embedding
    table (normal text decode).  When embed_mask is 1.0, the model uses
    inputs_embeds directly (vision token injection during prefill).

    This avoids dynamic control flow: the mixing is a simple linear blend:
        combined = embed_mask * inputs_embeds + (1 - embed_mask) * token_embeds
    All shapes are static.
    """

    def __init__(self, model: torch.nn.Module, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.model = model
        self.embed_tokens = model.get_input_embeddings()
        self.hidden_dim = hidden_dim
        self.static_cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=MAX_CACHE_LEN,
            device="cpu",
            dtype=torch.float16,
        )

    def forward(
        self,
        input_ids: torch.Tensor,        # [1, 1]      int64
        cache_position: torch.Tensor,   # [1]         int64
        inputs_embeds: torch.Tensor,    # [1, 1, H]   float16
        embed_mask: torch.Tensor,       # [1, 1, 1]   float16  (0.0 or 1.0)
    ) -> torch.Tensor:
        """Returns logits [1, 1, vocab_size]."""
        token_embeds = self.embed_tokens(input_ids)  # [1, 1, H]
        # AOT-friendly blend — no if/else on tensor values
        combined = embed_mask * inputs_embeds + (1.0 - embed_mask) * token_embeds
        outputs = self.model(
            inputs_embeds=combined,
            cache_position=cache_position,
            past_key_values=self.static_cache,
            use_cache=True,
        )
        return outputs.logits


class Gemma3VisionEncoderWrapper(torch.nn.Module):
    """
    Wraps the SigLIP vision tower + multi-modal projector from the full
    Gemma3ForConditionalGeneration model.

    Forward signature:

        (pixel_values [1, 3, 896, 896] float32)
        -> projected_tokens [1, N_IMAGE_TOKENS, HIDDEN_DIM] float16

    All shapes are static — no dynamic image resizing inside the graph.
    Preprocessing (resize, normalize) happens outside the graph on the
    device CPU / GPU before calling this encoder.
    """

    def __init__(self, vision_tower: torch.nn.Module,
                 multi_modal_projector: torch.nn.Module):
        super().__init__()
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [1, 3, 896, 896] → [1, N, H] float16."""
        vision_outputs = self.vision_tower(pixel_values)
        # SigLIP returns last_hidden_state [1, n_patches, vision_hidden]
        image_features = vision_outputs.last_hidden_state
        projected = self.multi_modal_projector(image_features)
        return projected.to(torch.float16)


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
# CoreML compile specs (shared between text decoder and vision encoder)
# ---------------------------------------------------------------------------

def _coreml_compile_specs():
    return CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT16,       # ANE needs fp16 activations
        compute_unit=ct.ComputeUnit.ALL,              # allow CPU + GPU + ANE
        minimum_deployment_target=ct.target.iOS17,   # requires iOS 17 (matches Info.plist)
    )


def _lower_and_serialize(exported_program, output_path: Path, label: str) -> None:
    """Lower an exported program to EdgeIR → CoreML and serialize to .pte."""
    edge_program = to_edge_transform_and_lower(
        exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[
            CoreMLPartitioner(compile_specs=_coreml_compile_specs()),
            XnnpackPartitioner(),  # handles ops not supported by CoreML
        ],
        transform_passes=[ConstrainedDynamicShapesPass()],
    )

    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            passes=[],
            extract_delegate_segments=True,  # store CoreML blob inline
        )
    )

    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)
    print(f"    Saved: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Vision encoder export (multimodal only)
# ---------------------------------------------------------------------------

def export_vision_encoder(full_model, output_dir: Path) -> None:
    """
    Extracts the SigLIP vision tower + multi-modal projector from the full
    Gemma3ForConditionalGeneration model and exports as a standalone .pte.
    """
    print("[V1] Extracting vision encoder + projector …")
    vision_tower = full_model.vision_tower
    multi_modal_projector = full_model.multi_modal_projector

    vision_wrapper = Gemma3VisionEncoderWrapper(vision_tower, multi_modal_projector)
    vision_wrapper.eval()

    # Static input: [1, 3, 896, 896] — no dynamic resizing
    dummy_pixels = torch.randn(1, 3, VISION_INPUT_RES, VISION_INPUT_RES)

    print("[V2] Running torch.export for vision encoder …")
    exported = torch.export.export(
        vision_wrapper,
        args=(dummy_pixels,),
        dynamic_shapes=None,
        strict=False,
    )

    print("[V3] Lowering vision encoder to CoreML …")
    pte_path = output_dir / "gemma3_vision_encoder.pte"
    _lower_and_serialize(exported, pte_path, "vision encoder")


# ---------------------------------------------------------------------------
# Main export routine
# ---------------------------------------------------------------------------

def export_gemma3(model_id: str, output_dir: Path, seq_len: int = 512,
                  multimodal: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if multimodal:
        _export_multimodal(model_id, output_dir, seq_len)
    else:
        _export_text_only(model_id, output_dir, seq_len)

    print("\nDone.  Add all artefacts to Xcode under 'Copy Bundle Resources'.")


def _export_text_only(model_id: str, output_dir: Path, seq_len: int) -> None:
    """Original text-only export path — 2-input decoder."""
    print(f"[1/6] Loading {model_id} (text-only) in float32 …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    hf_model.eval()

    print("[2/6] Applying INT4 weight-only quantisation …")
    apply_int4_weight_quantization(hf_model)

    print("[3/6] Wrapping model for export (StaticCache + KV-cache) …")
    wrapped = Gemma3ExportWrapper(hf_model)

    dummy_input_ids = torch.zeros(1, 1, dtype=torch.long)
    dummy_cache_pos = torch.tensor([seq_len - 1], dtype=torch.long)

    print("[4/6] Running torch.export …")
    exported_program = torch.export.export(
        wrapped, args=(dummy_input_ids, dummy_cache_pos),
        dynamic_shapes=None, strict=False,
    )

    print("[5/6] Lowering to EdgeIR → CoreML delegate …")
    pte_path = output_dir / "gemma3_4b_int4_coreml.pte"
    _lower_and_serialize(exported_program, pte_path, "text decoder")

    print("[6/6] Copying tokenizer …")
    _copy_tokenizer(tokenizer, output_dir)


def _export_multimodal(model_id: str, output_dir: Path, seq_len: int) -> None:
    """Multimodal export — 4-input decoder + separate vision encoder .pte."""
    from transformers import AutoModelForImageTextToText

    print(f"[1/8] Loading {model_id} (multimodal) in float32 …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    full_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    full_model.eval()

    # ---- Vision encoder (fp16, no INT4) ------------------------------------
    print("[2/8] Exporting vision encoder …")
    export_vision_encoder(full_model, output_dir)

    # ---- Text decoder (INT4) -----------------------------------------------
    # Extract the language model component for separate quantization + export
    language_model = full_model.language_model

    print("[3/8] Applying INT4 weight-only quantisation to text decoder …")
    apply_int4_weight_quantization(language_model)

    print("[4/8] Wrapping text decoder for multimodal export (4 inputs) …")
    wrapped = Gemma3MultimodalExportWrapper(language_model, hidden_dim=HIDDEN_DIM)

    dummy_input_ids    = torch.zeros(1, 1, dtype=torch.long)
    dummy_cache_pos    = torch.tensor([seq_len - 1], dtype=torch.long)
    dummy_embeds       = torch.zeros(1, 1, HIDDEN_DIM, dtype=torch.float16)
    dummy_embed_mask   = torch.zeros(1, 1, 1, dtype=torch.float16)

    print("[5/8] Running torch.export for text decoder …")
    exported_program = torch.export.export(
        wrapped,
        args=(dummy_input_ids, dummy_cache_pos, dummy_embeds, dummy_embed_mask),
        dynamic_shapes=None, strict=False,
    )

    print("[6/8] Lowering text decoder to CoreML …")
    pte_path = output_dir / "gemma3_4b_int4_coreml.pte"
    _lower_and_serialize(exported_program, pte_path, "text decoder (multimodal)")

    print("[7/8] Copying tokenizer …")
    _copy_tokenizer(tokenizer, output_dir)

    print("[8/8] Export complete.")


# ---------------------------------------------------------------------------
# Tokenizer copy helper
# ---------------------------------------------------------------------------

def _copy_tokenizer(tokenizer, output_dir: Path) -> None:
    """Copy the SentencePiece tokenizer.model alongside the .pte artefacts."""
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
        help="Directory to write .pte and tokenizer.model artefacts",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Initial cache position used for tracing the decode step (default: 512)",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Export with vision encoder + 4-input text decoder for image+text inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_gemma3(
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        seq_len=args.seq_len,
        multimodal=args.multimodal,
    )
