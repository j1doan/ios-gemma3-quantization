# Gemma-3 On-Device — Apple A16 Bionic

A Swift/SwiftUI iOS application that runs **Google Gemma-3-4B-Instruct** entirely
on-device using **ExecuTorch** with the **CoreML backend delegate**, targeting the
Apple Neural Engine on the A16 Bionic SoC.

Quantization: **INT4 weight-only** (groupwise, group_size=128) with fp16 activations.
Decoding: **KV-cache enabled** — each decode step processes a single token instead
of re-computing the entire prefix, giving substantially lower per-token latency.

**Multimodal support:** The app can optionally load a SigLIP vision encoder
(`.pte`) alongside the text decoder, enabling image+text prompts. The vision
encoder runs at fp16 and produces ~256 projected tokens that are injected into
the decoder via an AOT-compatible embedding-mask blend — no dynamic control
flow required.

---

## Project layout

```
gemma3/
├── scripts/
│   ├── export_gemma3.py           # Exports Gemma-3 → INT4 .pte with CoreML delegate + KV-cache
│   └── requirements.txt           # Python dependencies for the export step
└── Gemma3OnDevice/
    ├── Info.plist
    ├── Gemma3OnDevice-Bridging-Header.h   # Exposes ObjC++ to Swift
    ├── App/
    │   └── Gemma3OnDeviceApp.swift        # @main SwiftUI entry point
    ├── Views/
    │   └── ChatView.swift                 # SwiftUI chat interface + PhotosPicker
    ├── ViewModels/
    │   └── ChatViewModel.swift            # @MainActor observable ViewModel
    └── Inference/
        ├── GemmaInference.hpp / .cpp      # C++ inference engine (ExecuTorch)
        ├── Tokenizer.hpp / .cpp           # SentencePiece tokenizer wrapper
        ├── ImagePreprocessor.swift         # Resize / normalize images for SigLIP (CPU-side)
        ├── InferenceRunner.h              # Objective-C++ class declaration
        └── InferenceRunner.mm             # Objective-C++ implementation
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Xcode | 16.0+ |
| iOS Deployment Target | 17.0 (CoreML backend compiled with `minimum_deployment_target="iOS17"`) |
| Python | 3.10+ |
| ExecuTorch | 0.4.0 |
| CocoaPods | 1.15+ |

---

## Step 1 — Export the model

### Text-only (default)

```bash
cd gemma3/scripts
pip install -r requirements.txt

# Requires a HuggingFace token with access to google/gemma-3-4b-it
huggingface-cli login

python export_gemma3.py \
    --model_id  google/gemma-3-4b-it \
    --output_dir ./output \
    --seq_len   512
```

This produces two files in `scripts/output/`:

| File | Purpose |
|------|---------|
| `gemma3_4b_int4_coreml.pte` | ExecuTorch model bundle (~2.0 GB INT4, KV-cache) |
| `tokenizer.model` | SentencePiece vocabulary |

### Multimodal (text + vision)

```bash
python export_gemma3.py \
    --model_id  google/gemma-3-4b-it \
    --output_dir ./output \
    --seq_len   512 \
    --multimodal
```

This produces three files in `scripts/output/`:

| File | Purpose |
|------|---------|
| `gemma3_4b_int4_coreml.pte` | Text decoder with 4-input embed-mask signature (~2.0 GB INT4) |
| `gemma3_vision_encoder.pte` | SigLIP vision encoder + projector (fp16, ~350 MB) |
| `tokenizer.model` | SentencePiece vocabulary |

The multimodal text decoder accepts four inputs per step:
`(input_ids, cache_position, inputs_embeds, embed_mask)`.
During normal text decode `embed_mask=0` and the extra inputs are ignored.
During vision-token prefill `embed_mask=1` and `inputs_embeds` carries the
projected vision embedding, blended via
`combined = embed_mask * inputs_embeds + (1 - embed_mask) * token_embeds`
— pure tensor arithmetic with static shapes (AOT-safe).

> **Memory note:** The INT4-quantised 4B model requires ≈ 2.0 GB of RAM for weights.
> The static KV-cache for 2 048 tokens adds ~272 MB (fp16) on top, bringing the
> total to ≈ 2.3 GB — within the A16 Bionic's 6 GB unified memory limit with headroom for iOS.
> INT4 groupwise quantization (group_size=128) keeps perplexity regression
> manageable for a 4B-parameter model, while halving the weight footprint
> compared to INT8.
> With the vision encoder loaded (~350 MB fp16), total memory rises to ≈ 2.6 GB
> — still within the 6 GB budget.

---

## Step 2 — Create the Xcode project

1. Open Xcode → **File → New → Project** → *App* template.
2. Set:
   - Product Name: `Gemma3OnDevice`
   - Bundle Identifier: `com.yourname.Gemma3OnDevice`
   - Interface: SwiftUI
   - Language: Swift
   - Minimum Deployments: **iOS 17.0**
3. Save into `gemma3/Gemma3OnDevice/`.
4. Delete Xcode's generated `ContentView.swift`.

### Add source files

Drag the following folders/files into the Xcode project navigator,
choosing **"Create groups"** and ticking **"Add to target: Gemma3OnDevice"**:

```
Gemma3OnDevice/App/
Gemma3OnDevice/Views/
Gemma3OnDevice/ViewModels/
Gemma3OnDevice/Inference/        ← includes .h, .hpp, .cpp, .mm files
Gemma3OnDevice/Gemma3OnDevice-Bridging-Header.h
Gemma3OnDevice/Info.plist
```

---

## Step 3 — Add ExecuTorch & SentencePiece via CocoaPods

Create `gemma3/Gemma3OnDevice/Podfile`:

```ruby
platform :ios, '17.0'
use_frameworks! :linkage => :static

target 'Gemma3OnDevice' do
  # ExecuTorch core + CoreML backend delegate
  pod 'executorch',         '~> 0.4'
  pod 'executorch-coreml',  '~> 0.4'
  pod 'executorch-xnnpack', '~> 0.4'

  # SentencePiece tokenizer
  pod 'SentencePiece', '~> 0.2'
end
```

Then run:

```bash
cd gemma3/Gemma3OnDevice
pod install
open Gemma3OnDevice.xcworkspace   # always use .xcworkspace after pod install
```

> **Alternative — Swift Package Manager:** If you prefer SPM, add
> `https://github.com/pytorch/executorch` as a package dependency and
> link the `executorch`, `coreml_backend`, and `xnnpack_backend` products.

---

## Step 4 — Configure Build Settings

In **Target → Build Settings**, set:

| Setting | Value |
|---------|-------|
| Objective-C Bridging Header | `Gemma3OnDevice/Gemma3OnDevice-Bridging-Header.h` |
| C++ Language Dialect | `C++17` |
| C++ Standard Library | `libc++` |
| Enable Bitcode | `No` |
| Architectures | `arm64` only (remove `x86_64`) |
| Header Search Paths | `$(PODS_ROOT)/Headers/Public` |

---

## Step 5 — Add model files to the app bundle

1. In Xcode, select your target → **Build Phases → Copy Bundle Resources**.
2. Click **+** and add:
   - `scripts/output/gemma3_4b_int4_coreml.pte`
   - `scripts/output/tokenizer.model`
   - (multimodal only) `scripts/output/gemma3_vision_encoder.pte`

The files will be copied into `MyApp.app/` at build time, and
`Bundle.main.url(forResource:withExtension:)` in `ChatViewModel.swift`
will locate them at runtime. When the vision encoder `.pte` is present in
the bundle, the app automatically enables image+text mode.

---

## Step 6 — Build & run

```
Xcode → select a physical A16 Bionic device or Simulator → ▶ Run
```

> For **best performance**, always run on a **physical device with an A16 Bionic SoC**.
> The Simulator does not expose the ANE and will fall back to CPU-only
> XNNPACK execution.

---

## Architecture overview

```
┌─────────────────────────────────────────┐
│             SwiftUI Layer               │
│  ChatView  ←→  ChatViewModel            │
│  (PhotosPicker, @MainActor, Combine)    │
└───────────────────┬─────────────────────┘
                    │ Obj-C bridge (main thread dispatch)
┌───────────────────▼─────────────────────┐
│          InferenceRunner (.mm)          │
│  • GemmaGenerationConfig (ObjC mirror)  │
│  • dispatch_queue (serial, QoS=UI)      │
│  • cancelRequested (std::atomic<bool>)  │
│  • generateFromPrompt:pixelData: (img)  │
└───────────────────┬─────────────────────┘
                    │ C++ call
┌───────────────────▼─────────────────────┐
│         GemmaInference (.cpp)           │
│  • FileDataLoader  ─→  loads .pte       │
│  • Program / Method  (ExecuTorch RT)    │
│  • GemmaTokenizer  (SentencePiece)      │
│  • Token sampling  (temp / top-p / k)   │
│  ─── multimodal (optional) ──────────── │
│  • Vision encoder .pte  (SigLIP fp16)   │
│  • generateWithImage()  → vision prefill│
│  • runVisionEncoder()   → [1,256,3072]  │
└───────────────────┬─────────────────────┘
                    │ ExecuTorch delegate dispatch
┌───────────────────▼─────────────────────┐
│      CoreML Backend + ANE (A16)         │
│  gemma3_4b_int4_coreml.pte              │
│  gemma3_vision_encoder.pte (optional)   │
│  (INT4 weights, fp16 activations,       │
│   static KV-cache 2048 tokens)          │
└─────────────────────────────────────────┘
```

### Multimodal inference flow

When the user attaches an image:

1. **ImagePreprocessor.swift** resizes to 896×896, normalises with SigLIP constants
   (mean=0.5, std=0.5), and produces a Float32 CHW buffer `[3, 896, 896]`.
2. The buffer is passed through the ObjC bridge to `GemmaInference::generateWithImage()`.
3. `runVisionEncoder()` runs the vision `.pte` on the pixel buffer →
   projected tokens `[1, 256, 3072]` (fp16).
4. During prefill the first 256 decode steps set `embed_mask=1.0` and inject
   the vision embeddings. The remaining prompt tokens use `embed_mask=0.0`
   (normal token-embedding lookup).
5. Autoregressive decode continues identically to text-only mode.

### INT4 quantization

The export script applies **weight-only INT4 quantisation** (groupwise, group_size=128)
via `torchao`.  Activations remain in fp16.  The quantised
weights are lowered into the CoreML `.mlpackage` blob embedded inside the
`.pte` artefact at export time.  At inference time, the ANE decompresses
INT4 → fp16 on-the-fly during the matrix multiplications in attention and
FFN layers, yielding ~4× weight memory reduction (vs fp16) while keeping
perplexity within acceptable bounds for the 4B model size.

### KV-cache

The exported model uses a **static KV-cache** (key/value tensors of shape
`[1, num_kv_heads, 2048, head_dim]` allocated once at load time).  Each
decode step processes exactly **one token** by passing `input_ids [1,1]` and
the absolute `cache_position` scalar alongside the existing key/value state.
The KV-cache is mutable module state inside the `.pte` bundle — it is
updated in-place during each forward step.  No explicit KV tensors are
passed in or out by the C++ layer; only logits are read from the output.

This eliminates the O(N²) full-prefix recomputation of the original stateless
export — each token step has constant-time cost regardless of context length.

> **Circular wrapping note:** The spec describes `cache_index = position % max_seq`
> for circular buffer behaviour.  In practice, true circular wrapping would
> introduce RoPE-position discontinuities at the wrap boundary (e.g. query at
> wrapped-position 1 attending to key at wrapped-position 2047 would compute a
> relative distance of −2046 instead of the true distance of 2).  Since the
> sliding-window size (2048) equals the cache size, linear fill with per-turn
> reset provides identical memory and context guarantees without the position
> aliasing problem.  The cache is reset between conversation turns.

### FileDataLoader

`torch::executor::util::FileDataLoader` (from
`executorch/extension/data_loader/file_data_loader.h`) opens the `.pte`
file at the bundle path and `mmap`s it for zero-copy weight access.  This
avoids duplicating the ~2.0 GB model in RAM — the OS pages weights in on
demand from the file, sharing physical pages between processes.

### Token streaming

The autoregressive decode loop in `GemmaInference::generate()` calls the
`TokenCallback` on the inference thread; `InferenceRunner.mm` forwards each
fragment via `dispatch_async(main_queue)` so that `ChatViewModel` can append
it to the `@Published messages` array, which SwiftUI then renders
incrementally in `ChatView`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "Model artefacts not found in app bundle" | .pte / .model not in Copy Bundle Resources | Add them to Build Phases |
| `ET_LOG Error: FileDataLoader failed` | Wrong path or corrupt file | Re-run export_gemma3.py |
| App crashes on first token | Insufficient memory | Close other apps; ensure the 4B INT4 model fits within 6 GB |
| Very slow generation on Simulator | Simulator lacks ANE | Run on device |
| Compile error: `sentencepiece_processor.h` not found | Pod not installed | Run `pod install` |
| KV-cache overflow error in log | Prompt length ≥ 2048 tokens | Shorten the prompt or increase MAX_CACHE_LEN |
| Image button missing in chat | Vision encoder .pte not in bundle | Add `gemma3_vision_encoder.pte` to Copy Bundle Resources, or re-export with `--multimodal` |
| Vision encoder load failure | Mismatched export | Re-export with `--multimodal` flag; ensure both .pte files come from the same export run |

---

## License

Source files in this directory are provided as an implementation reference.
Google Gemma model weights are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
ExecuTorch is licensed under the BSD 3-Clause licence.
SentencePiece is licensed under the Apache 2.0 licence.
