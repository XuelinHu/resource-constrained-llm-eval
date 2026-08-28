# Qwen3.8 Compatibility Record

Date: 2026-08-27

## Selected Model

- Official model: `Qwen/Qwen3.8-27B`
- Quantized distribution: `unsloth/Qwen3.8-27B-GGUF`
- Quantization: `UD-Q4_K_M`
- Local file: `/ds2/workspace/ai/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`
- File size: `16,464,440,224` bytes
- SHA-256: `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`

## Hardware Decision

The latest `Qwen3.8-Flash-Next` release is not the suitable single-GPU artifact for this workstation. Its official BF16 distribution is approximately 335 GiB and its sparse activation design does not remove the need to store the full model weights. The workstation has one RTX 3090 with 24,576 MiB VRAM and 32 GiB host memory. The 27B Q4 GGUF is therefore used as the runnable Qwen3.8-family candidate.

## Smoke Test

The model was loaded with the CUDA build of `llama.cpp` at 2,048 context tokens and generated a coherent Chinese response. Return code was `0`; the recorded process-level peak GPU usage was `23,698 MiB` while another local process was present. The model's own loaded allocation was approximately `14.9 GiB`. Generation timing reported by `llama.cpp` was approximately `38.9 tokens/s`.

This is a compatibility check only. The model is registered as `post_freeze_candidate` and is excluded from the frozen five-generator IJWIS results. A formal comparison requires rerunning the same no-retrieval, BM25-RAG, approved-hybrid-RAG, bilingual QA, citation, translation and efficiency protocol.

The machine-readable record is `results/model_compatibility/qwen3_8_27b_gguf.json`.
