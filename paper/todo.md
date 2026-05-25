# Writing TODO

## Before Writing Results

1. Run the 1% smoke baseline over `configs/experiments/smoke_single_gpu_3090.yaml`.
2. Run one QLoRA smoke test with `qwen3_4b` on `domain_qa_smoke`.
3. Freeze the formal benchmark suite after smoke validation.
4. Export all raw formal metrics into CSV files.
5. Prepare one domain dataset description table.

## Figures to Produce

1. Overall task score comparison
2. Peak VRAM comparison
3. Throughput comparison
4. Score-versus-latency tradeoff
5. QLoRA gain on domain benchmark

## Tables to Produce

1. Model overview table
2. Main benchmark table
3. Efficiency table
4. QLoRA training configuration table
5. Pre/post adaptation comparison table
