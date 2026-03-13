# bench_zipserv_decode_attention

`bench_zipserv_decode_attention.py` benchmarks GPU-only decode attention for a single layer using:

- real Llama 3.1 70B `q_proj.weight`
- KV cache from `~/saved_kv_cache`
- token-length sweep over `kv_len`
- backend: `flashinfer` baseline for staged paths plus staged `flash_attn` baseline for ZipServ FlashAttention variants
- modes: `dense`, `staged`, `staged_reuse`, `zipserv_native`, `zipserv_flashattn`, `zipserv_flashattn_paged`, `zipserv_flashattn_fused`

## Environment

Use the `zipserv` conda environment on this machine.

Required:

- `torch` with CUDA
- `libL_API.so` built under `~/ZipServ_ASPLOS26/build`

Optional for external fused baselines:

- `flash-attn`
- `flashinfer-python`
- vendored `flash_attn_v283` source tree under `kernel_benchmark/third_party`

Current working environment:

- conda env: `zipserv`
- Python: `3.12`
- `torch`: `2.9.1+cu128`
- `flash-attn`: `2.8.3`
- `flashinfer-python`: `0.6.5`

## Run
Optional prebuild step (`sm_86` only):

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
TORCH_CUDA_ARCH_LIST=8.6 python build_zipserv_decode_attention_extensions.py --target all --verbose
```

`zipserv_native` only:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
python bench_zipserv_decode_attention.py \
  --layer 0 \
  --modes zipserv_native \
  --token_counts 1,2,4,8,16,32,64,128,256,512,1024,1535 \
  --warmup 10 \
  --iters 100 \
  --out_csv zipserv_decode_attention_native.csv
```

`flashinfer` staged baseline:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
python bench_zipserv_decode_attention.py \
  --layer 0 \
  --backend all \
  --modes staged_reuse \
  --token_counts 1,2,4,8,16,32,64,128,256,512,1024,1535 \
  --warmup 10 \
  --iters 100 \
  --out_csv zipserv_decode_attention_flashinfer.csv
```

`zipserv_flashattn` hybrid FlashAttention path:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
python bench_zipserv_decode_attention.py \
  --layer 0 \
  --modes zipserv_flashattn \
  --token_counts 1,2,4,8,16,32,64,128,256,512,1024,1535 \
  --warmup 10 \
  --iters 100 \
  --out_csv zipserv_decode_attention_flashattn.csv
```

explicit paged FlashAttention handoff path:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
python bench_zipserv_decode_attention.py \
  --layer 0 \
  --modes zipserv_flashattn_paged \
  --token_counts 1,2,4,8,16,32,64,128,256,512,1024,1535 \
  --warmup 10 \
  --iters 100 \
  --out_csv zipserv_decode_attention_flashattn_paged.csv
```

legacy vendored fused `flash_attn_ck` path:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
python bench_zipserv_decode_attention.py \
  --layer 0 \
  --modes zipserv_flashattn_fused \
  --token_counts 1,2,4,8,16,32,64,128,256,512,1024,1535 \
  --warmup 10 \
  --iters 100 \
  --out_csv zipserv_decode_attention_flashattn_fused.csv
```

## Notes

- decode-only workload: `q_len = 1`
- `Q` is generated from the real `q_proj.weight` and a deterministic synthetic hidden state
- `K/V` come from the sorted KV cache pair `2 * layer`, `2 * layer + 1`
- ZipServ compression is performed offline before timing
- `dense`: already-materialized dense K/V on GPU를 그대로 attention backend에 넣는 경로
- `staged`: 매 iteration마다 ZipServ K/V를 새 dense buffer로 decompress한 뒤 attention backend에 넣는 경로
- `staged_reuse`: `staged`와 동일하지만 dense K/V output buffer를 재사용해서 per-iteration allocation overhead를 제거한 경로
- `zipserv_native`: compressed K/V를 shared memory로 가져와 score 계산, row softmax, value accumulation을 수행하는 ZipServ 전용 decode-attention 경로. dense K/V materialization이 없다.
- `zipserv_flashattn`: 기본 hybrid 정책 경로. 현재 측정 기준으로 `kv_len <= 256`에서는 vendored fused `flash_attn_ck` 경로를 사용하고, `kv_len > 256`에서는 paged-KV handoff 경로를 사용한다.
- `zipserv_flashattn_paged`: ZipServ K/V를 FlashAttention paged-KV layout에 맞는 dense reusable workspace로 decompress한 뒤 `flash_attn_with_kvcache(..., block_table=...)`로 handoff하는 명시적 paged 경로다.
- `zipserv_flashattn_fused`: vendored `flash_attn_ck` 쪽 standalone fused decode kernel이 ZipServ compressed K/V를 직접 읽는 legacy 경로. compressed global tile load -> shared-memory decompress -> online softmax -> V accumulation 순서로 처리하며 dense K/V materialization이 없다.
- `zipserv_native`는 내부적으로 dense torch reference를 baseline으로 사용해 `base/path`, `max_abs_err`, `mean_abs_err`를 계산한다.
- `zipserv_flashattn`, `zipserv_flashattn_paged`, `zipserv_flashattn_fused`는 내부적으로 측정한 `staged_reuse + flash_attn` latency를 `base/path` 기준으로 사용하고, `max_abs_err`, `mean_abs_err`는 기존처럼 dense torch reference 기준으로 기록한다.
- 현재 `zipserv_native` 제약:
  `num_kv_heads == 8`
  `head_dim <= 256`
- 현재 `zipserv_flashattn` 제약:
  `flash-attn` import 가능
  short-range hybrid에서 fused 경로를 쓰려면 `num_kv_heads == 8`
  paged fallback은 block size `256`
- 현재 `zipserv_flashattn_paged` 제약:
  `flash-attn` import 가능
  paged KV block size `256`
- 현재 `zipserv_flashattn_fused` 제약:
  `bf16` only
  `batch_size == 1`
  `q_len == 1`
  `num_kv_heads == 8`
  `head_dim <= 256`
- CSV time metric is only `latency_ms`; staged component timings and compression-ratio metrics are intentionally omitted
- first run may spend extra time compiling/loading the local CUDA extensions; the local defaults now target `TORCH_CUDA_ARCH_LIST=8.6`
- verified smoke runs completed for `zipserv_native`, `zipserv_flashattn`, and `zipserv_flashattn_fused` on this machine
