# bench_zipserv_decode_attention

`bench_zipserv_decode_attention.py` benchmarks GPU-only decode attention for a single layer using:

- real Llama 3.1 70B `q_proj.weight`
- KV cache from `~/saved_kv_cache`
- token-length sweep over `kv_len`
- backends: `flashinfer`, `flash-attn`, `zipserv_native`
- modes: `dense`, `staged`, `staged_reuse`, `zipserv_native`

## Environment

Use the `zipserv` conda environment on this machine.

Required:

- `torch` with CUDA
- `libL_API.so` built under `~/ZipServ_ASPLOS26/build`

Optional for external fused baselines:

- `flashinfer-python`
- `flash-attn`

Current working environment:

- conda env: `zipserv`
- Python: `3.12`
- `torch`: `2.9.1+cu128`
- `flash-attn`: `2.8.3`
- `flashinfer-python`: `0.6.5`

## Run
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

`flashinfer` / `flash-attn` 비교 경로:

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

`flash-attn` run:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
python bench_zipserv_decode_attention.py \
  --layer 0 \
  --backend flashattn \
  --modes dense,staged,staged_reuse \
  --token_counts 1,16,128,512,1024,1535 \
  --warmup 10 \
  --iters 100 \
  --out_csv zipserv_decode_attention_flashattn.csv
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
- `zipserv_native`는 내부적으로 dense torch reference를 baseline으로 사용해 `base/path`, `max_abs_err`, `mean_abs_err`를 계산한다.
- 현재 `zipserv_native` 제약:
  `num_kv_heads == 8`
  `head_dim <= 256`
- CSV time metric is only `latency_ms`; staged component timings and compression-ratio metrics are intentionally omitted
- first run may spend extra time compiling/loading the local CUDA extension
- verified smoke runs completed for `zipserv_native` (`kv_len=1`) and for the existing `flashinfer` / `flash-attn` paths on this machine
