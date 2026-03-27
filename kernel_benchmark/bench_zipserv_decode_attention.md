# bench_zipserv_decode_attention

`bench_zipserv_decode_attention.py` now runs a batch-size sweep only.

- `kv_len` is configurable via `--kv_len` and defaults to `1020`
- supported modes are only `flashattn`, `flashinfer`, `staged_flashattn`, `staged_flashinfer`, `fused_flashattn`
- output rows vary by `batch_size`

## Environment

Use the `zipserv` conda environment on this machine.

Required:

- `torch` with CUDA
- `libL_API.so` built under `~/ZipServ_ASPLOS26/build`
- prebuilt `zipserv` extension from `build_zipserv_decode_attention_extensions.py --target zipserv`

Optional:

- `flash-attn`
- `flashinfer-python`
- vendored `flash_attn_v283` source tree under `kernel_benchmark/third_party` for `fused_flashattn`

## Prebuilt Extensions

`bench_zipserv_decode_attention.py` no longer auto-builds extensions.

- If a required `.so` is missing, the benchmark fails immediately.
- Prebuilt extension directories are ABI-specific under `kernel_benchmark/.prebuilt_extensions/<ABI_TAG>/...`.
- If you switch Python environments, check the ABI tag again before assuming an earlier build is reusable.

Check the current ABI tag:

```bash
python - <<'PY'
import sys
print(sys.implementation.cache_tag)
PY
```

## Build

Build the base ZipServ extension required by all modes:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
TORCH_CUDA_ARCH_LIST=8.6 python build_zipserv_decode_attention_extensions.py --target zipserv --verbose
```

Build the regular-only integrated FlashAttention extension used by the default `fused_flashattn` path:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
TORCH_CUDA_ARCH_LIST=8.6 python build_zipserv_decode_attention_extensions.py --target zipserv_flashattn --verbose
```

Build the split-kv integrated FlashAttention extension only if you plan to run `--fused_splitkv`:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
TORCH_CUDA_ARCH_LIST=8.6 python build_zipserv_decode_attention_extensions.py --target zipserv_flashattn_splitkv --verbose
```

If a long FlashAttention build is interrupted, resume it with `ninja` in the same ABI-specific build directory:

```bash
ABI_TAG=$(python - <<'PY'
import sys
print(sys.implementation.cache_tag)
PY
)

cd ~/ZipServ_ASPLOS26/kernel_benchmark/.prebuilt_extensions/$ABI_TAG/flash_attn_2_cuda_regular_only
ninja -v flash_attn_2_cuda.so
```

Use `flash_attn_2_cuda` instead of `flash_attn_2_cuda_regular_only` in the path above if you are resuming the split-kv build.

## Run

All supported modes together:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
python bench_zipserv_decode_attention.py \
  --layer 0 \
  --kv_len 1020 \
  --modes flashattn,flashinfer,staged_flashattn,staged_flashinfer,fused_flashattn \
  --batch_sizes 1,2,4,8,16,32,64,128,256,512 \
  --warmup 10 \
  --iters 100 \
  --out_csv zipserv_decode_attention_batch_sweep.csv
```

`fused_flashattn` only:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
python bench_zipserv_decode_attention.py \
  --layer 0 \
  --kv_len 1020 \
  --modes fused_flashattn \
  --batch_sizes 1,2,4,8,16,32,64,128,256,512 \
  --warmup 10 \
  --iters 100 \
  --out_csv zipserv_decode_attention_fused_paths.csv
```

`staged_flashattn` safe sweep:

```bash
source ~/ls/etc/profile.d/conda.sh
conda activate zipserv
cd ~/ZipServ_ASPLOS26/kernel_benchmark
python bench_zipserv_decode_attention.py \
  --layer 0 \
  --kv_len 1020 \
  --modes staged_flashattn \
  --batch_sizes 1,2,4,8,16,32,64,128,256,511 \
  --warmup 10 \
  --iters 100 \
  --out_csv zipserv_decode_attention_staged_flashattn_b1to511.csv
```

## Modes

- `flashattn`: dense K/V를 그대로 stock `flash_attn_with_kvcache`에 넣는 baseline
- `flashinfer`: dense K/V를 FlashInfer paged-KV cache로 materialize한 뒤 batch decode wrapper로 실행하는 baseline
- `staged_flashattn`: ZipServ compressed K/V를 매 iteration마다 reusable dense workspace로 decompress한 뒤 `flash_attn_with_kvcache`에 넣는 경로
- `staged_flashinfer`: ZipServ compressed K/V를 매 iteration마다 reusable paged-KV workspace로 decompress한 뒤 FlashInfer batch decode wrapper에 넣는 경로
- `fused_flashattn`: 수정된 원본 FlashAttention 경로로 직접 들어가 `zipserv_k`, `zipserv_v`, `zipserv_num_heads_k` metadata를 넘겨 split-kv tile load 시점에만 decompress하는 경로

## Notes

- decode-only workload: `q_len = 1`
- `Q` is generated from the real Llama 3.1 70B `q_proj.weight`
- `K/V` come from the sorted KV cache pair `2 * layer`, `2 * layer + 1`
- the benchmark checks for prebuilt extensions but does not build them for you
- 출력 CSV 열은 `layer, mode, batch_size, kv_len, q_heads, kv_heads, head_dim, latency_ms, base_path, status` 이다
- `base_path`는 `latency_ms` 대비 같은 계열의 direct baseline 비율이다
- `flashattn`, `flashinfer`는 각자 direct baseline이므로 `base_path = 1`이다
- `flashinfer` paths use page size `16`
- `fused_flashattn` currently requires `num_kv_heads == 8`
- `fused_flashattn`는 기본적으로 regular non-split FlashAttention path를 사용한다. 예전 split-kv 경로 비교가 필요할 때만 `--fused_splitkv`를 넣는다.
- `staged_flashattn`는 `batch_size=512`에서 현재 실패한다. 직접 원인은 총 VRAM 부족이라기보다, decompressor `BF16TripleBitmap_Decompress_API()`의 `GridDim.y = rows / 64`가 `65536`이 되어 `cc 8.6` 장치의 최대 grid `y` 차원 `65535`를 넘기기 때문이다. 같은 설정에서 `batch_size=511`은 `GridDim.y = 65408`으로 정상 동작한다.
- dense torch reference는 사용하지 않는다. 정확도는 reference 제거 전에 batch `1..256`에서 사전 확인했다.
