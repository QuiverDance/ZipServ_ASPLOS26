# ZipServ Llama 3.1 70B Benchmark Guide

이 문서는 실제 Llama 3.1 70B BF16 가중치를 사용해 다음을 측정하는 방법을 정리한다.
- `bench_zipserv_gemm_sweep`: layer 0 대상 Uncompressed GEMM vs ZipServ GEMM (tokens sweep)
- `bench_llama31_70b_compress`: layer별(0..79) tensor 단위 compress/decompress latency + 압축률

`bench_zipserv_gemm_sweep`의 현재 동작:
- `--ops`는 개별 tensor(`q/k/v/o/gate/up/down`) 선택에만 적용
- fused GEMM 2종(`qkv_fused`, `gateup_fused`)은 항상 추가 측정

## 1) 모델 다운로드

```bash
cd ~/ZipServ_ASPLOS26
python scripts/download_llama31_70b.py \
  --repo-id meta-llama/Llama-3.1-70B \
  --out-dir ~/models/llama31_70b/hf_snapshot
```

## 2) BF16 weight export + manifest 생성

```bash
cd ~/ZipServ_ASPLOS26
python scripts/export_llama31_70b_weights.py \
  --snapshot-dir ~/models/llama31_70b/hf_snapshot \
  --dtype bf16 \
  --layers 0-79
```

기본 출력:
- weights: `~/models/llama31_70b/weights_bf16/`
- manifest: `~/models/llama31_70b/manifest/weights_manifest.jsonl`

## 3) 빌드

```bash
cd ~/ZipServ_ASPLOS26
source Init.sh
cd build && make
cd ../kernel_benchmark
source test_env
export MY_PATH=~/ZipServ_ASPLOS26
export LD_LIBRARY_PATH=~/ZipServ_ASPLOS26/build:$LD_LIBRARY_PATH
make bench_zipserv_gemm_sweep bench_llama31_70b_compress
```

## 4) 실행 예시 (요청 3종)

### (1) GEMM sweep (layer 0, qkv만, 빠른 스모크)

```bash
cd ~/ZipServ_ASPLOS26/kernel_benchmark
./bench_zipserv_gemm_sweep \
  --manifest ~/models/llama31_70b/manifest/weights_manifest.jsonl \
  --layer 0 \
  --ops qkv \
  --tokens "1,16" \
  --iters 20 \
  --out gemm_qkv_smoke.csv
```

### (2) GEMM sweep (layer 0, all ops, 전체 tokens)

```bash
./bench_zipserv_gemm_sweep \
  --manifest ~/models/llama31_70b/manifest/weights_manifest.jsonl \
  --layer 0 \
  --ops qkv,o,gateup,down \
  --out gemm_all.csv
```

### (3) Compress ratio + latency (layers 0-79)

```bash
./bench_llama31_70b_compress \
  --manifest ~/models/llama31_70b/manifest/weights_manifest.jsonl \
  --layers 0-79 \
  --out compress_layers.csv
```

## 5) 추가 스모크 (빠른 확인)

```bash
./bench_llama31_70b_compress \
  --manifest ~/models/llama31_70b/manifest/weights_manifest.jsonl \
  --layers 0 \
  --filter q_proj \
  --iters 20 \
  --out compress_layer0_qproj_smoke.csv
```

## 6) 결과 컬럼

### GEMM CSV (`bench_zipserv_gemm_sweep`)
- `name,M(tok),N,K,base_ms,zip_ms,zip/base`

### Compress CSV (`bench_llama31_70b_compress`)
- `Layer,Tensor,Shape,DType,Orignal_Size,Compressed_Size,Ratio,Comp(ms),H2D(ms),Decomp(ms),H2D+Decomp(ms),CompBW,DecompBW`

## 7) ZipServ vs bitcomp Compress Benchmark (BF16, Real Llama 3.1 70B Weights)

새 벤치 `bench_llama31_70b_zipserv_vs_bitcomp_compress`는 실제 Llama 3.1 70B BF16 weight tensor를 동일 입력으로 사용해 ZipServ 압축과 nvCOMP bitcomp 압축을 비교한다.

공정성 기준:
- 핵심 지표는 `compress API 호출 후 output buffer write 완료` 시점까지의 시간(`compress_write_ms`)이다.
- bitcomp에서 compressed size 확보가 별도 단계이면 `size_query_ms`로 분리한다.
- `total_ms = compress_write_ms + size_query_ms`로 정의한다.

측정 포함/제외:
- 포함: backend compress API 호출 시간(완료 동기화 기준), bitcomp size query(옵션 사용 시)
- 제외: 모델 로딩, 입력 H2D 준비, output/workspace 할당, handle/plan 생성

빌드:
- `NVCOMP_ROOT`에는 placeholder가 아니라 실제 nvCOMP 설치 루트를 넣어야 한다.
- 예: 이 머신에서는 `/home/pjw7200/ls` 아래에 `include/`, `lib/`가 있다.

```bash
cd ~/ZipServ_ASPLOS26/build
make

cd ~/ZipServ_ASPLOS26/kernel_benchmark
export MY_PATH=~/ZipServ_ASPLOS26

# nvCOMP 설치 루트(이 머신 기본값: /home/pjw7200/ls)
export NVCOMP_ROOT=${NVCOMP_ROOT:-/home/pjw7200/ls}
if [ ! -f "${NVCOMP_ROOT}/include/nvcomp/native/bitcomp.h" ] || [ ! -f "${NVCOMP_ROOT}/lib/libnvcomp.so" ]; then
  echo "Invalid NVCOMP_ROOT: ${NVCOMP_ROOT}"
  echo "Set NVCOMP_ROOT to a real nvCOMP root (contains include/ and lib/)."
  exit 1
fi
export LD_LIBRARY_PATH=~/ZipServ_ASPLOS26/build:${NVCOMP_ROOT}/lib:$LD_LIBRARY_PATH

make bench_llama31_70b_zipserv_vs_bitcomp_compress
```

nvCOMP 미설치 시:
- 타겟은 빌드되지만 실행 시 bitcomp 비활성 안내 메시지와 함께 종료한다.

실행 예시:

```bash
# 1) 단일 레이어 + q_proj 필터
./bench_llama31_70b_zipserv_vs_bitcomp_compress \
  --model_path ~/models/llama31_70b \
  --layer_idx 0 \
  --weight_filter q_proj \
  --warmup 10 \
  --iters 100 \
  --csv zipserv_vs_bitcomp_layer0_qproj.csv
```

```bash
# 2) 여러 텐서 aggregate 결과만 출력
./bench_llama31_70b_zipserv_vs_bitcomp_compress \
  --model_path ~/models/llama31_70b \
  --max_tensors 16 \
  --aggregate_only 1 \
  --iters 50 \
  --csv zipserv_vs_bitcomp_agg.csv
```

```bash
# 3) manifest 직접 지정 + size query 분리 비활성화
./bench_llama31_70b_zipserv_vs_bitcomp_compress \
  --manifest ~/models/llama31_70b/manifest/weights_manifest.jsonl \
  --bitcomp_include_size_query 0 \
  --iters 50 \
  --csv zipserv_vs_bitcomp_manifest.csv
```

## 8) KV Cache ZipServ Benchmark (Head-Split + Pad, BF16 Fixed)

새 벤치 `bench_kv_cache_zipserv_compress`는 저장된 KV cache `.npy`(float16)를 로드해:
- `(T, H, D) -> (T*H, D)`로 2D 매핑
- `M/K`를 각각 64 배수로 zero-padding
- FP16 -> BF16 변환 후 ZipServ 압축/복원 성능과 정확성을 측정한다.

핵심 지표:
- `logical_bytes`: 패딩 전 논리 KV 크기(BF16 기준)
- `input_bytes`: 패딩 후 ZipServ 입력 크기
- `compressed_bytes`
- `ratio_logical = compressed_bytes / logical_bytes`
- `ratio_input = compressed_bytes / input_bytes`
- `convert_ms`, `compress_ms`, `decompress_ms`

빌드:

```bash
cd ~/ZipServ_ASPLOS26
source Init.sh
cd build
make

cd ../kernel_benchmark
source test_env

export MY_PATH=~/ZipServ_ASPLOS26
export LD_LIBRARY_PATH=~/ZipServ_ASPLOS26/build:$LD_LIBRARY_PATH
make bench_zipserv_gemm_sweep bench_llama31_70b_compress

# bitcomp/nvCOMP까지 함께 빌드할 때만 설정
export NVCOMP_ROOT=${NVCOMP_ROOT:-/home/pjw7200/ls}
if [ ! -f "${NVCOMP_ROOT}/include/nvcomp/native/bitcomp.h" ] || [ ! -f "${NVCOMP_ROOT}/lib/libnvcomp.so" ]; then
  echo "Invalid NVCOMP_ROOT: ${NVCOMP_ROOT}"
  echo "Set NVCOMP_ROOT to a real nvCOMP root (contains include/ and lib/)."
  exit 1
fi
export LD_LIBRARY_PATH=~/ZipServ_ASPLOS26/build:${NVCOMP_ROOT}/lib:$LD_LIBRARY_PATH

make bench_kv_cache_zipserv_compress
make bench_kv_cache_zipserv_compress_scaling
```

주의:
- `NVCOMP_ROOT`는 실제 nvCOMP 설치 루트여야 한다. placeholder가 들어가면 `-lnvcomp` 링크에 실패한다.
- nvCOMP를 쓰지 않는 타겟은 `NVCOMP_ROOT` 없이도 빌드 가능하다.

실행 예시:

```bash
# 기본 실행
./bench_kv_cache_zipserv_compress \
  --input_dir ~/saved_kv_cache \
  --verify 1 \
  --out_csv kv_cache_zipserv.csv
```

```bash
# 스모크(2개 파일, 짧은 반복)
./bench_kv_cache_zipserv_compress \
  --input_dir ~/saved_kv_cache \
  --max_files 2 \
  --warmup 2 \
  --iters 5 \
  --verify 1 \
  --out_csv kv_cache_zipserv_smoke.csv
```

주의:
- ZipServ API 제약으로 2D + `M/K % 64 == 0` 형식이 필요하므로 패딩이 필수다.
- pad 비율이 큰 텐서는 `ratio_input`과 `ratio_logical` 해석이 달라질 수 있다.

## 9) KV Cache ZipServ File Scaling Benchmark

`bench_kv_cache_zipserv_compress_scaling`은 KV cache 파일 개수에 따른 압축 latency를 CPU/GPU 모드로 비교한다.

동작 방식:
- sweep point는 고정된 file count 집합 `1,2,4,8,16,32,64,128,256,512`이다.
- 각 point에서 정렬된 입력 파일의 앞 `N`개를 선택한다.
- 각 파일은 내부적으로 `16-token chunk`로 나눈 뒤 압축한다.
- 이 벤치는 token sweep를 더 이상 포함하지 않는다.

### CPU 모드 (`--mode cpu`, 기본값)

- **Single**: `InitBF16MatrixTripleBitmap_Reuse` (OMP 1 thread)
- **MT**: `InitBF16MatrixTripleBitmap_Reuse` (OMP max threads)
- **MT-AVX**: `InitBF16MatrixTripleBitmap_Reuse_SIMD` (OMP max threads + SIMD)
- **bitcomp**: nvCOMP bitcomp lossless 호스트 압축 (`NVCOMP_ROOT` 필요)

### GPU 모드 (`--mode gpu`)

- **ZipServ-GPU**: `InitBF16MatrixTripleBitmap_GPU_Fused`
- **bitcomp-GPU**: `bitcompCompressLossless` (`NVCOMP_ROOT` 필요)

GPU 구현 메모:
- ZipServ-GPU는 `analyze -> classify/local-pack -> prefix/scatter`를 단일 fused kernel로 수행한다.
- benchmark는 3-slot stream ring + pinned host staging buffer로 chunk H2D와 다음 chunk 준비를 겹친다.
- CPU의 `MT`는 chunk-level OpenMP 병렬화이고, GPU의 `MT-like`는 slot 기반 chunk pipelining이다.

### 빌드

```bash
cd ~/ZipServ_ASPLOS26
source Init.sh

cd build
make

cd ~/ZipServ_ASPLOS26/kernel_benchmark
source test_env

export MY_PATH=~/ZipServ_ASPLOS26
export LD_LIBRARY_PATH=~/ZipServ_ASPLOS26/build:$LD_LIBRARY_PATH

# bitcomp / bitcomp-GPU까지 포함해서 빌드할 때만 설정
export NVCOMP_ROOT=${NVCOMP_ROOT:-/home/pjw7200/ls}
if [ ! -f "${NVCOMP_ROOT}/include/nvcomp/native/bitcomp.h" ] || [ ! -f "${NVCOMP_ROOT}/lib/libnvcomp.so" ]; then
  echo "Invalid NVCOMP_ROOT: ${NVCOMP_ROOT}"
  echo "Set NVCOMP_ROOT to a real nvCOMP root (contains include/ and lib/)."
  exit 1
fi
export LD_LIBRARY_PATH=~/ZipServ_ASPLOS26/build:${NVCOMP_ROOT}/lib:$LD_LIBRARY_PATH

make bench_kv_cache_zipserv_compress_scaling \
     bench_kv_cache_zipserv_compress_token_scaling
```

주의:
- `NVCOMP_ROOT`를 설정하면 `-lnvcomp` 링크와 `-DZIPSERV_ENABLE_NVCOMP`가 자동 활성화된다.
- `NVCOMP_ROOT`는 실제 nvCOMP 설치 루트여야 한다. `${NVCOMP_ROOT}/lib/libnvcomp.so`가 존재해야 한다.
- 이 머신 예시: `export NVCOMP_ROOT=/home/pjw7200/ls`

### 실행 예시

```bash
# CPU file sweep
./bench_kv_cache_zipserv_compress_scaling \
  --input_dir ~/saved_kv_cache \
  --mode cpu \
  --warmup 10 \
  --iters 100 \
  --out_csv kv_cache_zipserv_scaling_cpu.csv
```

```bash
# GPU file sweep
./bench_kv_cache_zipserv_compress_scaling \
  --input_dir ~/saved_kv_cache \
  --mode gpu \
  --device 0 \
  --warmup 10 \
  --iters 100 \
  --out_csv kv_cache_zipserv_scaling_gpu.csv
```

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input_dir` | `~/saved_kv_cache` | 입력 KV cache 디렉토리 |
| `--mode` | `cpu` | 벤치마크 모드 (`cpu` 또는 `gpu`) |
| `--warmup` | `10` | 워밍업 반복 횟수 |
| `--iters` | `100` | 측정 반복 횟수 |
| `--max_files` | `-1` (전체) | preload할 최대 파일 수 |
| `--recursive` | `0` | 하위 디렉토리 재귀 탐색 (`0` 또는 `1`) |
| `--out_csv` | `kv_cache_zipserv_scaling_results.csv` | 출력 CSV 경로 |
| `--ext_filter` | (없음) | 확장자 필터 (예: `npy,noext`) |
| `--device` | `0` | CUDA 디바이스 인덱스 |

### 출력 CSV

컬럼: `file_count,mode,method,latency_ms,api_latency_ms,comp_ratio`

- `file_count`: 해당 point에서 처리한 파일 수
- `mode`: `cpu` 또는 `gpu`
- `method`: CPU 모드 시 `Single`/`MT`/`MT-AVX`/`bitcomp`, GPU 모드 시 `ZipServ-GPU`/`bitcomp-GPU`
- `latency_ms`: 선택된 file set 전체 평균 latency
- `api_latency_ms`: 실제 압축 API 구간만의 평균 시간. CPU 모드에서는 `0.0`
- `comp_ratio`: 압축률 (원본 크기 / 압축 크기)

## 10) KV Cache ZipServ Token Scaling Benchmark

`bench_kv_cache_zipserv_compress_token_scaling`은 각 KV 파일의 앞 `N`개 token만 처리하는 token-prefix 벤치다. 기존 file scaling 벤치와 분리되어 있으며, token 처리 방식은 여기서만 정의된다.

token 의미:
- FlexGen KV tensor shape는 `(T, H, D)`이고 첫 축 `T`가 token 축이다.
- token_count `N`이면 각 eligible file에서 앞 `N` tokens만 읽어서 압축한다.
- 즉, 파일 전체를 16-token 단위로 무조건 자르는 예전 token sweep와 다르게, 먼저 `N`개 token만 선택한 뒤 필요할 때만 추가 chunking을 적용한다.

chunking 의미:
- `--chunk_len 0`이면 chunking 없이 앞 `N` tokens 전체를 한 번에 압축한다.
- `--chunk_len K`이고 `N <= K`이면 chunking은 무시되고 앞 `N` tokens 전체를 한 번에 압축한다.
- `--chunk_len K`이고 `N > K`이면 앞 `N` tokens를 길이 `K`의 연속 chunk로 나눠 처리한다.

eligible file 규칙:
- 기본값 `--require_full_token_count 1`이면 `seq_len >= N`인 파일만 포함한다.
- `--require_full_token_count 0`이면 `seq_len < N`인 파일도 포함하고, 가능한 길이까지만 압축한다.

타이밍 방식:
- 각 iteration에서 eligible file 전체를 처리한 뒤, 실제 성공 처리된 `processed_files`로 나눠 per-file 평균 latency를 기록한다.
- GPU 모드는 파일 단위 strict isolation은 유지하지만, 파일 내부 chunk는 3-slot stream ring으로 겹쳐 처리한다.
- ZipServ-GPU 경로는 exponent 분석(histogram + top-7 contiguous range 선택)과 압축을 모두 GPU fused path에서 수행한다.
- `core_latency_ms`는 GPU 압축 API 구간 평균 시간이다. ZipServ-GPU는 fused API, bitcomp-GPU는 `compress`, CPU 모드에서는 `0.0`이다.

### 실행 예시

```bash
# CPU token sweep, chunking 없음
./bench_kv_cache_zipserv_compress_token_scaling \
  --input_dir ~/saved_kv_cache \
  --mode cpu \
  --token_counts 1,2,4,8,16,32,64,128,256,512,1024 \
  --chunk_len 0 \
  --warmup 10 \
  --iters 100 \
  --out_csv kv_cache_zipserv_token_scaling_cpu.csv
```

```bash
# CPU token sweep, 16-token chunking
./bench_kv_cache_zipserv_compress_token_scaling \
  --input_dir ~/saved_kv_cache \
  --mode cpu \
  --token_counts 1,2,4,8,16,32,64,128,256,512,1024 \
  --chunk_len 16 \
  --warmup 10 \
  --iters 100 \
  --out_csv kv_cache_zipserv_token_scaling_cpu_chunk16.csv
```

```bash
# GPU token sweep
./bench_kv_cache_zipserv_compress_token_scaling \
  --input_dir ~/saved_kv_cache \
  --mode gpu \
  --token_counts 1,2,4,8,16,32,64,128,256,512,1024 \
  --chunk_len 0 \
  --device 0 \
  --warmup 10 \
  --iters 100 \
  --out_csv kv_cache_zipserv_token_scaling_gpu.csv
```

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input_dir` | `~/saved_kv_cache` | 입력 KV cache 디렉토리 |
| `--mode` | `cpu` | 벤치마크 모드 (`cpu` 또는 `gpu`) |
| `--warmup` | `10` | 워밍업 반복 횟수 |
| `--iters` | `100` | 측정 반복 횟수 |
| `--max_files` | `-1` (전체) | preload할 최대 파일 수 |
| `--recursive` | `0` | 하위 디렉토리 재귀 탐색 (`0` 또는 `1`) |
| `--out_csv` | `kv_cache_zipserv_token_scaling_results.csv` | 출력 CSV 경로 |
| `--ext_filter` | (없음) | 확장자 필터 (예: `npy,noext`) |
| `--device` | `0` | CUDA 디바이스 인덱스 |
| `--token_counts` | (자동 생성) | token sweep에서 사용할 token count CSV |
| `--require_full_token_count` | `1` | `seq_len >= token_count`인 파일만 포함할지 여부 |
| `--chunk_len` | `0` | 선택적 token chunk 길이 (`0`이면 비활성화) |

### 출력 CSV

컬럼: `token_count,eligible_files,processed_files,requested_chunk_len,effective_chunk_len,mode,method,latency_ms,core_latency_ms,comp_ratio`

- `token_count`: 요청한 token prefix 길이
- `eligible_files`: 해당 point에서 실제 포함된 파일 수
- `processed_files`: 해당 point에서 실제 성공 처리된 파일 수
- `requested_chunk_len`: 사용자가 넘긴 `--chunk_len` 값
- `effective_chunk_len`: 실제로 적용된 chunk 길이. `token_count <= requested_chunk_len`이면 `token_count`와 같다
- `mode`: `cpu` 또는 `gpu`
- `method`: CPU 모드 시 `Single`/`MT`/`MT-AVX`/`bitcomp`, GPU 모드 시 `ZipServ-GPU`/`bitcomp-GPU`
- `latency_ms`: 전체 처리 시간을 `processed_files`로 정규화한 per-file 평균 latency
- `core_latency_ms`: 핵심 압축 API 구간 평균 시간. ZipServ-GPU는 fused API, bitcomp-GPU는 `compress`, CPU 모드에서는 `0.0`
- `comp_ratio`: 압축률 (원본 크기 / 압축 크기)

## 11) KV Cache ZipServ Token Decompression Scaling Benchmark

`bench_kv_cache_zipserv_decompress_token_scaling`은 token sweep(`--token_counts`)별로 ZipServ/bitcomp 압축 해제 지연 시간을 비교한다.

공정성 기준:
- 각 sweep point에서 **입력 compressed KV는 측정 전에 미리 준비**한다.
- **compressed KV 이동 시간(H2D/D2H)은 측정에서 제외**한다.
- 시간 측정 구간에는 decompress API 호출만 포함한다.

측정 모드:
- CPU: `ZipServ-CPU` (host-side reference decompressor), `bitcomp-CPU` (`bitcompHostUncompress`)
- GPU: `ZipServ-GPU` (`BF16TripleBitmap_Decompress_API`), `bitcomp-GPU` (`bitcompUncompress`)

### 빌드

```bash
cd ~/ZipServ_ASPLOS26
source Init.sh
cd build
make

cd ../kernel_benchmark
source test_env
export MY_PATH=~/ZipServ_ASPLOS26

# nvCOMP(bitcomp) 비교까지 포함할 때 필요
export NVCOMP_ROOT=${NVCOMP_ROOT:-/home/pjw7200/ls}
if [ ! -f "${NVCOMP_ROOT}/include/nvcomp/native/bitcomp.h" ] || [ ! -f "${NVCOMP_ROOT}/lib/libnvcomp.so" ]; then
  echo "Invalid NVCOMP_ROOT: ${NVCOMP_ROOT}"
  echo "Set NVCOMP_ROOT to a real nvCOMP root (contains include/ and lib/)."
  exit 1
fi
export LD_LIBRARY_PATH=~/ZipServ_ASPLOS26/build:${NVCOMP_ROOT}/lib:$LD_LIBRARY_PATH

make bench_kv_cache_zipserv_decompress_token_scaling
```

### 실행 예시

```bash
# CPU token sweep
./bench_kv_cache_zipserv_decompress_token_scaling \
  --input_dir ~/saved_kv_cache \
  --mode cpu \
  --token_counts 1,2,4,8,16,32,64,128,256,512,1024 \
  --chunk_len 16 \
  --warmup 10 \
  --iters 100 \
  --out_csv kv_cache_zipserv_decompress_token_scaling_cpu.csv
```

```bash
# GPU token sweep
./bench_kv_cache_zipserv_decompress_token_scaling \
  --input_dir ~/saved_kv_cache \
  --mode gpu \
  --device 0 \
  --token_counts 1,2,4,8,16,32,64,128,256,512,1024 \
  --chunk_len 0 \
  --warmup 10 \
  --iters 100 \
  --out_csv kv_cache_zipserv_decompress_token_scaling_gpu.csv
```

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input_dir` | `~/saved_kv_cache` | 입력 KV cache 디렉토리 |
| `--mode` | `cpu` | 벤치마크 모드 (`cpu` 또는 `gpu`) |
| `--warmup` | `10` | 워밍업 반복 횟수 |
| `--iters` | `100` | 측정 반복 횟수 |
| `--max_files` | `-1` (전체) | preload할 최대 파일 수 |
| `--recursive` | `0` | 하위 디렉토리 재귀 탐색 (`0` 또는 `1`) |
| `--out_csv` | `kv_cache_zipserv_decompress_token_scaling_results.csv` | 출력 CSV 경로 |
| `--ext_filter` | (없음) | 확장자 필터 (예: `npy,noext`) |
| `--device` | `0` | CUDA 디바이스 인덱스 |
| `--token_counts` | (자동 생성) | token sweep에서 사용할 token count CSV |
| `--require_full_token_count` | `1` | `seq_len >= token_count`인 파일만 포함할지 여부 |
| `--chunk_len` | `0` | 선택적 token chunk 길이 (`0`이면 비활성화) |

### 출력 CSV

컬럼: `token_count,eligible_files,processed_files,requested_chunk_len,effective_chunk_len,mode,method,latency_ms,core_latency_ms,comp_ratio`

- `token_count`: 요청한 token prefix 길이
- `eligible_files`: 해당 point에서 조건을 만족한 파일 수
- `processed_files`: 해당 method에서 실제 성공 처리된 파일 수
- `requested_chunk_len`: 사용자가 넘긴 `--chunk_len` 값
- `effective_chunk_len`: 실제로 적용된 chunk 길이
- `mode`: `cpu` 또는 `gpu`
- `method`: CPU 모드 `ZipServ-CPU`/`bitcomp-CPU`, GPU 모드 `ZipServ-GPU`/`bitcomp-GPU`
- `latency_ms`: transfer/prep 제외 per-file 평균 decompress latency
- `core_latency_ms`: GPU 모드의 API 구간 시간(이벤트 기반), CPU 모드에서는 `0.0`
- `comp_ratio`: 압축률 (원본 크기 / 압축 크기)
