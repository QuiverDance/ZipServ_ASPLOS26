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
