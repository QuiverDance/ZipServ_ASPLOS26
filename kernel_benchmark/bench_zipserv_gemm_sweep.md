# bench_zipserv_gemm_sweep

## 1) 개요
`bench_zipserv_gemm_sweep.cu`는 manifest에서 선택한 레이어의 BF16 가중치에 대해 token sweep 벤치마크를 수행한다.
각 포인트에서 아래 2개 경로를 같은 입력으로 측정한다.
- baseline: `cublasGemmEx` (비압축 weight)
- zipserv: `BF16TripleBitmap_MM_API` (압축 weight)

출력 CSV 컬럼:
- `name,M(tok),N,K,base_ms,zip_ms,zip/base`

## 2) 측정 대상
### 2.1 `--ops`로 선택되는 개별 tensor
- `qkv`: `q_proj`, `k_proj`, `v_proj`
- `o`: `o_proj`
- `gateup`: `gate_proj`, `up_proj`
- `down`: `down_proj`

### 2.2 fused tensor
아래 2개는 `--ops`와 무관하게 항상 측정된다.
- `qkv_fused` (`group=qkv`): `q_proj -> k_proj -> v_proj` row-concat
- `gateup_fused` (`group=gateup`): `gate_proj -> up_proj` row-concat

## 3) CLI
| 인자 | 기본값 | 설명 |
|---|---:|---|
| `--manifest <path>` | 필수 | manifest JSONL 경로 |
| `--layer <int>` | `0` | 대상 layer |
| `--ops <spec>` | `qkv,o,gateup,down` | 개별 tensor 그룹 선택 (`all` 지원) |
| `--tokens <list>` | preset | token sweep 목록 |
| `--warmup <int>` | `10` | warmup 반복 수 |
| `--iters <int>` | `100` | 측정 반복 수 |
| `--out <path>` | `results_zipserv_gemm.csv` | 출력 CSV |
| `--device <int>` | `0` | CUDA device |
| `--seed <int>` | `1234` | 입력 생성 seed |
| `--weight_offload <0\|1>` | `0` | 반복 구간에 weight H2D 포함 여부 |

## 4) 핵심 동작
1. manifest에서 layer tensor를 수집한다.
2. `--ops`로 선택된 개별 tensor를 로드한다.
3. fused 구성에 필요한 `q/k/v/gate/up` 존재를 검증하고 fused weight 2개를 생성한다.
4. 각 weight(개별 + fused)에 대해:
   - host에서 1회 압축
   - device 압축 버퍼 준비
   - token sweep에서 baseline/zipserv 평균 지연시간 측정
5. 결과를 CSV에 row 단위로 기록한다.

## 5) offload 타이밍
- `--weight_offload=0`: GEMM 커널만 측정
- `--weight_offload=1`: 반복마다 weight H2D + GEMM 측정
  - baseline: uncompressed weight H2D 포함
  - zipserv: compressed weight H2D 포함
