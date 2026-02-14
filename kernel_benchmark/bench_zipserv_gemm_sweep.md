# 1) 개요
`bench_zipserv_gemm_sweep.cu`는 manifest에서 선택한 레이어의 BF16 가중치(`qkv/o/gateup/down`)를 대상으로, 토큰 길이(`--tokens`)를 sweep하면서 두 경로를 같은 입력에 대해 반복 측정한다.
- baseline: `cublasGemmEx` (비압축 weight 사용)
- zipserv: `BF16TripleBitmap_MM_API` (사전 압축 weight 사용)

핵심 출력 지표.
- `lat_ms`: warmup 이후 `iters` 반복의 평균 지연시간(ms/iter)

`--emit_aggregate=1`이면 `qkv_sum`, `gateup_sum` aggregate row도 추가로 계산한다(개별 tensor row 합산).

# 2) CLI/파라미터
`ProgramOptions` 기본값과 `ParseOptions` 파싱 로직 기준.

| 인자 | 기본값 | 의미 | 벤치마크 영향 |
|---|---:|---|---|
| `--manifest <path>` | 없음(필수) | manifest JSONL 경로 | 측정 대상 tensor 집합이 결정됨. 미입력 시 종료 |
| `--layer <int>` | `0` | 대상 layer index | 해당 layer의 weight만 필터링 |
| `--ops <spec>` | `qkv,o,gateup,down` | 그룹 선택(`qkv`,`o`,`gateup`,`down`,`all`) | 그룹별 tensor 포함/제외 결정 |
| `--tokens <list>` | `1,2,4,...,256,512,1024` | 토큰 sweep 리스트 | sweep 축. 각 token 값마다 baseline/zipserv 모두 측정 |
| `--warmup <int>` | `10` | warmup 반복 횟수 | 타이머 시작 전 예열 횟수 |
| `--iters <int>` | `100` | 측정 반복 횟수 | 평균 지연시간 계산 분모 |
| `--out <path>` | `results_zipserv_gemm.csv` | CSV 출력 파일 | 결과 저장 경로 |
| `--device <int>` | `0` | CUDA device index | `cudaSetDevice` 대상 |
| `--seed <int>` | `1234` | 입력 B 생성 seed 기준값 | 각 sweep 포인트 입력값 재현성에 영향 |
| `--emit_aggregate <0\|1>` | `1` | aggregate row 출력 여부 | `qkv_sum`,`gateup_sum` row 추가 여부 |
| `--weight_offload <0\|1>` | `0` | weight H2D를 측정 구간에 포함할지 | `1`이면 baseline/zipserv 모두 반복마다 weight H2D + GEMM 측정 |
| `--help` | - | usage 출력 | 즉시 종료 |

`sweep`의 직접 축은 `tokens`이며, 실제 실행은 `선택된 weight들 × tokens`의 2중 sweep이다.

# 3) 실행 흐름 — main() 기준 단계별 타임라인
## Phase 1: 옵션 파싱/검증
- (a) 수행 함수/코드 위치: `main` 초반, `ParseOptions`, `ParseOps`, `ParseIntList`
- (b) 하는 일: CLI 파싱, `ops`/`tokens` 유효성 검증
- (c) 입력/출력 데이터: `argv` 입력, `ProgramOptions`, `wanted_ops(set)`, `tokens(vector<int>)` 생성
- (d) 다음 단계 조건: 파싱/검증 성공 시 진행, 실패 시 종료

## Phase 2: 디바이스/manifest 로드
- (a) 수행 함수/코드 위치: `cudaSetDevice`, `LoadManifest`
- (b) 하는 일: CUDA 디바이스 선택, manifest JSONL 읽기
- (c) 입력/출력 데이터: manifest line에서 `layer,name,shape,dtype,path,nbytes` 로드
- (d) 다음 단계 조건: 디바이스 설정 및 manifest 로드 성공

## Phase 3: 대상 weight 선별 및 BF16 로드
- (a) 수행 함수/코드 위치: `by_suffix` 구성, `kOpSpecs`, `ConvertRawToBF16`
- (b) 하는 일: layer/ops 필터로 대상 tensor 순서 확정 후 파일에서 BF16 raw 로딩
- (c) 입력/출력 데이터: 각 weight의 `rows=tensor.shape[0]`, `cols=tensor.shape[1]`, `host_weight(vector<bf16>)`
- (d) 다음 단계 조건: 필요한 tensor가 모두 존재하고 2D/bf16 제약 만족

## Phase 4: 출력/핸들 초기화
- (a) 수행 함수/코드 위치: CSV open + `WriteCsvHeader`, `cublasCreate`, `cublasSetStream`, `cublasSetMathMode`
- (b) 하는 일: CSV 헤더 작성, cublas 핸들 생성, stream=0 설정, math mode=default 설정/조회
- (c) 입력/출력 데이터: `baseline_cfg`, `zipserv_cfg` 문자열 생성
- (d) 다음 단계 조건: 파일/핸들 초기화 성공

## Phase 5: weight 단위 준비(압축 + 디바이스 버퍼)
- (a) 수행 함수/코드 위치: weight loop 내부 `CompressWeightOnce`, `PrepareDeviceCompressed`, `cudaMalloc(d_weight)`
- (b) 하는 일: 원본 weight 1회 압축, 압축 버퍼 GPU 업로드, baseline용 d_weight 준비
- (c) 입력/출력 데이터:
  - 입력: `host_weight[rows, cols]`
  - 출력: `CompressedBuffers(host)`, `DeviceCompressedBuffers(device)`, `d_weight`
- (d) 다음 단계 조건: 압축 성공(행/열 64 배수 제약 포함), GPU 메모리 준비 성공

## Phase 6: tokens sweep 측정(핵심 측정 루프)
- (a) 수행 함수/코드 위치: token loop 내부 `BuildRandomInputB`, `RunBaselineGemm`, `RunZipservGemm`
- (b) 하는 일: token별 입력 B 생성 후 baseline/zipserv 순서로 측정
- (c) 입력/출력 데이터:
  - 입력 텐서: `B[K_in, tokens]`, weight(`W[N_out,K_in]` 또는 compressed weight)
  - 출력 텐서: `Out[N_out, tokens]` (baseline/zipserv 별도 버퍼)
  - 논리 GEMM 크기 기록: `M=tokens`, `N=w.rows`, `K=w.cols`
- (d) 다음 단계 조건: 각 경로 실행 성공 시 CSV row용 결과 추가 후 다음 token

## Phase 7: aggregate row 생성(옵션)
- (a) 수행 함수/코드 위치: `if (opts.emit_aggregate)` 블록
- (b) 하는 일: `qkv`, `gateup`의 같은 `mode/tokens` row를 합산
- (c) 입력/출력 데이터: `orig_bytes`, `comp_bytes`, `lat_ms` 합산한 `kind=aggregate` row
- (d) 다음 단계 조건: 옵션이 켜져 있으면 수행, 아니면 건너뜀

## Phase 8: 정렬/저장/종료
- (a) 수행 함수/코드 위치: `std::sort(all_rows)`, `WriteCsvRow`, `cublasDestroy`
- (b) 하는 일: row 정렬 후 CSV 기록, 핸들/리소스 정리
- (c) 입력/출력 데이터: 최종 CSV 파일, stdout 요약(`rows_written`, `csv`)
- (d) 다음 단계 조건: 없음(프로그램 종료)

# 4) 측정 로직 상세
## 측정 timer
공통:
- 타이머 구현: `bench_common::CudaEventTimer` (`cudaEventRecord(start/stop)`, `cudaEventSynchronize(stop)`)
- 단위: `cudaEventElapsedTime` 결과(ms)
- 최종 `lat_ms`: `total_ms / iters`

baseline(`RunBaselineGemm`):
- warmup: `for i in [0, warmup)`
  - `weight_offload=1`이면 `cudaMemcpyAsync(d_weight <- h_weight)` 포함
  - `cublasGemmEx(...)` 실행
- warmup 후: `cudaStreamSynchronize(stream)`
- 측정 구간:
  - 시작: `timer.RecordStart(stream)`
  - 반복: `for i in [0, iters)`에서 (선택적 weight H2D + `cublasGemmEx`)
  - 종료: `timer.RecordStop(stream)` -> `timer.SyncStop()` -> `ElapsedMs`

zipserv(`RunZipservGemm`):
- warmup: `for i in [0, warmup)`
  - `weight_offload=1`이면 `CopyCompressedToDeviceAsync(...)` 포함
  - `BF16TripleBitmap_MM_API(...)` 실행
- warmup 후: `cudaStreamSynchronize(stream)`
- 측정 구간:
  - 시작: `timer.RecordStart(stream)`
  - 반복: `for i in [0, iters)`에서 (선택적 compressed weight H2D + `BF16TripleBitmap_MM_API`)
  - 종료: `timer.RecordStop(stream)` -> `timer.SyncStop()` -> `ElapsedMs`

## 집계 방식
- 통계: 평균만 사용(`total_ms / iters`)
- 중앙값/min/max/outlier 제거: 없음
- 단위 변환: 없음(ms 그대로), `GBps = bytes / (ms/1000) / 1e9`

# 5) 비교 대상/베이스라인
## 경로 선택/분기
- 이 코드는 각 sweep 포인트에서 baseline과 zipserv를 모두 실행한다(선택 분기 아님).
- 실행 순서는 고정: `RunBaselineGemm` 후 `RunZipservGemm`.

## baseline(cublas) 핵심 호출 파라미터
- API: `cublasGemmEx`
- `TransA=T`, `TransB=N`
- `m=N_out`, `n=tokens`, `k=K_in`
- A/B/C type: `CUDA_R_16BF`
- compute type: `CUDA_R_32F`
- algo: `0`
- `lda=K_in`, `ldb=K_in`, `ldc=N_out`
- cublas math mode: 코드에서 `CUBLAS_DEFAULT_MATH`로 설정

## zipserv 경로 핵심 호출 파라미터
- API: `BF16TripleBitmap_MM_API`
- 입력: 압축 버퍼(`sign_mantissa`, `compressed_full`, `bitmap1/2/3`, `tile_offsets_*`)
- 크기: `M_Global=N_out`, `N_Global=tokens`, `K_Global=K_in`
- `Split_K=1`(고정), `stream=0`

# 6) 데이터 준비/메모리 흐름
## 메모리 할당/해제
| 대상 | 위치/방식 | 생명주기 |
|---|---|---|
| 원본 weight | host `std::vector<__nv_bfloat16>` | weight별 로드 후 프로그램 종료까지 유지 |
| 압축 host 버퍼 | `malloc` (`CompressedBuffers`) | weight별 생성 후 weight loop 끝에서 `FreeCompressedBuffers` |
| 압축 device 버퍼 | `cudaMalloc` (`DeviceCompressedBuffers`) | weight별 생성 후 `FreeDeviceCompressedBuffers` |
| baseline weight `d_weight` | `cudaMalloc` | weight별 1회 할당/복사 후 weight loop 종료 시 해제 |
| 입력 `d_b` | `cudaMalloc` | token별 할당 후 즉시 해제 |
| 출력 `d_out_*` | `cudaMalloc` | token별 할당 후 즉시 해제 |

## 초기화/seed
- 입력 B는 `BuildRandomInputB`에서 `std::mt19937(seed)` + `uniform(-1,1)`로 생성.
- seed는 `opts.seed + wi*97 + tok`로 계산되어 weight/token 조합별 결정적(deterministic)이다.

## 전송 포함/제외(측정 구간 관점)
- 항상 측정 제외:
  - 원본 weight 파일 로드
  - weight 압축(`CompressWeightOnce`)
  - token별 입력 B 생성
  - token별 `d_b` 초기 H2D, 출력 버퍼 `cudaMemset`
  - `PrepareDeviceCompressed` 초기 H2D
- `weight_offload=0`:
  - 반복 측정 구간은 GEMM 커널 실행만 포함
- `weight_offload=1`:
  - baseline: 반복마다 원본 weight H2D + `cublasGemmEx` 포함
  - zipserv: 반복마다 압축 버퍼 H2D + `BF16TripleBitmap_MM_API` 포함

압축/해제 라이프사이클:
- 이 벤치에서는 “압축”만 host에서 1회 수행하고, “압축 데이터 기반 GEMM”만 측정한다.
- 별도 decompression benchmark 단계는 없다.

# 7) Sweep 설계 요약
토큰이 직접 sweep 파라미터이며, weight 집합(ops/layer 필터 결과)과의 곱집합으로 실행된다.

```cpp
parse opts/ops/tokens
load manifest and select weights by {layer, ops}

for each weight in fixed request order:
    load bf16 weight from file
    compress once on host
    allocate/copy compressed buffers on device
    allocate/copy uncompressed d_weight

    for each tok in sorted_unique(tokens):
        build random B(K, tok) with deterministic seed
        allocate d_b, d_out_baseline, d_out_zipserv
        copy B to d_b

        baseline_ms = RunBaselineGemm(...)
        zipserv_ms  = RunZipservGemm(...)

        append tensor rows (baseline + zipserv)
        print per-tensor stdout row
        free token-local device buffers

    free weight-local buffers

if emit_aggregate:
    aggregate qkv and gateup rows by (group, mode, tokens)
    append aggregate rows

sort all_rows and write CSV
```

각 sweep 포인트(특정 weight, 특정 token)마다 `측정 -> row 추가 -> stdout 출력 -> 다음 token` 순서로 진행된다.
