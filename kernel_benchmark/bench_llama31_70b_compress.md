# 1) 개요
`bench_llama31_70b_compress.cu`는 manifest에 있는 BF16 텐서를 대상으로, ZipServ 포맷 압축(`InitBF16MatrixTripleBitmap`)과 복원(`BF16TripleBitmap_Decompress_API`)의 성능을 텐서 단위로 측정한다.  
코드는 각 텐서마다 압축 평균 지연시간(`compress_ms`), H2D 평균 지연시간(`h2d_ms`), 해제 평균 지연시간(`decompress_ms`), H2D+해제 평균 지연시간(`h2d_decompress_ms`), 원본/압축 크기(`orig_bytes`, `comp_bytes`), 압축률(`orig/comp`), 압축/해제 대역폭(원본 바이트 기준 GB/s)을 계산해 터미널 표와 CSV로 출력한다.

# 2) CLI/파라미터
| 옵션 | 기본값 | 의미 | 벤치마크 영향 |
|---|---:|---|---|
| `--manifest <path>` | 없음(필수) | manifest JSONL 경로 | 측정 대상 텐서 집합 결정 |
| `--warmup <int>` | `10` | 텐서별 warmup 반복 수 | 압축 warmup/해제 warmup 횟수 |
| `--iters <int>` | `100` | 텐서별 측정 반복 수 | 평균 latency 계산 분모 |
| `--device <int>` | `0` | CUDA device index | `cudaSetDevice` 대상 |
| `--out <path>` | `llama31_70b_compress_results.csv` | CSV 출력 파일 | 결과 저장 위치 |
| `--layers <spec>` | 빈 문자열 | 레이어 필터 (`0-79`, `0,1,2`) | 선택 텐서 집합 축소/확장 |
| `--filter <spec>` | 빈 문자열 | 텐서 이름 substring 필터(콤마 구분) | 선택 텐서 집합 축소 |
| `--flush-l2` | `false` | 해제 반복 전 L2 flush | 해제 측정의 캐시 상태 변경 |
| `--help` | - | help 출력 | 즉시 종료 |

# 3) 실행 흐름 — main() 기준 단계별 타임라인
## Phase 1: 옵션 파싱/검증
- (a) 수행 함수/코드 위치: `ParseOptions`, `ParseLayerSpec`
- (b) 하는 일: CLI 파싱, `warmup/iters/device` 유효성 검사, layer spec 파싱
- (c) 입력/출력 데이터: `argv` -> `ProgramOptions`, `allowed_layers(set<int>)`
- (d) 다음 단계 조건: 필수 인자(`--manifest`) 포함 및 파싱 성공

## Phase 2: 장치 선택 및 manifest 로드
- (a) 수행 함수/코드 위치: `cudaSetDevice`, `LoadManifest`
- (b) 하는 일: GPU 선택, JSONL manifest 읽기
- (c) 입력/출력 데이터: manifest line -> `ManifestTensor{layer,name,shape,dtype,path,orig_bytes}`
- (d) 다음 단계 조건: 로드 성공, 이후 `--layers/--filter` 적용

## Phase 3: 대상 텐서 선택/정렬
- (a) 수행 함수/코드 위치: `selected_tensors` 구성 루프 + `std::sort`
- (b) 하는 일: layer 필터, 이름 substring 필터 적용 후 정렬
- (c) 입력/출력 데이터: `all_tensors` -> `selected_tensors` (정렬 키: layer, name)
- (d) 다음 단계 조건: `selected_tensors`가 비어있지 않아야 진행

## Phase 4: 출력 초기화
- (a) 수행 함수/코드 위치: `ofstream` open, `WriteCSVHeader`, `PrintTableHeader`
- (b) 하는 일: CSV 헤더 출력, 터미널 표 헤더 출력
- (c) 입력/출력 데이터: CSV 파일 생성, stdout 초기 정보 출력
- (d) 다음 단계 조건: CSV 파일 open 성공

## Phase 5: 텐서별 처리 (`ProcessTensor`)
- (a) 수행 함수/코드 위치: main 루프 내부 `ProcessTensor`
- (b) 하는 일:
  - shape/dtype/file 체크
  - 파일 읽기 + BF16 변환
  - exponent 분포 분석
  - 압축 warmup/측정
  - H2D / 해제 / H2D+해제 warmup/측정
- (c) 입력/출력 데이터:
  - 입력: 텐서 BF16 host buffer (`rows x cols`)
  - 출력: `BenchRow` (size/ratio/latency/BW/status/note)
- (d) 다음 단계 조건: 처리 완료 후 CSV 1행 + 터미널 1행 출력

## Phase 6: 집계 및 종료
- (a) 수행 함수/코드 위치: main 루프 종료 후 summary 출력
- (b) 하는 일: `ok/skipped/failed` 개수 출력, CSV close
- (c) 입력/출력 데이터: 카운트 결과 + 파일 flush/close
- (d) 다음 단계 조건: 없음 (프로그램 종료)

# 4) 측정 로직 상세 (압축/해제/변환 각각)
## 4-1) 압축 단계
- warmup: `for i in [0, warmup)`에서 `RunCompressionOnce(...)` 호출 후 즉시 free
- 측정 반복: `for i in [0, iters)`에서 `std::chrono::high_resolution_clock`으로 `RunCompressionOnce` 1회 시간 측정
- 측정 시작/종료: `begin = now()` / `end = now()` (CPU timer)
- 동기화: 별도 CUDA 이벤트 동기화 없음 (압축 측정은 host 함수 호출 시간 기준)
- 평균 계산: `compress_ms = total_compress_ms / iters`
- 포함 항목:
  - `InitBF16MatrixTripleBitmap` 호출
  - 압축 버퍼 생성(호스트 malloc 계열)
- 제외 항목:
  - 반복 후 free 시간(타이머 밖)
  - exponent 분석(`analyzeExponentDistribution_BF16`) 시간

## 4-2) 해제 단계 (`decompress_ms`)
- 사전 준비(측정 제외):
  - `PrepareDeviceCompressedBuffers`에서 `cudaMalloc`, 초기 H2D `cudaMemcpy`, `cudaMemset(output)`
- warmup:
  - `for i in [0, warmup)`에서 `BF16TripleBitmap_Decompress_API(...)`
  - warmup 후 `cudaDeviceSynchronize()`
- 측정 반복:
  - `RecordStart(0)` -> `BF16TripleBitmap_Decompress_API(...)` -> `RecordStop(0)` -> `SyncStop()` -> `ElapsedMs()`
- 평균 계산: `decompress_ms = total_ms / iters`
- 포함/제외:
  - 포함: Decompress API 커널 시간
  - 제외: 버퍼 할당, 초기 H2D, memset, L2 flush 시간(`RecordStart` 이전 호출)

## 4-3) H2D 단계 (`h2d_ms`)
- warmup:
  - `for i in [0, warmup)`에서 `CopyCompressedToDeviceAsync(...)` 실행 후 `cudaDeviceSynchronize()`
- 측정 반복:
  - `RecordStart(0)` -> `CopyCompressedToDeviceAsync(...)` -> `RecordStop(0)` -> `SyncStop()` -> `ElapsedMs()`
- 평균 계산: `h2d_ms = total_h2d_ms / iters`
- 포함/제외:
  - 포함: 압축 버퍼 7종(`sign_mantissa`, `compressed_full`, `bitmap1/2/3`, `tile_offsets_median/global`) H2D 복사
  - 제외: 버퍼 할당/해제

## 4-4) H2D+해제 단계 (`h2d_decompress_ms`)
- warmup:
  - `for i in [0, warmup)`에서 `CopyCompressedToDeviceAsync(...)` 후 `BF16TripleBitmap_Decompress_API(...)`
- 측정 반복:
  - `RecordStart(0)` -> `CopyCompressedToDeviceAsync(...)` -> `BF16TripleBitmap_Decompress_API(...)` -> `RecordStop(0)` -> `SyncStop()` -> `ElapsedMs()`
- 평균 계산: `h2d_decompress_ms = total_h2d_decompress_ms / iters`
- 포함/제외:
  - 포함: H2D 복사 + Decompress API 커널 시간
  - 제외: 버퍼 할당/해제, L2 flush 시간(`RecordStart` 이전 호출)

## 4-5) 포맷 변환/입력 준비 단계
- `ReadBinaryFile` + `ConvertRawToBF16` + `analyzeExponentDistribution_BF16`는 측정 구간 밖
- 즉, I/O 및 전처리 비용은 `compress_ms`, `h2d_ms`, `decompress_ms`, `h2d_decompress_ms`에 포함되지 않음

## 4-6) 지표 계산/단위
- 압축률: `orig_bytes / comp_bytes` (`FormatRatioX`, 표시는 `xx.xx x`)
- 대역폭: `BytesPerSecondToGBps(orig_bytes, ms)`  
  - 압축/해제 모두 분자에 `orig_bytes` 사용
- 출력 단위:
  - 시간: `xxx.xxxms`
  - 크기: `B/KiB/MiB/GiB/TiB`
  - BW: `B/s ~ TB/s`
- 통계 방식:
  - 평균만 사용
  - min/max/중앙값/outlier 제거 없음

# 5) 메모리/버퍼 라이프사이클
| 버퍼 | 위치 | 할당/해제 | 측정 포함 여부 |
|---|---|---|---|
| `raw_data` | Host (`std::vector<uint8_t>`) | 파일 읽기 시 생성/자동 해제 | 제외 |
| `host_bf16` | Host (`std::vector<__nv_bfloat16>`) | 변환 시 생성/자동 해제 | 제외 |
| `warmup_buffers` | Host (`CompressedBuffers`) | warmup 반복마다 생성/즉시 free | warmup이므로 제외 |
| `iteration_buffers` | Host (`CompressedBuffers`) | 측정 반복마다 생성, 대부분 반복에서 즉시 free | 생성은 포함, free는 제외 |
| `final_buffers` | Host (`CompressedBuffers`) | 마지막 압축 결과 유지 후 decomp 완료 뒤 free | decomp 준비/해제 자체는 제외 |
| `DeviceCompressedBuffers` | Device (`cudaMalloc`) | `BenchmarkDecompress` 시작 시 할당, 종료 시 free | 할당은 제외, 반복 H2D는 `h2d_ms`/`h2d_decompress_ms`에 포함 |
| L2 flush 버퍼(`d_flush_buffer`) | Device static | 첫 flush 시 1회 할당, 프로세스 종료까지 유지 | flush 자체는 측정 제외 |

스트림/이벤트:
- 스트림: 기본 stream `0`만 사용
- 이벤트: `CudaEventTimer`의 start/stop 이벤트를 decomp 반복마다 사용

# 6) Sweep 설계 요약
이 벤치의 sweep은 “선택된 텐서 목록” 순회이다.

```cpp
parse options
load manifest
selected_tensors = filter_by_layers_and_name(manifest)
sort(selected_tensors by layer, name)

for tensor in selected_tensors:
    row = ProcessTensor(tensor):
        validate shape/dtype/file
        read file -> host_bf16
        analyzeExponentDistribution_BF16(host_bf16)

        repeat warmup:
            RunCompressionOnce(...)

        repeat iters:
            measure (chrono) RunCompressionOnce(...)
        keep last compressed buffers as final_buffers

        benchmark decomp with final_buffers:
            prepare device buffers once
            repeat warmup:
                optional flush_l2
                BF16TripleBitmap_Decompress_API(...)
            repeat iters:
                optional flush_l2
                measure (cuda events) one Decompress API call

        benchmark h2d with final_buffers:
            repeat warmup:
                CopyCompressedToDeviceAsync(...)
            repeat iters:
                measure (cuda events) one H2D copy batch

        benchmark h2d+decomp with final_buffers:
            repeat warmup:
                optional flush_l2
                CopyCompressedToDeviceAsync(...)
                BF16TripleBitmap_Decompress_API(...)
            repeat iters:
                optional flush_l2
                measure (cuda events) H2D + Decompress

        compute ratio/BW
    write CSV row
    print terminal row
```
