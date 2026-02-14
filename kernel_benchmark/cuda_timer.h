#pragma once

#include <cuda_runtime.h>

#include <string>

namespace bench_common {

class CudaEventTimer {
public:
    CudaEventTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaEventTimer() {
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    bool RecordStart(cudaStream_t stream, std::string* error) const {
        cudaError_t e = cudaEventRecord(start_, stream);
        if (e != cudaSuccess) {
            if (error != nullptr) {
                *error = std::string("cudaEventRecord(start) failed: ") + cudaGetErrorString(e);
            }
            return false;
        }
        return true;
    }

    bool RecordStop(cudaStream_t stream, std::string* error) const {
        cudaError_t e = cudaEventRecord(stop_, stream);
        if (e != cudaSuccess) {
            if (error != nullptr) {
                *error = std::string("cudaEventRecord(stop) failed: ") + cudaGetErrorString(e);
            }
            return false;
        }
        return true;
    }

    bool SyncStop(std::string* error) const {
        cudaError_t e = cudaEventSynchronize(stop_);
        if (e != cudaSuccess) {
            if (error != nullptr) {
                *error = std::string("cudaEventSynchronize(stop) failed: ") + cudaGetErrorString(e);
            }
            return false;
        }
        return true;
    }

    bool ElapsedMs(float* total_ms, std::string* error) const {
        cudaError_t e = cudaEventElapsedTime(total_ms, start_, stop_);
        if (e != cudaSuccess) {
            if (error != nullptr) {
                *error = std::string("cudaEventElapsedTime failed: ") + cudaGetErrorString(e);
            }
            return false;
        }
        return true;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

}  // namespace bench_common

