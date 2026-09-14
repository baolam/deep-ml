#include <cuda_runtime.h>

__global__ void kernel(const float* input, const float* bias, float* output, int rows, int cols) {
    // TODO
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;

    if (idx < total_elements)
    {
        int col = idx % cols;
        float val = input[idx] + bias[col];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

void solve(const float* input, const float* bias, float* output, int rows, int cols) {
    // TODO: allocate device memory, copy in, launch kernel, copy out, free
    size_t matrix_bytes = (size_t) rows * cols * sizeof(float);
    size_t bias_types = (size_t) cols * sizeof(float);
    float *d_input = nullptr;
    float *d_bias = nullptr;
    float *d_output = nullptr;

    // cudaMalloc((void**) &d_input, matrix_bytes);
    // cudaMalloc((void**)&d_bias, matrix_bytes);
    // cudaMalloc((void**)&d_output, matrix_bytes);
    cudaMalloc(&d_input, matrix_bytes);
    cudaMalloc(&d_bias, bias_types);
    cudaMalloc(&d_output, matrix_bytes);

    cudaMemcpy(d_input, input, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_types, cudaMemcpyHostToDevice);

    int total_elements = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_bias, d_output, rows, cols);

    // cudaEventSynchronize();

    cudaMemcpy(output, d_output, matrix_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);
}