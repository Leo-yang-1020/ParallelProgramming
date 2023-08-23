#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"
#define THREADPERBLOCK 2

extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}
__global__ void 
upsweep(int twod, int *device_result)
{
    int twod1 = twod*2;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index % twod1 == 0)
        device_result[index + twod1 -1] += device_result[index + twod -1];
}

__global__ void 
downsweep(int twod, int *device_result)
{
    int temp;
    int twod1 = twod*2;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index % twod1 == 0) {
        temp = device_result[index + twod -1];
        device_result[index + twod -1] = device_result[index + twod1 -1];
        device_result[index + twod1 -1] += temp;
    }
}
 
void exclusive_scan(int length, int *device_result) {
    /* TODO
     * Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the data in device memory
     * The data are initialized to the inputs.  Your code should
     * do an in-place scan, generating the results in the same array.
     * This is host code -- you will need to declare one or more CUDA
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the data array is sized to accommodate the next
     * power of 2 larger than the input.
     */

    // length is the actual array size, valid_size is the value from 
    // nextPow2(length)
    int threadsPerBlock = 2;
    int valid_size = nextPow2(length);
    int num_blocks = (valid_size + threadsPerBlock -1) / threadsPerBlock;

    int step = 0;

    // stage1
    for (step = 1; step < valid_size/threadsPerBlock ; step*=2) {
        upsweep<<<num_blocks, threadsPerBlock>>>(step, device_result);
    }

    // stage2
    cudaMemset(device_result + valid_size - 1, 0, sizeof(int));
    for (step = valid_size/2; step > 0; step/=2) {
        downsweep<<<num_blocks, threadsPerBlock>>>(step, device_result);
    }
    
    return ;
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(end - inarray, device_result);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void
find_repeats_pos(int *device_input, int length, int *device_repeats_pos) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length && device_input[index] == device_input[index+1])
        device_repeats_pos[index] = 1;
    else 
        device_repeats_pos[index] = 0;
}

__global__ void
write_pos_back(int *device_repeats_pos, int length, int *device_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int resultIdx = device_repeats_pos[index];

    if (index < length && device_repeats_pos[index] != device_repeats_pos[index + 1]) {
        device_output[resultIdx] = index;
    }
}

int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */
    int threadsPerBlock = 2;
    int result = 0;
    int validSize;
    int block_num = (length + threadsPerBlock -1) / threadsPerBlock;
    int *device_repeats_pos;

    validSize = nextPow2(length);

    cudaMalloc(&device_repeats_pos, sizeof(int) * validSize);

    find_repeats_pos<<<block_num, threadsPerBlock>>> (device_input, length, device_repeats_pos);
    exclusive_scan(length, device_repeats_pos);
    write_pos_back<<<block_num, threadsPerBlock>>> (device_repeats_pos, length, device_output);
    cudaMemcpy(&result, &device_repeats_pos[length - 1], sizeof(int), cudaMemcpyDeviceToHost); /* tail of pos arr indicates total number result */
    return result;
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
