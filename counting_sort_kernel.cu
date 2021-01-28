/* Write GPU code to perform the step(s) involved in counting sort. 
 Add additional kernels and device functions as needed. */
#define MAX_KERNEL_WIDTH 256
#define HISTOGRAM_SIZE 256
__constant__ int kernel_c[MAX_KERNEL_WIDTH];

__global__ void counting_sort_kernel(int* in, int* bins, int num_elements, int histogram_size)
{
    __shared__ int s[HISTOGRAM_SIZE];
    /* Initialize shared memory */ 
    if(threadIdx.x < histogram_size)
        s[threadIdx.x] = 0;
		
    __syncthreads();

    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
	
    while (offset < num_elements) {
        atomicAdd(&s[in[offset]], 1);
        offset += stride;
    }	  
	
    __syncthreads();

    /* Accumulate histogram in shared memory into global memory */
    if (threadIdx.x < histogram_size) 
        atomicAdd(&bins[threadIdx.x], s[threadIdx.x]);
    return;
}

__global__ void scan_kernel(int *out, int *in, int n)
{
    /* Dynamically allocated shared memory for storing the scan array */
    extern  __shared__  int temp[];

    int tid = threadIdx.x;

    /* Indices for the ping-pong buffers */
    int pout = 0;
    int pin = 1;

    /* Load the in array from global memory into shared memory */
    if (tid > 0) 
        temp[pout * n + tid] = in[tid - 1];
    else
        temp[pout * n + tid] = 0;

    int offset;
    for (offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;
        pin  = 1 - pout;
        __syncthreads();

        temp[pout * n + tid] = temp[pin * n + tid];

        if (tid >= offset)
            temp[pout * n + tid] += temp[pin * n + tid - offset];
    }

    __syncthreads();
    out[tid] = temp[pout * n + tid];
}


/* scan_d stored in kernel_c */
__global__ void recreate_array_kernel(int* output, int num_elements, int histogram_size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int curr_bin = 0;
    int next_bin = 1;

    /* 
        Stride over the output element
        stride over the bins at the same time
        if the bin value needs writing to the current offset, write and stride
        else move to next bin

        runtime = O(bin_size + num_element / total_threads)
    */

    /* 
        if b[prev] =< i < b[curr] ::=> Write+stride
        else ::=> bin++
    */

    while((offset < num_elements) || (curr_bin < histogram_size)){
        if( ( offset >= kernel_c[curr_bin] ) && ( offset < kernel_c[next_bin] ) ){
            output[offset] = curr_bin;
            offset += stride;
        }else{
            curr_bin = next_bin;
            next_bin++;
        }
    }
    


}