#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

extern "C" {
#include "hd_encoder.h"
}

// number of threads per block in the grid
#define NUM_THREADS_IN_BLOCK 128
#define NUM_INPUT_CHUNKS 2

#define NUM_HD_BLOCKS_IN_BLOCK (NUM_THREADS_IN_BLOCK / NUM_INPUT_CHUNKS)

#define MAX_NUM_ITEMS 32

// TODO: Stuff to optimize
// * Copy one part of x to device, then compute it and copy the next part at the same time.
//   Use different streams for each part of the input, and use cudaMemcopyAsync.
// * make clip also on the gpu, using the data from before, and don't copy the ngramm_sum_buffer over.

// encode the whole input with a chunk of the HD vector (a single)
template<int NGRAMM>
__global__ void hd_encoder_kernel(
    const int n_blk,
    uint32_t * __restrict__ ngramm_sum_buffer,
    const block_t * __restrict__ item_lookup,
    const int n_items,
    const feature_t * __restrict__ x,
    const int n_x
)
{
    // compute the index of the block on which we must work
    int blk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x_chunk_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // exit if blk_idx is outside of the range
    if (blk_idx >= n_blk || x_chunk_idx >= NUM_INPUT_CHUNKS) {
        return;
    }

    // there are (NUM_INPUT_CHUNKS - 1) overlaps of (NGRAMM - 1),
    // but the number of XOR calculations only depends on
    // the number of outputs to the accummulation buffer

    // example: n_x = 9, ngramm = 3, NUM_INPUT_CHUNKS = 2
    //          xored_n_x := 7, x_chunk_len = 4 (3 for last chunk)
    //
    // o - load only, x - load and xor
    // chunk 0: ooxxxx---
    // chunk 1: ----ooxxx
    //            <--> chunk 0 length (pre-loaded data not included)
    const int xored_n_x = n_x - (NGRAMM - 1);
    int x_chunk_len = (xored_n_x + NUM_INPUT_CHUNKS - 1) / NUM_INPUT_CHUNKS;
    const int x_chunk_start = x_chunk_idx * x_chunk_len;
    if (x_chunk_start + x_chunk_len > n_x) {
        x_chunk_len = n_x - x_chunk_start;
    }

    int i; // iterator

    // local copies of the:
    // - HD feature vector chunk n-gramm buffer
    block_t l_item_buffer[NGRAMM];
    memset(l_item_buffer, 0, sizeof(l_item_buffer));

    // - encoded n-gramm summation chunk buffer
    uint32_t l_ngramm_sum_buffer[sizeof(block_t) * 8];
    memset(l_ngramm_sum_buffer, 0, sizeof(l_ngramm_sum_buffer));

    // - HD vector chunk lookup array
    block_t l_item_lookup[MAX_NUM_ITEMS];
    for (i = 0; i < n_items; i++) {
        l_item_lookup[i] = item_lookup[i * n_blk + blk_idx];
    }

    // loop over every single feature
    int x_chunk_iter;
    for (x_chunk_iter = 0; x_chunk_iter < NGRAMM - 1 + x_chunk_len; x_chunk_iter++) {
        // barrel shift each HD feature vector chunk as it gets a feature increment older
        int i;
        for (i = NGRAMM - 1; i >= 1; i--) {
            block_t previous = l_item_buffer[i-1];
            l_item_buffer[i] = (previous << 1) | (previous >> 31);
        }

        // populate new HD feature vector chunk
        feature_t item_lookup_idx = x[x_chunk_start + x_chunk_iter];
        block_t item = l_item_lookup[item_lookup_idx];
        l_item_buffer[0] = item;

        // only pre-load the first (NGRAMM - 1) items
        if (x_chunk_iter >= NGRAMM - 1) {
            // compute the encoded n-gramm
            block_t tmp_ngramm_buffer = item;
            for (i = 1; i < NGRAMM; i++) {
                tmp_ngramm_buffer ^= l_item_buffer[i];
            }
    
            // unpack and accumulate the encoded n-gramm
            for (i = 0; i < sizeof(block_t) * 8; i++) {
                l_ngramm_sum_buffer[i] += (tmp_ngramm_buffer >> i) & 1;
            }
        }
    }

    // accumulating the results to the ngramm_sum_buffer creates a memory race condition
    // avoid this by means of linear reduction across threads
    // TODO implement a better reduction
    int curr_x_chunk;
    for (curr_x_chunk = 0; curr_x_chunk < NUM_INPUT_CHUNKS; curr_x_chunk++) {
        __syncthreads();
        if (curr_x_chunk == x_chunk_idx) {
            // copy values back to ngramm_sum_buffer
            for (i = 0; i < sizeof(block_t) * 8; i++) {
                ngramm_sum_buffer[i * n_blk + blk_idx] += l_ngramm_sum_buffer[i];
            }
        }
    }
}

// Wrapper function to call the kernel. Input data (x) must already be copied to the device.
// if stream is NULL, then the default stream is used.
extern "C" void hd_encoder_call_kernel(
    struct hd_encoder_t * const state,
    const feature_t * d_x,
    const int n_x,
    cudaStream_t stream = NULL
)
{
    // Each grid block calculates a chunk of the HD vector
    // for the entire input. Withing the block, threads divide work
    // both along the HD vector and the input.
    dim3 threads(NUM_THREADS_IN_BLOCK / NUM_INPUT_CHUNKS, NUM_INPUT_CHUNKS);

    // compute the number of blocks used
    int num_blocks = (state->n_blk + NUM_HD_BLOCKS_IN_BLOCK - 1) / NUM_HD_BLOCKS_IN_BLOCK;

    dim3 grid(num_blocks);

    switch(state->ngramm) {
#define CALL_KERNEL_CASE(N) \
        case N: \
            hd_encoder_kernel<N><<<grid, threads, 0, stream>>>( \
                state->n_blk, \
                state->device.ngramm_sum_buffer, \
                state->device.item_lookup, \
                state->n_items, \
                d_x, n_x); \
            break;

        CALL_KERNEL_CASE(2)
        CALL_KERNEL_CASE(3)
        CALL_KERNEL_CASE(4)
        CALL_KERNEL_CASE(5)
        CALL_KERNEL_CASE(6)
        CALL_KERNEL_CASE(7)
        CALL_KERNEL_CASE(8)

        default:
            printf("Error! ngramm must be between 2 and 8, but it was %d\n", state->ngramm);
    }
}

extern "C" void hd_encoder_setup_device(struct hd_encoder_t * const state) {
    // allocate memory
    cudaMalloc(&(state->device.item_lookup), state->n_items * state->n_blk * sizeof(block_t));
    cudaMalloc(&(state->device.ngramm_sum_buffer), state->n_blk * sizeof(block_t) * 8 * sizeof(uint32_t));
    cudaMalloc(&(state->device.ngramm_buffer), state->ngramm * state->n_blk * sizeof(block_t));

    // copy LUT to device
    cudaMemcpy(
        state->device.item_lookup,
        state->item_lookup,
        state->n_items * state->n_blk * sizeof(block_t),
        cudaMemcpyHostToDevice
    );
}

extern "C" void hd_encoder_free(struct hd_encoder_t * const state) {
    cudaFree(state->device.item_lookup);
    cudaFree(state->device.ngramm_sum_buffer);
    cudaFree(state->device.ngramm_buffer);

    free(state->ngramm_buffer);
    free(state->ngramm_sum_buffer);
    free(state->item_buffer);
    free(state->item_lookup);

    cudaDeviceReset();
}

extern "C" void hd_encoder_encode (
    struct hd_encoder_t * const state,
    const feature_t * const x,
    const int n_x
)
{
    const int n_blk = state->n_blk;

    // reset the sum count and buffer
    state->ngramm_sum_count = 0;
    cudaMemset(
        state->device.ngramm_sum_buffer,
        0,
        n_blk * sizeof(block_t) * 8 * sizeof(uint32_t)
    );

    // allocate input data memory on the device
    feature_t * d_x;
    cudaMalloc(&d_x, n_x * sizeof(feature_t));
    // copy the input data
    cudaMemcpy(d_x, x, n_x * sizeof(feature_t), cudaMemcpyHostToDevice);

    // call the kernel
    hd_encoder_call_kernel(state, d_x, n_x);

    cudaDeviceSynchronize();

    // copy the output (ngramm_sum_buffer) back from the device
    cudaMemcpy(
        state->ngramm_sum_buffer,
        state->device.ngramm_sum_buffer,
        n_blk * sizeof(block_t) * 8 * sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    );

    // free input memory
    cudaFree(d_x);

    // set the ngramm_sum_count
    state->ngramm_sum_count += n_x - (state->ngramm - 1);
}

void clip(
    const uint32_t * const in,
    const int n_in,
    const int count,
    block_t * const out
)
{
    int threshold = count / 2;

    memset(out, 0, (n_in + sizeof(block_t) * 8 - 1) / (sizeof(block_t) * 8));

    // we ignore the randomization here...

    int n_blk = n_in / 32;
    int blk_idx;
    for (blk_idx = 0; blk_idx < n_blk; blk_idx++) {
        int i;
        for (i = 0; i < 32; i++) {
            out[blk_idx] <<= 1;
            out[blk_idx] += ((uint32_t)(threshold - in[i * n_blk + blk_idx])) >> 31;
        }
    }

}

void hd_encoder_clip(
    struct hd_encoder_t * const state
)
{
    clip(
        state->ngramm_sum_buffer,
        sizeof(block_t) * 8 * state->n_blk,
        state->ngramm_sum_count,
        state->ngramm_buffer
    );
}