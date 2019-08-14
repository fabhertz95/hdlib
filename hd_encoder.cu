#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

extern "C" {
#include "hd_encoder.h"
}

#define MAX_NUM_ITEMS 32

// TODO: Stuff to optimize
// * Copy one part of x to device, then compute it and copy the next part at the same time.
//   Use different streams for each part of the input, and use cudaMemcopyAsync.
// * make clip also on the gpu, using the data from before, and don't copy the ngramm_sum_buffer over.
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
    int blk = blockIdx.x * blockDim.x + threadIdx.x;

    // exit if blk is outside of the range
    if (blk >= n_blk) {
        return;
    }

    // prepare local memory
    block_t l_ngramm_buffer[NGRAMM];
    uint32_t l_ngramm_sum_buffer[sizeof(uint32_t) * 8];
    block_t l_item_lookup[MAX_NUM_ITEMS];

    // reset ngramm_sum_buffer
    memset(l_ngramm_sum_buffer, 0, sizeof(l_ngramm_sum_buffer[0]) * 8 * sizeof(uint32_t));
    memset(l_ngramm_buffer, 0, sizeof(l_ngramm_buffer[0]) * NGRAMM);

    // load item_lookup
    uint32_t s_i; // iterator
    for (s_i = 0; s_i < n_items; s_i++) {
        l_item_lookup[s_i] = item_lookup[s_i * n_blk + blk];
    }

    // loop over every single feature
    int feat_idx;
    for (feat_idx = 0; feat_idx < n_x; feat_idx++) {
        // get position of the item in the lookup table
        feature_t item_lookup_idx = x[feat_idx];
        // get the part of the item
        block_t item = l_item_lookup[item_lookup_idx];

        // Shift the parts in in the item_buffer and add the new one
        int i;
        for (i = NGRAMM - 1; i >= 1; i--) {
            block_t previous = l_ngramm_buffer[i-1];
            l_ngramm_buffer[i] = (previous << 1) | (previous >> 31);
        }
        // set the new value
        l_ngramm_buffer[0] = item;

        // compute the encoded ngramm
        block_t tmp_ngramm_buffer = item;
        for (i = 1; i < NGRAMM; i++) {
            tmp_ngramm_buffer ^= l_ngramm_buffer[i];
        }

        // add to sum buffer
        if (feat_idx >= NGRAMM - 1) {
            int j;
            for (j = 0; j < sizeof(block_t) * 8; j++) {
                l_ngramm_sum_buffer[j] += (tmp_ngramm_buffer >> j) & 1;
            }
        }
    }

    // copy values back to ngramm_sum_buffer
    for (s_i = 0; s_i < sizeof(uint32_t) * 8; s_i++) {
        ngramm_sum_buffer[s_i * n_blk + blk] = l_ngramm_sum_buffer[s_i];
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
    // compute the number of blocks used
    int num_blocks = (state->n_blk + NUM_THREADS_IN_BLOCK - 1) / NUM_THREADS_IN_BLOCK;

    switch(state->ngramm) {
        case 2:
            hd_encoder_kernel<2><<<num_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
                state->n_blk,
                state->device.ngramm_sum_buffer,
                state->device.item_lookup,
                state->n_items,
                d_x, n_x);
            break;
        case 3:
            hd_encoder_kernel<3><<<num_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
                state->n_blk,
                state->device.ngramm_sum_buffer,
                state->device.item_lookup,
                state->n_items,
                d_x, n_x);
            break;
        case 4:
            hd_encoder_kernel<4><<<num_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
                state->n_blk,
                state->device.ngramm_sum_buffer,
                state->device.item_lookup,
                state->n_items,
                d_x, n_x);
            break;
        case 5:
            hd_encoder_kernel<5><<<num_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
                state->n_blk,
                state->device.ngramm_sum_buffer,
                state->device.item_lookup,
                state->n_items,
                d_x, n_x);
            break;
        case 6:
            hd_encoder_kernel<6><<<num_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
                state->n_blk,
                state->device.ngramm_sum_buffer,
                state->device.item_lookup,
                state->n_items,
                d_x, n_x);
            break;
        case 7:
            hd_encoder_kernel<7><<<num_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
                state->n_blk,
                state->device.ngramm_sum_buffer,
                state->device.item_lookup,
                state->n_items,
                d_x, n_x);
            break;
        case 8:
            hd_encoder_kernel<8><<<num_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
                state->n_blk,
                state->device.ngramm_sum_buffer,
                state->device.item_lookup,
                state->n_items,
                d_x, n_x);
            break;
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

    // reset the sum count
    state->ngramm_sum_count = 0;

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
    int blk;
    for (blk = 0; blk < n_blk; blk++) {
        int s_i;
        for (s_i = 0; s_i < 32; s_i++) {
            out[blk] <<= 1;
            out[blk] += ((uint32_t)(threshold - in[s_i * n_blk + blk])) >> 31;
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