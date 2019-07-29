#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

extern "C" {
#include "hd_encoder.h"
}

// rand() generates a random number between 0 and RAND_MAX, which is
// guaranteed to be no less than 32767 on any standard implementation.
#if (RAND_MAX >= (1u << 32) - 1u)
#define RAND_BYTES 4
#elif (RAND_MAX >= (1u << 16) - 1u)
#define RAND_BYTES 2
#elif (RAND_MAX >= (1u << 8) - 1u)
#define RAND_BYTES 1
#endif

// number of threads per block in the grid
#define NUM_THREADS_IN_BLOCK 128

__global__ void hd_encoder_kernel(
    const int n_blk,
    const int ngramm,
    uint32_t * __restrict__ ngramm_sum_buffer,
    const block_t * __restrict__ item_lookup,
    block_t * __restrict__ item_buffer,
    const feature_t * __restrict__ x,
    const int n_x
)
{
    // compute the index of the block on which we must work
    int blk = blockIdx.x*blockDim.x + threadIdx.x;

    // exit if blk is outside of the range
    if (blk >= n_blk) {
        return;
    }

    // loop over every single feature
    int feat_idx;
    for (feat_idx = 0; feat_idx < n_x; feat_idx++) {
        // get position of the item in the lookup table
        feature_t item_lookup_idx = x[feat_idx];
        // get the part of the item
        block_t item = *(item_lookup + item_lookup_idx * n_blk + blk);

        // Shift the parts in in the item_buffer and add the new one
        int i;
        for (i = ngramm - 1; i >= 1; i--) {
            block_t previous = item_buffer[(i-1) * n_blk + blk];
            item_buffer[i * n_blk + blk] = (previous << 1) | (previous >> 31);
        }
        // set the new value
        item_buffer[blk] = item;

        // compute the encoded ngramm
        block_t tmp_ngramm_buffer = item;
        for (i = 1; i < ngramm; i++) {
            tmp_ngramm_buffer ^= item_buffer[i * n_blk + blk];
        }

        // add to sum buffer
        if (feat_idx >= ngramm - 1) {
            uint32_t * ngramm_sum_buffer_iter = ngramm_sum_buffer + blk * sizeof(block_t) * 8;
            int j;
            for (j = 0; j < sizeof(block_t) * 8; j++) {
                *ngramm_sum_buffer_iter++ += ((tmp_ngramm_buffer) >> j) & 1;
            }
        }
    }
}

extern "C" void hd_encoder_init(
    struct hd_encoder_t * const state,
    const int n_blk,
    const int ngramm,
    const int n_items
)
{
    state->n_blk = n_blk;
    state->ngramm = ngramm;
    state->n_items = n_items;
    state->ngramm_buffer = (block_t*)malloc(n_blk * sizeof(block_t));
    state->ngramm_sum_buffer = (uint32_t*)malloc(n_blk * sizeof(block_t) * 8 * sizeof(uint32_t));
    state->item_buffer = (block_t*)malloc(ngramm * n_blk * sizeof(block_t));
    state->item_buffer_head = 0;
    state->item_lookup = (block_t*)malloc(n_items * n_blk * sizeof(block_t));

    // initialise HD vector lookup table with uniformly distributed 0s and 1s
    int i;
    for (i = 0; i < n_items * n_blk; ++i)
    {
        state->item_lookup[i] = 0;

        int j;
        for (j = 0; j < sizeof(state->item_lookup[0]) / RAND_BYTES; j++)
        {
            state->item_lookup[i] <<= 8 * RAND_BYTES;
            state->item_lookup[i] += rand() & ((1u << 8 * RAND_BYTES) - 1u);
        }
    }
}

extern "C" void hd_encoder_encode (
    struct hd_encoder_t * const state,
    const feature_t * const x,
    const int n_x
)
{
    const int n_blk = state->n_blk;
    const int ngramm = state->ngramm;
    const int n_items = state->n_items;

    // allocate memory on the device
    uint32_t * d_ngramm_sum_buffer;
    block_t * d_item_buffer;
    feature_t * d_x;
    printf("allocate ngramm_sum_buffer: %d\n", n_blk * sizeof(block_t) * 8);
    // TODO hangs here
    cudaMalloc(&d_ngramm_sum_buffer, n_blk * sizeof(block_t) * 8);
    printf("allocate item_buffer: %d\n", ngramm * n_blk * sizeof(block_t));
    cudaMalloc(&d_item_buffer, ngramm * n_blk * sizeof(block_t));
    printf("allocate x: %d\n", n_x * sizeof(feature_t));
    cudaMalloc(&d_x, n_x * sizeof(feature_t));

    // TODO allocate and copy these values in some init function, because they will remain constant for all samples
    block_t * d_item_lookup;
    printf("allocate item_lookup: %d\n", n_items * n_blk * sizeof(block_t));
    cudaMalloc(&d_item_lookup, n_items * n_blk * sizeof(block_t));

    // reset sum buffer and item buffer
    cudaMemset(d_ngramm_sum_buffer, 0, n_blk * sizeof(block_t) * 8);
    cudaMemset(d_item_buffer, 0, ngramm * n_blk * sizeof(block_t));

    // copy the item lookup
    // TODO allocate and copy these values in some init function, because they will remain constant for all samples
    cudaMemcpy(d_item_lookup, state->item_lookup, n_items * n_blk * sizeof(block_t), cudaMemcpyHostToDevice);
    // copy the input data
    cudaMemcpy(d_x, x, n_x * sizeof(feature_t), cudaMemcpyHostToDevice);

    // call the kernel
    printf("call kernel\n");
    int num_blocks = (n_blk + NUM_THREADS_IN_BLOCK - 1) / NUM_THREADS_IN_BLOCK;
    hd_encoder_kernel<<<num_blocks, NUM_THREADS_IN_BLOCK>>>(
        n_blk,
        ngramm,
        d_ngramm_sum_buffer,
        d_item_lookup,
        d_item_buffer,
        d_x,
        n_x
    );

    // copy the output (ngramm_sum_buffer) back from the device
    cudaMemcpy(state->ngramm_sum_buffer, d_ngramm_sum_buffer, n_blk * sizeof(block_t) * 8, cudaMemcpyDeviceToHost);

    // free all memory
    cudaFree(d_ngramm_sum_buffer);
    cudaFree(d_item_buffer);
    cudaFree(d_item_lookup);
    cudaFree(d_x);

    // set the ngramm_sum_count
    state->ngramm_sum_count += n_x - (state->ngramm - 1);
}

extern "C" void hd_encoder_clip(
    const uint32_t * const in,
    const int n_in,
    const int count,
    block_t * const out
)
{
    int threshold = count / 2;

    memset(out, 0, (n_in + sizeof(block_t) * 8 - 1) / (sizeof(block_t) * 8));

    // add a random vector to break ties if case an even number of elements were summed
    if (count % 2 == 0)
    {
        // TODO: can we reuse randomness? e.g. have a fixed length of say 32 bytes
        uint32_t random_vector[(n_in + 31) / 32];
        int i;
        for (i = 0; i > sizeof(random_vector) / sizeof(random_vector[0]); i++)
        {
            random_vector[i] = 0;
            int j;
            for (j = 0; j < RAND_BYTES; j++)
            {
                random_vector[i] <<= 8 * RAND_BYTES;
                random_vector[i] += rand() & ((1u << 8 * RAND_BYTES) - 1u);
            }
        }

        for (i = 0; i < n_in; i++)
        {
            int in_with_rand = in[i] + (random_vector[i / 32] & 1);
            random_vector[i / 32] >>= 1;
            out[i / 32] <<= 1;
            // set to 1 if above threshold and 0 otherwise
            out[i / 32] += ((uint32_t)(threshold - in_with_rand)) >> 31;
        }
    }
    else
    {
        int i;
        for (i = 0; i < n_in; i++)
        {
            out[i / 32] <<= 1;
            out[i / 32] += ((uint32_t)(threshold - in[i])) >> 31;
        }
    }

}
