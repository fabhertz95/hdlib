#include <string.h>
#include <stdint.h>
#include <stdlib.h>

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

#define MAX_NUM_ITEMS 32
#define MAX_NGRAMM 8

// TODO: Stuff to optimize
// * Copy one part of x to device, then compute it and copy the next part at the same time.
//   Use different streams for each part of the input, and use cudaMemcopyAsync.
// * make clip also on the gpu, using the data from before, and don't copy the ngramm_sum_buffer over.
__global__ void hd_encoder_kernel(
    const int n_blk,
    const int ngramm,
    uint32_t * __restrict__ ngramm_sum_buffer,
    const block_t * __restrict__ item_lookup,
    const int n_items,
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

    // prepare local memory
    block_t l_ngramm_buffer[MAX_NGRAMM];
    uint32_t l_ngramm_sum_buffer[sizeof(uint32_t) * 8];
    block_t l_item_lookup[MAX_NUM_ITEMS];
    
    // reset ngramm_sum_buffer
    memset(l_ngramm_sum_buffer, 0, sizeof(l_ngramm_sum_buffer[0]) * 8 * sizeof(uint32_t));
    memset(l_ngramm_buffer, 0, sizeof(l_ngramm_buffer[0]) * ngramm);

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
        for (i = ngramm - 1; i >= 1; i--) {
            block_t previous = l_ngramm_buffer[i-1];
            l_ngramm_buffer[i] = (previous << 1) | (previous >> 31);
        }
        // set the new value
        l_ngramm_buffer[0] = item;

        // compute the encoded ngramm
        block_t tmp_ngramm_buffer = item;
        for (i = 1; i < ngramm; i++) {
            tmp_ngramm_buffer ^= l_ngramm_buffer[i];
        }

        // add to sum buffer
        if (feat_idx >= ngramm - 1) {
            int j;
            for (j = 0; j < sizeof(block_t) * 8; j++) {
                l_ngramm_sum_buffer[j] += (tmp_ngramm_buffer >> j) & 1;
            }
        }
    }

    // copy values back to ngramm_sum_buffer
    // TODO reorder sum buffer such that we can just use memcopy
    for (s_i = 0; s_i < sizeof(uint32_t) * 8; s_i++) {
        ngramm_sum_buffer[blk * sizeof(uint32_t) * 8 + s_i] = l_ngramm_sum_buffer[s_i];
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

    // reset the sum count
    state->ngramm_sum_count = 0;

    // allocate input data memory on the device
    feature_t * d_x;
    cudaMalloc(&d_x, n_x * sizeof(feature_t));
    // copy the input data
    cudaMemcpy(d_x, x, n_x * sizeof(feature_t), cudaMemcpyHostToDevice);

    // call the kernel
    int num_blocks = (n_blk + NUM_THREADS_IN_BLOCK - 1) / NUM_THREADS_IN_BLOCK;

    hd_encoder_kernel<<<num_blocks, NUM_THREADS_IN_BLOCK>>>(
        n_blk,
        ngramm,
        state->device.ngramm_sum_buffer,
        state->device.item_lookup,
        n_items,
        d_x,
        n_x
    );

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
