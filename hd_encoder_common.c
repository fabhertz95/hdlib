#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include "hd_encoder.h"

// rand() generates a random number between 0 and RAND_MAX, which is
// guaranteed to be no less than 32767 on any standard implementation.
#if (RAND_MAX >= (1u << 32) - 1u)
#define RAND_BYTES 4
#elif (RAND_MAX >= (1u << 16) - 1u)
#define RAND_BYTES 2
#elif (RAND_MAX >= (1u << 8) - 1u)
#define RAND_BYTES 1
#endif

void hd_encoder_init(
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

void clip(
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
