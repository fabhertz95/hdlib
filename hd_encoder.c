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

// Choose type of shift used. Possibilities: CIRCSHIFT_NAIVE, CIRCSHIFT_WORD, CIRCSHIFT_BIT
#define CIRCSHIFT_BIT

// TODO: optimise shifts by always shifting by 1 to the right?

void circshift_naive(block_t * const arr, const int n_blk, const int n)
{
    if (n > 0)
    {
        // shift left
        int shift = n;
        int shift_carry = sizeof(block_t) * 8 - n;
        block_t carry = arr[n_blk-1] >> shift_carry;
        block_t tmp;
        int i;
        for (i = 0; i < n_blk; i++)
        {
            tmp = arr[i];
            arr[i] = (arr[i] << shift) | carry;
            carry = tmp >> shift_carry;
        }
    }
    if (n < 0)
    {
        // shift right
        int shift = -n;
        int shift_carry = sizeof(block_t) * 8 + n;
        block_t carry = arr[n_blk-1] << shift_carry;
        block_t tmp;
        int i;
        for (i = 0; i < n_blk; i++)
        {
            tmp = arr[i];
            arr[i] = (arr[i] >> shift) | carry;
            carry = tmp << shift_carry;
        }
    }
}

void circshift_word(block_t * const arr, const int n_blk, const int n)
{
    if (n > 0)
    {
        // shift right
        int shift = n;
        block_t tmp[shift];
        memcpy(tmp, arr + (n_blk - shift), shift * sizeof arr[0]);
        memcpy(arr + shift, arr, (n_blk - shift) * sizeof arr[0]);
        memcpy(arr, tmp, shift * sizeof arr[0]);
    }
    if (n < 0)
    {
        // shift left
        int shift = -n;
        block_t tmp[shift];
        memcpy(tmp, arr, shift * sizeof arr[0]);
        memcpy(arr, arr + shift, (n_blk - shift) * sizeof arr[0]);
        memcpy(arr + (n_blk - shift), tmp, shift * sizeof arr[0]);
    }
}

void circshift_bit(block_t * const arr, const int n_blk, const int n)
{
    if (n > 0)
    {
        // shift left
        int shift = n;
        int shift_carry = sizeof(block_t) * 8 - n;
        int i;
        for (i = 0; i < n_blk; i++)
        {
            arr[i] = (arr[i] << shift) | (arr[i] >> shift_carry);
        }
    }
    if (n < 0)
    {
        // shift right
        int shift = -n;
        int shift_carry = sizeof(block_t) * 8 + n;
        int i;
        for (i = 0; i < n_blk; i++)
        {
            arr[i] = (arr[i] >> shift) | (arr[i] << shift_carry);
        }
    }
}

void circshift(block_t * const arr, const int n_blk, const int n)
{
#if defined CIRCSHIFT_NAIVE
    circshift_naive(arr, n_blk, n);
#elif defined CIRCSHIFT_WORD
    circshift_word(arr, n_blk, n);
#elif defined CIRCSHIFT_BIT
    circshift_bit(arr, n_blk, n);
#else
    circshift_word(arr, n_blk, n);
#endif
}


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
    state->ngramm_buffer = malloc(n_blk * sizeof(block_t));
    state->ngramm_sum_buffer = malloc(n_blk * sizeof(block_t) * 8 * sizeof(uint32_t));
    state->item_buffer = malloc(ngramm * n_blk * sizeof(block_t));
    state->item_buffer_head = 0;
    state->item_lookup = malloc(n_items * n_blk * sizeof(block_t));

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

// overall speedup ideas:
// - perform circular shift locally, e.g. every 32 entries
// - split HD vectors into blocks and encode large inputs in chunks, i.e.
//   for D=10000 encode the whole input using first 1000 vector elements,
//   than next 1000 etc.
void hd_encoder_encode_ngramm(
    struct hd_encoder_t * const state,
    block_t * const item
)
{
    const int n_blk = state->n_blk;
    const int ngramm = state->ngramm;
    block_t * buf = state->item_buffer;

    int *p_head = &state->item_buffer_head;

    // advance item circular buffer head
    *p_head = (*p_head + 1) % ngramm;

    // transform previous items
    int i;
    for (i = 0; i < ngramm; i++)
    {
        if (i != *p_head)
        {
            circshift(&buf[n_blk * i], n_blk, 1);
        }
    }

    // write new first entry
    memcpy(buf + n_blk * state->item_buffer_head, item, n_blk * sizeof buf[0]);

    // calculate n-gramm of all items
    memcpy(state->ngramm_buffer, buf, n_blk * sizeof buf[0]);
    for (i = 1; i != ngramm; i++)
    {
        block_t * output_iter = state->ngramm_buffer;
        block_t * buf_iter = buf + (i * n_blk);
        int j;
        for (j = 0; j < n_blk; j++)
        {
            *output_iter++ ^= *buf_iter++;
        }
    }
}

void hd_encoder_encode (
    struct hd_encoder_t * const state,
    const feature_t * const x,
    const int n_x
)
{
    const int n_blk = state->n_blk;
    const int ngramm = state->ngramm;

    memset(state->ngramm_sum_buffer, 0, sizeof(block_t) * 8 * n_blk * sizeof(state->ngramm_sum_buffer[0]));
    state->ngramm_sum_count = 0;

    memset(state->item_buffer, 0, n_blk * state->ngramm * sizeof(state->ngramm_sum_buffer[0]));

    // loop over every feature (indexed character of the text)
    int feat_idx;
    for (feat_idx = 0; feat_idx < n_x; feat_idx++) {
        // get position of item in itemMemory for current feature (indexed character)
        feature_t item_lookup_idx = x[feat_idx];

        // get pointer to item
        block_t * item = state->item_lookup + item_lookup_idx * n_blk;

        // do ngrammencoding, store temporary result in output
        hd_encoder_encode_ngramm(state, item);

        if (feat_idx >= ngramm - 1) {
            // add temporary output to sumVec
            uint32_t * ngramm_sum_buffer_iter = state->ngramm_sum_buffer;
            block_t * ngramm_buffer_iter = state->ngramm_buffer;
            int i;
            for (i = 0; i < n_blk; i++) {
                int j;
                for (j = 0; j < sizeof(block_t) * 8; j++) {
                    *ngramm_sum_buffer_iter++ += ((*ngramm_buffer_iter) >> j) & 1;
                }
                ngramm_buffer_iter++;
            }
        }
    }

    state->ngramm_sum_count += n_x - (state->ngramm - 1);
}

void hd_encoder_clip(
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
