#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include "hd_encoder.h"

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

void hd_encoder_setup_device(struct hd_encoder_t * const state) {
    // nothing to do!
}

void hd_encoder_free(
    struct hd_encoder_t * const state
)
{
    free(state->ngramm_buffer);
    free(state->ngramm_sum_buffer);
    free(state->item_buffer);
    free(state->item_lookup);
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
