#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include "hd_encoder.h"

// Choose type of shift used. Possibilities: CIRCSHIFT_NAIVE, CIRCSHIFT_WORD, CIRCSHIFT_BIT
#define CIRCSHIFT_WORD

void circshift_naive(uint32_t * const arr, const int n_blk, const int n)
{
    if (n > 0)
    {
        // shift left
        int shift = n;
        int shift_carry = 32 - n;
        uint32_t carry = arr[n_blk-1] >> shift_carry;
        uint32_t tmp;
        for (int i = 0; i < n_blk; i++)
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
        int shift_carry = 32 + n;
        uint32_t carry = arr[n_blk-1] << shift_carry;
        uint32_t tmp;
        for (int i = 0; i < n_blk; i++)
        {
            tmp = arr[i];
            arr[i] = (arr[i] >> shift) | carry;
            carry = tmp << shift_carry;
        }
    }
}

void circshift_word(uint32_t * const arr, const int n_blk, const int n)
{
    if (n > 0)
    {
        // shift right
        int shift = n;
        uint32_t tmp[shift];
        memcpy(tmp, arr + (n_blk - shift), shift * sizeof arr[0]);
        memcpy(arr + shift, arr, (n_blk - shift) * sizeof arr[0]);
        memcpy(arr, tmp, shift * sizeof arr[0]);
    }
    if (n < 0)
    {
        // shift left
        int shift = -n;
        uint32_t tmp[shift];
        memcpy(tmp, arr, shift * sizeof arr[0]);
        memcpy(arr, arr + shift, (n_blk - shift) * sizeof arr[0]);
        memcpy(arr + (n_blk - shift), tmp, shift * sizeof arr[0]);
    }
}

void circshift_bit(uint32_t * const arr, const int n_blk, const int n)
{
    if (n > 0)
    {
        // shift left
        int shift = n;
        int shift_carry = 32 - n;
        for (int i = 0; i < n_blk; i++)
        {
            arr[i] = (arr[i] << shift) | (arr[i] >> shift_carry);
        }
    }
    if (n < 0)
    {
        // shift right
        int shift = -n;
        int shift_carry = 32 + n;
        for (int i = 0; i < n_blk; i++)
        {
            arr[i] = (arr[i] >> shift) | (arr[i] << shift_carry);
        }
    }
}

void circshift(uint32_t * const arr, const int n_blk, const int n)
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
    struct hd_encoder_t * const p_state,
    const int n_blk,
    const int ngramm,
    const int n_items
)
{
    p_state->n_blk = n_blk;
    p_state->ngramm = ngramm;
    p_state->ngramm_buffer = malloc(n_blk * sizeof(uint32_t));
    p_state->ngramm_sum_buffer = malloc(n_blk * 32 * sizeof(uint32_t));
    p_state->item_buffer = malloc(ngramm * n_blk * sizeof(uint32_t));
    p_state->item_buffer_head = 0;
    p_state->item_lookup = malloc(n_items * n_blk * sizeof(uint32_t));

    // initialise HD vector lookup table with uniformly distributed 0s and 1s
    for (int i = 0; i < n_items * n_blk; ++i)
        // TODO RAND_MAX may be lower than 1<<32 (on my libux laptop, it is 1<<31).
        // By computing two random numbers, and using their 16 lowest bits, I think
        // this function remains portable.
        p_state->item_lookup[i] = ((rand() % (1<<16)) << 16) + (rand() % (1<<16));
}

// overall speedup ideas:
// - perform circular shift locally, e.g. every 32 entries
// - split HD vectors into blocks and encode large inputs in chunks, i.e.
//   for D=10000 encode the whole input using first 1000 vector elements,
//   than next 1000 etc.
// - binary HD vectors
void hd_encoder_encode_ngramm(
    struct hd_encoder_t * const p_state,
    uint32_t * const item
)
{
    const int n_blk = p_state->n_blk;
    const int ngramm = p_state->ngramm;
    uint32_t * buf = p_state->item_buffer;

    int *p_head = &p_state->item_buffer_head;

    // advance item circular buffer head
    *p_head = (*p_head + 1) % ngramm;

    // transform previous items
    for (int i = 0; i < ngramm; i++)
    {
        if (i != *p_head)
        {
            circshift(&buf[n_blk * i], n_blk, 1);
        }
    }

    // write new first entry
    memcpy(buf + n_blk * p_state->item_buffer_head, item, n_blk * sizeof buf[0]);

    // calculate n-gramm of all items
    memcpy(p_state->ngramm_buffer, buf, n_blk * sizeof buf[0]);
    for (int i = 1; i != ngramm; i++)
    {
        uint32_t * output_iter = p_state->ngramm_buffer;
        uint32_t * buf_iter = buf + (i * n_blk);
        for (int j = 0; j < n_blk; j++)
        {
            *output_iter++ ^= *buf_iter++;
        }
    }

}

void hd_encoder_encode (
    struct hd_encoder_t * const p_state,
    uint32_t * const data,
    const int n_data
)
{
    const int n_blk = p_state->n_blk;
    const int ngramm = p_state->ngramm;

    memset(p_state->ngramm_sum_buffer, 0, 32 * n_blk * sizeof(p_state->ngramm_sum_buffer[0]));
    p_state->ngramm_sum_count = 0;

    memset(p_state->item_buffer, 0, n_blk * p_state->ngramm * sizeof(p_state->ngramm_sum_buffer[0]));

    // loop over every feature (character of the text)
    for (int feat_idx = 0; feat_idx < n_data; feat_idx++) {
        // get position of item in itemMemory for current feature (character)
        uint32_t char_idx = data[feat_idx];

        // get pointer to item
        uint32_t * p_item = p_state->item_lookup + (char_idx) * n_blk;

        // do ngrammencoding, store temporary result in output
        hd_encoder_encode_ngramm(p_state, p_item);

        if (feat_idx >= ngramm - 1) {
            // add temporary output to sumVec
            uint32_t * p_sumVec = p_state->ngramm_sum_buffer;
            uint32_t * p_tmp_ngramm = p_state->ngramm_buffer;
            for (int i = 0; i < n_blk; i++) {
                for (int j = 0; j < 32; j++) {
                    *p_sumVec++ += ((*p_tmp_ngramm) >> j) & 1;
                }
                p_tmp_ngramm++;
            }
        }
    }

    p_state->ngramm_sum_count += n_data - (p_state->ngramm - 1);
}

void hd_encoder_clip(
    struct hd_encoder_t * const p_state
)
{
    // add a random vector to break ties if case an odd number of elements were summed
    if (p_state->ngramm_sum_count % 2 == 0)
    {
        for (int i = 0; i < 32 * p_state->n_blk; i++)
        {
            p_state->ngramm_sum_buffer[i] += rand() % 2;
        }
        p_state->ngramm_sum_count++;
    }

    int threshold = p_state->ngramm_sum_count / 2;

    for (int i = 0; i < 32 * p_state->n_blk; i++)
    {
        // set to 1 if above threshold and 0 otherwise
        p_state->ngramm_sum_buffer[i] = ((uint32_t)(threshold - p_state->ngramm_sum_buffer[i])) >> 31;
    }
    p_state->ngramm_sum_count = 1;
}
