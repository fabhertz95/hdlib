#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include "hd_encoder.h"

void circshift(int32_t * const dest, int32_t * const src, const int sz, const int n);

void circshift_inplace(int32_t * const arr, const int sz, const int n);

void circshift(int32_t * const dest, int32_t * const src, const int sz, const int n)
{
    if (n > 0)
    {
        // shift right
        int shift = n;
        memcpy(dest, src + (sz - shift), shift * sizeof src[0]);
        memcpy(dest + shift, src, (sz - shift) * sizeof src[0]);
    }
    if (n < 0)
    {
        // shift left
        int shift = -n;
        memcpy(dest + (sz - shift), src, shift * sizeof src[0]);
        memcpy(dest, src + shift, (sz - shift) * sizeof src[0]);
    }
}

void circshift_inplace(int32_t * const arr, const int sz, const int n)
{
    if (n > 0)
    {
        // shift right
        int shift = n;
        int32_t tmp[shift];
        memcpy(tmp, arr + (sz - shift), shift * sizeof arr[0]);
        memcpy(arr + shift, arr, (sz - shift) * sizeof arr[0]);
        memcpy(arr, tmp, shift * sizeof arr[0]);
    }
    if (n < 0)
    {
        // shift left
        int shift = -n;
        int32_t tmp[shift];
        memcpy(tmp, arr, shift * sizeof arr[0]);
        memcpy(arr, arr + shift, (sz - shift) * sizeof arr[0]);
        memcpy(arr + (sz - shift), tmp, shift * sizeof arr[0]);
    }
}

void hd_encoder_init(
    struct hd_encoder_t * const p_state,
    const int d,
    const int ngramm,
    const int n_items
)
{
    p_state->d = d;
    p_state->ngramm = ngramm;
    p_state->ngramm_buffer = malloc(d * sizeof(int32_t));
    p_state->encoder_buffer = malloc(d * sizeof(int32_t));
    p_state->item_buffer = malloc(ngramm * d * sizeof(int32_t));
    p_state->item_lookup = malloc(n_items * d * sizeof(int32_t));

    // Initialise HD vector lookup table with uniformly distributed 0s and 1s
    for (int i = 0; i < n_items * d; ++i)
        p_state->item_lookup[i] = rand() % 2;
}

// overall speedup ideas:
// - perform circular shift locally, e.g. every 32 entries
// - split HD vectors into blocks and encode large inputs in chunks, i.e.
//   for D=10000 encode the whole input using first 1000 vector elements,
//   than next 1000 etc.
// - binary HD vectors
void ngrammencoding(
    const struct hd_encoder_t * const p_state,
    int32_t * const item
)
{
    const int d = p_state->d;
    const int ngramm = p_state->ngramm;
    int32_t * block = p_state->item_buffer;

    // shift current block to accommodate new entry
    // TODO: convert to a circular buffer and shift in-place
    for (int i = ngramm - 1; i != 0; --i)
    {
        circshift(&block[d * i], block + d * (i - 1), d, 1);
    }

    // write new first entry
    memcpy(block, item, d * sizeof block[0]);

    // calculate n-gramm of the block
    memcpy(p_state->ngramm_buffer, block, d * sizeof block[0]);
    for (int i = 1; i != ngramm; i++)
    {
        int32_t * p_output = p_state->ngramm_buffer;
        int32_t * p_block = block + (i * d);
        for (int j = 0; j < d; j++)
        {
            *p_output++ ^= *p_block++;
        }
    }

}

void ngrammencoding_string (
    struct hd_encoder_t * const p_state,
    int32_t * const data,
    const int n_data
)
{
    const int d = p_state->d;
    const int ngramm = p_state->ngramm;

    memset(p_state->encoder_buffer, 0, p_state->d * sizeof(p_state->encoder_buffer[0]));
    p_state->encoder_count = 0;

    memset(p_state->item_buffer, 0, p_state->d * p_state->ngramm * sizeof(p_state->encoder_buffer[0]));

    // loop over every feature (character of the text)
    for (int feat_idx = 0; feat_idx < n_data; feat_idx++) {
        // get position of item in itemMemory for current feature (character)
        int32_t char_idx = data[feat_idx];

        // get pointer to item
        int32_t * p_item = p_state->item_lookup + (char_idx) * d;

        // do ngrammencoding, store temporary result in output
        ngrammencoding(p_state, p_item);

        if (feat_idx >= ngramm - 1) {
            // add temporary output to sumVec
            int32_t * p_sumVec = p_state->encoder_buffer;
            int32_t * p_tmp_ngramm = p_state->ngramm_buffer;
            for (int j = 0; j < d; j++) {
                *p_sumVec++ += *p_tmp_ngramm++;
            }
        }
    }

    p_state->encoder_count += n_data - (p_state->ngramm - 1);
}

void hd_encoder_clip(
    struct hd_encoder_t * const p_state
)
{
    // Add a random vector to break ties if case an odd number of elements were summed
    if (p_state->encoder_count % 2 == 0)
    {
        for (int i = 0; i < p_state->d; i++)
        {
            p_state->encoder_buffer[i] += rand() % 2;
        }
        p_state->encoder_count++;
    }

    int threshold = p_state->encoder_count / 2;

    for (int i = 0; i < p_state->d; i++)
    {
        // Set to 1 if above threshold and 0 otherwise
        p_state->encoder_buffer[i] = ((uint32_t)(threshold - p_state->encoder_buffer[i])) >> 31;
    }
    p_state->encoder_count = 1;
}
