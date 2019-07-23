#include <string.h>
#include <stdint.h>

#include "hd_encoder.h"

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

// overall speedup ideas:
// - perform circular shift locally, e.g. every 32 entries
// - split HD vectors into blocks and encode large inputs in chunks, i.e.
//   for D=10000 encode the whole input using first 1000 vector elements,
//   than next 1000 etc.
// - binary HD vectors
void ngrammencoding(
    int32_t * const output,
    const int d,
    const int ngramm,
    int32_t * const block,
    int32_t * item
)
{
    // shift current block to accommodate new entry
    // TODO: convert to a circular buffer and shift in-place
    for (int i = ngramm - 1; i != 0; --i)
    {
        circshift(block + d * i, block + d * (i - 1), d, 1);
    }

    // write new first entry
    memcpy(block, item, d * sizeof block[0]);

    // calculate n-gramm of the block
    memcpy(output, block, d * sizeof block[0]);
    for (int i = 1; i != ngramm; ++i)
    {
        int32_t * p_output = output;
        int32_t * p_block = block + (i * d);
        for (int j = 0; j < d; ++j)
        {
            *p_output++ ^= *p_block++;
        }
    }

}

void ngrammencoding_string (
    int32_t * const sumVec,
    const int d,
    const int ngramm,
    const int n_feat,
    int32_t * const data,
    int32_t * const block,
    int32_t * const itemMemory,
    int32_t * const tmp_ngramm
)
{
    // loop over every feature (character of the text)
    for (int feat_idx = 0; feat_idx < n_feat; feat_idx++) {
        // get position of item in itemMemory for current feature (character)
        int32_t char_idx = data[feat_idx];

        // get pointer to item
        int32_t * p_item = itemMemory + (char_idx) * d;

        // do ngrammencoding, store temporary result in output
        ngrammencoding(tmp_ngramm, d, ngramm, block, p_item);

        if (feat_idx >= ngramm - 1) {
            // add temporary output to sumVec
            int32_t * p_sumVec = sumVec;
            int32_t * p_tmp_ngramm = tmp_ngramm;
            for (int j = 0; j < d; j++) {
                *p_sumVec++ += *p_tmp_ngramm++;
            }
        }
    }

}
