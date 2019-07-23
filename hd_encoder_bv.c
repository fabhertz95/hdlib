#include <string.h>
#include <stdint.h>

#include "hd_encoder_bv.h"

void circshift_bv_naive(uint32_t * const dest, uint32_t * const src, const int n_blk, const int n) {
    if (n > 0) {
        // shift left
        int shift = n;
        int shift_carry = 32 - n;
        int32_t carry = src[n_blk-1] >> shift_carry;
        for (int i = 0; i < n_blk; i++) {
            dest[i] = (src[i] << shift) | carry;
            carry = src[i] >> shift_carry;
        }
    }
    if (n < 0) {
        // shift right
        int shift = -n;
        int shift_carry = 32 + n;
        int32_t carry = src[n_blk-1] << shift_carry;
        for (int i = 0; i < n_blk; i++) {
            dest[i] = (src[i] >> shift) | carry;
            carry = src[i] << shift_carry;
        }
    }
}

void circshift_inplace_bv_naive(uint32_t * const arr, const int n_blk, const int n) {
    if (n > 0) {
        // shift left
        int shift = n;
        int shift_carry = 32 - n;
        uint32_t carry = arr[n_blk-1] >> shift_carry;
        uint32_t tmp;
        for (int i = 0; i < n_blk; i++) {
            tmp = arr[i];
            arr[i] = (arr[i] << shift) | carry;
            carry = tmp >> shift_carry;
        }
    }
    if (n < 0) {
        // shift right
        int shift = -n;
        int shift_carry = 32 + n;
        uint32_t carry = arr[n_blk-1] << shift_carry;
        uint32_t tmp;
        for (int i = 0; i < n_blk; i++) {
            tmp = arr[i];
            arr[i] = (arr[i] >> shift) | carry;
            carry = tmp << shift_carry;
        }
    }
}

void circshift_bv_word(uint32_t * const dest, uint32_t * const src, const int n_blk, const int n) {
    if (n > 0)
    {
        // shift right
        int shift = n;
        memcpy(dest, src + (n_blk - shift), shift * sizeof src[0]);
        memcpy(dest + shift, src, (n_blk - shift) * sizeof src[0]);
    }
    if (n < 0)
    {
        // shift left
        int shift = -n;
        memcpy(dest + (n_blk - shift), src, shift * sizeof src[0]);
        memcpy(dest, src + shift, (n_blk - shift) * sizeof src[0]);
    }
}

void circshift_inplace_bv_word(uint32_t * const arr, const int n_blk, const int n) {
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

void circshift_bv_bit(uint32_t * const dest, uint32_t * const src, const int n_blk, const int n) {
    if (n > 0) {
        // shift left
        int shift = n;
        int shift_carry = 32 - n;
        for (int i = 0; i < n_blk; i++) {
            dest[i] = (src[i] << shift) | (src[i] >> shift_carry);
        }
    }
    if (n < 0) {
        // shift right
        int shift = -n;
        int shift_carry = 32 + n;
        for (int i = 0; i < n_blk; i++) {
            dest[i] = (src[i] >> shift) | (src[i] << shift_carry);
        }
    }
}

void circshift_inplace_bv_bit(uint32_t * const arr, const int n_blk, const int n) {
    if (n > 0) {
        // shift left
        int shift = n;
        int shift_carry = 32 - n;
        for (int i = 0; i < n_blk; i++) {
            arr[i] = (arr[i] << shift) | (arr[i] >> shift_carry);
        }
    }
    if (n < 0) {
        // shift right
        int shift = -n;
        int shift_carry = 32 + n;
        for (int i = 0; i < n_blk; i++) {
            arr[i] = (arr[i] >> shift) | (arr[i] << shift_carry);
        }
    }
}

void ngrammencoding_bv(
    uint32_t * const output,
    const int n_blk,
    const int ngramm,
    uint32_t * const block,
    uint32_t * item
)
{
    // shift current block to accommodate new entry
    // TODO: convert to a circular buffer and shift in-place
    for (int i = ngramm - 1; i != 0; --i)
    {
        circshift_bv_word(block + n_blk * i, block + n_blk * (i - 1), n_blk, 1);
    }

    // write new first entry
    memcpy(block, item, n_blk * sizeof block[0]);

    // calculate n-gramm of the block
    memcpy(output, block, n_blk * sizeof block[0]);
    for (int i = 1; i != ngramm; ++i)
    {
        uint32_t * p_output = output;
        uint32_t * p_block = block + (i * n_blk);
        for (int j = 0; j < n_blk; ++j)
        {
            *p_output++ ^= *p_block++;
        }
    }

}

void ngrammencoding_string_bv (
    uint32_t * const sumVec,     // int32_t[n_blk*32]
    const int n_blk,             // d = n_blk * 32
    const int ngramm,            // 
    const int n_feat,            // number of characters
    uint32_t * const data,       // int32_t[n_feat, n_blk]
    uint32_t * const block,      // int32_t[ngramm, n_blk]
    uint32_t * const itemMemory, // int32_t[n_chars, n_blk]
    uint32_t * const tmp_ngramm  // int32_t[n_blk]
)
{
    // loop over every feature (character of the text)
    for (int feat_idx = 0; feat_idx < n_feat; feat_idx++) {
        // get position of item in itemMemory for current feature (character)
        uint32_t char_idx = data[feat_idx];

        // get pointer to item
        uint32_t * p_item = itemMemory + (char_idx) * n_blk;

        // do ngrammencoding, store temporary result in output
        ngrammencoding_bv(tmp_ngramm, n_blk, ngramm, block, p_item);

        if (feat_idx >= ngramm - 1) {
            // add temporary output to sumVec
            uint32_t * p_sumVec = sumVec;
            uint32_t * p_tmp_ngramm = tmp_ngramm;
            for (int i = 0; i < n_blk; i++) {
                for (int j = 0; j < 32; j++) {
                    *p_sumVec++ += ((*p_tmp_ngramm) >> j) & 1;
                }
                p_tmp_ngramm++;
            }
        }
    }

}
