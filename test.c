#include <string.h>

#include "test.h"

void circshift(int * const dest, int * const src, const int sz, const int n)
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

void circshift_inplace(int * const arr, const int sz, const int n)
{
    if (n > 0)
    {
        // shift right
        int shift = n;
        int tmp[shift];
        memcpy(tmp, arr + (sz - shift), shift * sizeof arr[0]);
        memcpy(arr + shift, arr, (sz - shift) * sizeof arr[0]);
        memcpy(arr, tmp, shift * sizeof arr[0]);
    }
    if (n < 0)
    {
        // shift left
        int shift = -n;
        int tmp[shift];
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
    int * const output,
    const int d,
    const int ngramm,
    int * const block,
    int * items,
    int * const X,
    const int start
)
{
    // shift current block to accommodate new entry
    // TODO: convert to a circular buffer and shift in-place
    for (int i = ngramm - 1; i != 0; --i)
    {
        circshift(block + d * i, block + d * (i - 1), d, 1);
    }

    // write new first entry
    memcpy(block, items + X[start] * d, d * sizeof block[0]);

    // calculate n-gramm of the block
    memcpy(output, block, d * sizeof block[0]);
    for (int i = 1; i != ngramm; ++i)
    {
        int * p_output = output;
        int * p_block = block + (i * d);
        for (int j = 0; j < d; ++j)
        {
            *p_output++ ^= *p_block++;
        }
    }

}

