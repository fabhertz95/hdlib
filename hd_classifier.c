#include <stdint.h>

#include "hd_classifier.h"

void hd_classifier_init(
    struct hd_classifier_t * const state
)
{

}

char hamming_distance_lookup[1 << 8];

void hamming_distance_init()
{
    for (int i = 0; i < 1 << 8; i++)
    {
        int tmp = i;
        hamming_distance_lookup[i] = 0;
        for (int j = 0; j < 8; j++)
        {
            hamming_distance_lookup[i] += tmp & 1;
            tmp >>= 1;
        }
    }
}

int hamming_distance(
    const void * const a,
    const void * const b,
    const int n
)
{
    int result = 0;
    const uint8_t * a_iter = a;
    const uint8_t * b_iter = b;

    for (int i = 0; i < n; i++)
    {
        result += hamming_distance_lookup[*a_iter++ ^ *b_iter++];
    }

    return result;
}
