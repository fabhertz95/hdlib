#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#include "hd_encoder.h"
#include "hd_classifier.h"

void hd_classifier_init(
    struct hd_classifier_t * const state,
    const int n_blk,
    const int n_class
)
{
    state->n_blk = n_blk;
    state->n_class = n_class;
    // state->am = malloc(n_class * n_blk * sizeof(block_t));
}

class_t hd_classifier_predict(
    const struct hd_classifier_t * const state,
    struct hd_encoder_t * const encoder_state,
    const feature_t * const x,
    const int n_x
)
{
    hd_encoder_encode(encoder_state, x, n_x);
    hd_encoder_clip(encoder_state);

    int best_score = INT_MAX;
    class_t best_class;
    for (class_t class = 0; class < state->n_class; class++)
    {
        // TODO: move encoder output to a separate output buffer
        // or to ngramm_buffer when clip packing is implemented
        int score = hamming_distance(
            encoder_state->ngramm_sum_buffer,
            // TODO: remove " * sizeof(block_t) * 8" when packing is implemented
            state->am + class * state->n_blk * sizeof(block_t) * 8,
            // TODO: remove " * sizeof(block_t) * 8" when packing is implemented
            state->n_blk * sizeof(block_t) * 8 * sizeof(state->am[0])
        );

        if (score < best_score)
        {
            best_score = score;
            best_class = class;
        }
    }

    return best_class;
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
