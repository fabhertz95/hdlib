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
    // am and am_count are not initialised, they are set externally
    state->am_clipped = malloc(n_class * n_blk * sizeof(block_t));
}

void hd_classifier_threshold(
    const struct hd_classifier_t * const state
)
{
    for (class_t class = 0; class < state->n_class; class++)
    {
        hd_encoder_clip(
            state->am + class * state->n_blk * sizeof(block_t) * 8,
            state->n_blk * sizeof(block_t) * 8,
            state->am_count[class],
            state->am_clipped + class * state->n_blk
        );
    }
}

class_t hd_classifier_predict(
    const struct hd_classifier_t * const state,
    struct hd_encoder_t * const encoder_state,
    const feature_t * const x,
    const int n_x
)
{
    hd_encoder_encode(encoder_state, x, n_x);
    // TODO: move rename hd_encoder_clip to clip
    // and implement this call as hd_encoder_clip
    hd_encoder_clip(
        encoder_state->ngramm_sum_buffer,
        sizeof(block_t) * 8 * encoder_state->n_blk,
        encoder_state->ngramm_sum_count,
        encoder_state->ngramm_buffer
    );

    int best_score = INT_MAX;
    class_t best_class;
    for (class_t class = 0; class < state->n_class; class++)
    {
        int score = hamming_distance(
            encoder_state->ngramm_buffer,
            state->am_clipped + class * state->n_blk,
            state->n_blk * sizeof(state->am_clipped[0])
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
