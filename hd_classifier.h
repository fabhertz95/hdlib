struct hd_classifier_t
{
};

void hd_classifier_init(
    struct hd_classifier_t * const state
);

void hamming_distance_init();

int hamming_distance(
    const void * const a,
    const void * const b,
    const int n
);
