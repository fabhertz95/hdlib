typedef uint8_t class_t;

struct hd_classifier_t
{
    // HD vector length
    int n_blk;

    // number of classes supported by hd_classifier
    int n_class;

    // trained class vectors
    // shape: [n_class, d]
    // TODO shape: [n_class, n_blk] and block_t
    uint32_t * am;
};

void hd_classifier_init(
    struct hd_classifier_t * const state,
    const int n_blk,
    const int n_class
);

class_t hd_classifier_predict(
    const struct hd_classifier_t * const state,
    struct hd_encoder_t * const encoder_state,
    const feature_t * const x,
    const int n_x
);

void hamming_distance_init();

int hamming_distance(
    const void * const a,
    const void * const b,
    const int n
);
