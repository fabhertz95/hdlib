// number of threads per block in the grid
#define NUM_THREADS_IN_BLOCK 128

// input vector feature
typedef uint8_t feature_t;

// packed HD vector block
typedef uint32_t block_t;

// struct to store pointers on the device, which remain
struct device_data_t
{
    // shape: [ngramm, n_blk]
    block_t * ngramm_buffer;
    // shape: [n_blk * 32]
    uint32_t * ngramm_sum_buffer;
    // shape: [n_items, n_blk]
    block_t * item_lookup;
};

struct hd_encoder_t
{
    // HD vector length
    int n_blk;

    // n-gramm length
    int ngramm;

    // encoded n-gramm buffer
    // shape: [n_blk]
    block_t * ngramm_buffer;

    // encoded n-gramm summation buffer
    // shape: [n_blk * 32]
    uint32_t * ngramm_sum_buffer;
    int ngramm_sum_count;

    // HD vector n-gramm circular buffer (head: last copied item)
    // shape: [ngramm, n_blk]
    block_t * item_buffer;
    int item_buffer_head;

    // HD vector lookup table
    // shape: [n_items, n_blk]
    block_t * item_lookup;
    int n_items;

    // device pointers
    struct device_data_t device;
};

void hd_encoder_init(
    struct hd_encoder_t * const state,
    const int n_blk,
    const int ngramm,
    const int n_items
);

void hd_encoder_setup_device(
    struct hd_encoder_t * const state
);

void hd_encoder_free(
    struct hd_encoder_t * const state
);

void hd_encoder_encode_ngramm(
    struct hd_encoder_t * const state,
    block_t * item
);

// encodes an array of features x (shape: [n_x])
// and puts the result in state->ngramm_sum_buffer
void hd_encoder_encode (
    struct hd_encoder_t * const state,
    const feature_t * const x,
    const int n_x
);

void hd_encoder_clip(
    struct hd_encoder_t * const state
);

// thresholds the in buffer at half count,
// and packs the output into the out buffer
void clip(
    const uint32_t * const in,
    const int n_in,
    const int count,
    block_t * const out
);
