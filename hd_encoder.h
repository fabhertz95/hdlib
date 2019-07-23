struct hd_encoder_t
{
    // HD vector length
    int d;

    // n-gramm length
    int ngramm;

    // encoded n-gramm buffer
    // shape: [d]
    int32_t * ngramm_buffer;

    // encoded n-gramm summation buffer
    // shape: [d]
    int32_t * ngramm_sum_buffer;
    int ngramm_sum_count;

    // HD vector n-gramm circular buffer (head: last copied item)
    // shape: [ngramm, d]
    int32_t * item_buffer;
    int item_buffer_head;

    // HD vector lookup table
    // shape: [n_items, d]
    int32_t * item_lookup;
};

void hd_encoder_init(
    struct hd_encoder_t * const p_state,
    const int d,
    const int ngramm,
    const int n_items
);

void hd_encoder_encode_ngramm(
    struct hd_encoder_t * const p_state,
    int32_t * item
);

void hd_encoder_encode (
    struct hd_encoder_t * const p_state,
    int32_t * const data,
    const int n_data
);

void hd_encoder_clip(
    struct hd_encoder_t * const p_state
);
