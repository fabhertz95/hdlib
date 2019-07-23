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
    int32_t * encoder_buffer;
    int encoder_count;

    // HD vector n-gramm buffer
    // shape: [ngramm, d]
    int32_t * item_buffer;

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

void ngrammencoding(
    const struct hd_encoder_t * const p_state,
    int32_t * item
);

void ngrammencoding_string (
    struct hd_encoder_t * const p_state,
    int32_t * const data,
    const int n_data
);

void hd_encoder_clip(
    struct hd_encoder_t * const p_state
);
