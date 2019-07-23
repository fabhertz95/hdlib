void circshift_bv_naive(uint32_t * const dest, uint32_t * const src, const int n_blk, const int n);

void circshift_inplace_bv_naive(uint32_t * const arr, const int n_blk, const int n);

void circshift_bv_word(uint32_t * const dest, uint32_t * const src, const int n_blk, const int n);

void circshift_inplace_bv_word(uint32_t * const arr, const int n_blk, const int n);

void circshift_bv_bit(uint32_t * const dest, uint32_t * const src, const int n_blk, const int n);

void circshift_inplace_bv_bit(uint32_t * const arr, const int n_blk, const int n);

void ngrammencoding_bv(
    uint32_t * const output,
    const int n_blk,
    const int ngramm,
    uint32_t * const block,
    uint32_t * item
);

void ngrammencoding_string_bv (
    uint32_t * const sumVec,
    const int d,
    const int ngramm,
    const int n_feat,
    uint32_t * const data,
    uint32_t * const block,
    uint32_t * const itemMemory,
    uint32_t * const tmp_ngramm
);
