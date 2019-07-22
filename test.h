void circshift(int32_t * const dest, int32_t * const src, const int sz, const int n);

void circshift_inplace(int32_t * const arr, const int sz, const int n);

void ngrammencoding(
    int32_t * const output,
    const int d,
    const int ngramm,
    int32_t * const block,
    int32_t * item
);

void ngrammencoding_string (
    int32_t * const output,
    const int d,
    const int ngramm,
    const int n_feat,
    int32_t * const data,
    int32_t * const block,
    int32_t * const itemMemory,
    int32_t * const tmp_ngramm
);
