void circshift(int32_t * const dest, int32_t * const src, const int sz, const int n);

void circshift_inplace(int32_t * const arr, const int sz, const int n);

void ngrammencoding(
    int32_t * const output,
    const int d,
    const int ngramm,
    int32_t * const block,
    int32_t * item
);
