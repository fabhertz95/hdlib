# Hyperdimensional Computing Library

### Prerequisites

- python3.6
- numpy
- pytorch4.0

The packages can be installed easily with conda and the _config.yml file: 
```
$ conda env create -f hdlib-env.yml -n hdlib-env
$ source activate hdlib-env 
```


## Author

* **Michael Hersche** - *Initial work* - [MHersche](https://github.com/MHersche)
* **Sebastian Kurella** - [skurella](https://github.com/skurella)
* **Tibor Schneider** - [tiborschneider](https://github.com/tiborschneider)

## Optimizations

- [x] CPU (bit packing, cicular buffers, hamming distance LUT)
- [x] GPU with global memory
- [x] GPU with shared memory
- [x] GPU with thread-local memory
- [x] GPU with thread-local memory and batches (does not work well)
- [ ] better batching
- [ ] memory coalescing
- [ ] memory bank alignment
- [ ] clipping and inference to GPU

## Measurements

- [x] CPU
- [ ] GPU with global memory
- [x] GPU with shared memory
- [x] GPU with thread-local memory
- [x] GPU with thread-local memory and batches (does not work well)

## Blocks and Threads

### No input-data-parallelism

```
| D pack | characters in input sample --> | Block |
|--------|--------------------------------|-------|
| 0      |                                |       |
| :      |                                |       |
| :      | 128 threads                    | 0     |
| :      |                                |       |
| 127    |                                |       |
|--------|--------------------------------|-------|
| 128    |                                |       |
| :      |                                |       |
| :      | 128 threads                    | 1     |
| :      |                                |       |
| 255    |                                |       |
|--------|--------------------------------|-------|
| 256    |                                |       |
| :      |                                |       |
| :      | 128 threads                    | 2     |
| :      |                                |       |
| 312    |                                |       |
|--------|--------------------------------|-------|
```

### input-data-parallelism

```
| D pack | characters in input sample --> | Block |
|--------|--------------------------------|-------| 
| 0      |                |               |       |
| :      | 64 threads     | 64 threads    | 0     |
| 63     |                |               |       |
|--------|--------------------------------|-------|
| 64     |                |               |       |
| :      | 64 threads     | 64 threads    | 1     |
| 127    |                |               |       |
|--------|--------------------------------|-------|
| 128    |                |               |       |
| :      | 64 threads     | 64 threads    | 2     |
| 191    |                |               |       |
|--------|--------------------------------|-------|
| 192    |                |               |       |
| :      | 64 threads     | 64 threads    | 3     |
| 255    |                |               |       |
|--------|--------------------------------|-------|
| 256    |                |               |       |
| :      | 64 threads     | 64 threads    | 4     |
| 319    |                |               |       |
|--------|--------------------------------|-------|
```

- input-dimension division `m`.
- HD dimension D (packed) for each thread block: `128 / m`
- Shared memory size (per block):
  - `item_lookup`: `(n_items * 128 / m) * 4 [bytes]`
  - `ngramm_sum_buffer`: `((32 * 128 / m) * m) * 4 [bytes] = (32 * 128) * 4 [bytes] `
  - Total: (with `n_items = 29`):
    - `sizeof(item_lookup) = 14336 / m [bytes]`
    - `sizeof(ngramm_sum_buffer) = 16384 [bytes]`
    - `sizeof(shared_memory) <= 30720 [bytes] < 65536 [bytes]`
