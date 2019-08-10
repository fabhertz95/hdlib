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
