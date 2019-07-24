#!/usr/bin/env python3

'''
=================
HD encoding class
=================
'''
import torch as t
from abc import ABC, abstractmethod

__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "17.5.2019"


class hd_encoder(ABC):
    @abstractmethod
    def encode(X):
        pass

    @abstractmethod
    def clip(X):
        pass


class sng_encoder_bv(hd_encoder):
    '''
    Sum n-gramm encoder, external library binding for encoding completely in C, using bitvectors
    '''
    def __init__(self, D, nitem, ngramm):
        # round D down to nearest multiple of 32
        self._n_blk = int(D / 32)
        self._D = self._n_blk * 32
        self._ngramm = ngramm

        # malloc for Ngramm block, ngramm result, and sum vector
        self._block = t.Tensor(self._ngramm, self._n_blk).type(t.int32)
        self._Y = t.Tensor(self._n_blk).type(t.int32)
        self._SumVec = t.Tensor(self._D).type(t.int32)

        self._add_cnt = 0

        # item memory initialization
        self._itemMemory = t.randint(0, 1 << 32, (nitem, D), dtype=t.int32)

        import cffi
        import os
        import platform

        self._ffi = cffi.FFI()
        path = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
        # path = '.'
        self._ffi.cdef(open(os.path.join(path, 'hd_encoder_bv.h'), 'r').read())

        self._lib = self._ffi.dlopen(
            os.path.join(path, f'hd_encoder_bv_{platform.machine()}.so')
        )
        # TODO: close self._lib

    def encode(self, X):
        # reset block to zero
        self._block.zero_()
        self._SumVec.zero_()

        # compute dimensionality
        n_samples, n_feat = X.shape

        # prepare data
        tmp_ngramm = t.zeros(self._n_blk).type(t.int32)

        self._lib.ngrammencoding_string_bv(
            self._ffi.cast('uint32_t * const', self._SumVec.data_ptr()),
            self._n_blk,
            self._ngramm,
            n_feat,
            self._ffi.cast('uint32_t * const', X.data_ptr()),
            self._ffi.cast('uint32_t * const', self._block.data_ptr()),
            self._ffi.cast('uint32_t * const', self._itemMemory.data_ptr()),
            self._ffi.cast('uint32_t * const', tmp_ngramm.data_ptr()),
        )

        self._add_cnt = n_feat - (self._ngramm - 1)
        return self._SumVec.type(t.float), self._add_cnt

    def clip(self):
        '''
        Clip sum of ngramms to 1-bit values
        '''

        # Add random vector to break ties
        if self._add_cnt % 2 == 0:
            self._SumVec.add_(t.randint(0, 2, (self._D,), dtype=t.int32))
            self._add_cnt += 1

        self._SumVec = (self._SumVec > (self._add_cnt / 2)).type(t.int32)
        self._add_cnt = 1

        return self._SumVec.type(t.float), self._add_cnt


class sng_encoder_ext(hd_encoder):
    '''
    Sum n-gramm encoder, external library binding
    '''

    def __init__(self, D, nitem, ngramm):
        import cffi
        import os
        import platform

        n_blk = int(D / 32)
        D = n_blk * 32

        path = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))

        self._ffi = cffi.FFI()
        self._ffi.cdef(open(os.path.join(path, 'hd_encoder.h'), 'r').read())
        self._lib = self._ffi.dlopen(
            os.path.join(path, f'hd_encoder_{platform.machine()}.so')
        )

        self._data = self._ffi.new('struct hd_encoder_t *')
        self._lib.hd_encoder_init(self._data, n_blk, ngramm, nitem)

        # overwrite the encoder summing buffer with a torch tensor's pointer
        # this is so that the result can be communicated without copying
        # TODO this will likely be unnecesary in the future
        self._ngramm_sum_buffer = t.Tensor(D).type(t.int32).contiguous()
        self._data.ngramm_sum_buffer = self._ffi.cast('uint32_t * const', self._ngramm_sum_buffer.data_ptr())

        # TODO: release memory and close self._lib

    def encode(self, X):
        self._data.ngramm_sum_buffer = self._ffi.cast('uint32_t * const', self._ngramm_sum_buffer.data_ptr())
        # TODO something breaks without the previous line

        # compute dimensionality
        n_samples, n_feat = X.shape

        self._lib.hd_encoder_encode(
            self._data,
            self._ffi.cast('uint32_t * const', X.data_ptr()),
            n_feat
        )

        return self._ngramm_sum_buffer.type(t.float), self._data.ngramm_sum_count

    def clip(self):
        self._lib.hd_encoder_clip(
            self._data
        )
        
        return self._ngramm_sum_buffer.type(t.float), self._data.ngramm_sum_count


class sng_encoder(hd_encoder):
    '''
    Sum n-gramm encoder
    '''

    def __init__(self, D, device, nitem=1, ngramm=3):
        '''
        Encoding

        Parameters
        ----------
        nitem: int
                number of items in itemmemory
        ngramm: int
                number of ngramms
        '''

        self._D = D
        self._device = device
        self._ngramm = ngramm

        # malloc for Ngramm block, ngramm result, and sum vector
        self._block = t.Tensor(self._ngramm, self._D).zero_().to(self._device)
        self._Y = t.Tensor(self._D).to(self._device)
        self._SumVec = t.Tensor(self._D).zero_().to(self._device)

        self._add_cnt = 0

        # item memory initialization
        self._itemMemory = t.randint(0, 2, (nitem, D)).to(self._device)

        return

    def encode(self, X):
        '''
        compute sum of ngramms

        Parameters
        ----------
        X: torch tensor, size = [n_samples,n_feat]
                feature vectors

        Return
        ------
        SumVec: torch tensor, size = [D,]
                sum of encoded n-gramms
        add_cnd: int
                number of encoded n-gramms
        '''

        # reset block to zero
        self._block.zero_().to(self._device)
        self._SumVec.zero_()

        n_samlpes, n_feat = X.shape
        self._add_cnt = 0

        for feat_idx in range(n_feat):
            ngramm = self._ngrammencoding(X[0], feat_idx)
            if feat_idx >= self._ngramm - 1:
                self._SumVec.add_(ngramm)
                self._add_cnt += 1

        return self._SumVec, self._add_cnt

    def clip(self):
        '''
        clip sum of ngramms to 1-bit values
        '''

        self._SumVec = self._threshold(self._SumVec, self._add_cnt)
        self._add_cnt = 1

        return self._SumVec, self._add_cnt

    def _ngrammencoding(self, X, start):
        '''
        Load next ngramm

        Parameters
        ----------
        X: Torch tensor, size = [n_samples, D]
                Training samples

        Results
        -------
        Y: Torch tensor, size = [D,]
        '''

        # rotate shift current block
        for i in range(self._ngramm - 1, 0, -1):
            self._block[i] = self._circshift(self._block[i - 1], 1)
        # write new first entry
        self._block[0] = self._itemMemory[X[start]]

        # calculate ngramm of _block
        self._Y = self._block[0]

        for i in range(1, self._ngramm):
            self._Y = self._bind(self._Y, self._block[i])

        return self._Y

    def _circshift(self, X, n):
        '''
        Load next ngramm

        Parameters
        ----------
        X: Torch tensor, size = [D,]


        Results
        -------
        Y: Torch tensor, size = [n_samples-n]
        '''
        return t.cat((X[-n:], X[:-n]))

    def _bind(self, X1, X2):
        '''
        Bind two vectors with XOR

        Parameters
        ----------
        X1: Torch tensor, size = [D,]
                input vector 1
        X2: Torch tensor, size = [D,]
                input vector 2

        Results
        -------
        Y: Torch tensor, size = [D,]
                bound vector
        '''
        # X1!= X2
        return ((t.mul((-2 * X1 + 1), (2 * X2 - 1)) + 1) / 2)

    def _threshold(self, X, cnt):
        '''
        Threshold a vector to binary

        Parameters
        ----------
        X : Torch tensor, size = [D,]
                input vector to be thresholded
        cnt: int
                number of added binary vectors, used for determininig threshold

        Results
        -------
        Y: Torch tensor, size = [D,]
                thresholded vector
        '''
        # even
        if cnt % 2 == 0:
            X.add_(t.randint(0, 2, (self._D,)).type(
                t.FloatTensor).to(self._device))  # add random vector
            cnt += 1

        return (X > (cnt / 2)).type(t.FloatTensor)
