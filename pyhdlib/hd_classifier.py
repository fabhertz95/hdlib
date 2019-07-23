#!/usr/bin/env python3

'''
==============================================================================
Associative Memory (AM) classifier for binary Hyperdimensional (HD) Comuputing 
==============================================================================
'''
import time
import sys
import torch as t
import numpy as np
import cloudpickle as cpckl


from hd_encoder import sng_encoder_bv
from am_classifier import am_classifier


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "17.5.2019"


class hd_classifier(am_classifier):

    def __init__(self, D=10000, encoding='sumNgramm', device='cpu', nitem=1, ngramm=3, name='test'):
        '''

        Parameters
        ----------
        D : int
                HD dimension
        encode: hd_encoding class
                encoding class
        '''

        # make sure that D is a multiple of 32
        if D % 32 != 0:
            D = (int(D / 32) + 1) * 32
            print(f"Dimensionality given which is not a multiple of 32! Using {D} instead")

        self._name = name
        try:
            self.load()
        except:

            use_cuda = t.cuda.is_available()
            _device = t.device(device if use_cuda else "cpu")

            if encoding is 'sumNgramm':
                _encoder = sng_encoder_bv(D, nitem, ngramm)
            else:
                raise ValueError(f'{encoding} encoding not supported')

            super().__init__(D, _encoder, _device)

    def save(self):
        '''
        save class as self.name.txt
        '''
        file = open(self._name + '.txt', 'wb')
        cpckl.dump(self.__dict__, file)
        file.close()

    def load(self):
        '''
        try load self._name.txt
        '''
        file = open(self._name + '.txt', 'rb')

        self.__dict__ = cpckl.load(file)

    def save2binary_model(self):
        '''
        try load self._name_bin.npz
        '''

        _am = bin2int(self._am.cpu().type(t.LongTensor).numpy())
        _itemMemory = bin2int(
            self._encoder._itemMemory.cpu().type(t.LongTensor).numpy())

        np.save(self._name + 'bin', _n_classes=self._n_classes, _am=_am,
                _itemMemory=_itemMemory, _encoding=self._encoding)

        return


def bin2int(x):
    '''
    try load self._name_bin.npz

    Parameters
    ----------
    x : numpy array size = [u,v]
            input array binary
    Restults
    --------
    y : numpy array uint32 size = [u, ceil(v/32)]
    '''

    u, v = x.shape

    v_out = int(np.ceil(v / 32))
    y = np.zeros((u, v_out), dtype=np.uint32)

    for uidx in range(u):
        for vidx in range(v_out):
            for bidx in range(32):  # iterate through all bit index
                if vidx * 32 + bidx < v:
                    y[uidx, vidx] += x[uidx, vidx * 32 + bidx] << bidx

    return y
