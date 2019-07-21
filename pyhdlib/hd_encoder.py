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

class sng_encoder_ext(hd_encoder):
	'''
	Sum n-gramm encoder, external library binding
	'''
	
	def __init__(self, D, nitem, ngramm):
		self._D = D
		self._ngramm = ngramm

		# malloc for Ngramm block, ngramm result, and sum vector  
		self._block = t.Tensor(self._ngramm,self._D).type(t.int32)
		self._Y = t.Tensor(self._D).type(t.int32)
		self._SumVec= t.Tensor(self._D).type(t.int32)

		self._add_cnt = 0

		# item memory initialization 
		self._itemMemory = t.randint(0,2,(nitem,D), dtype=t.int32)

		import cffi
		import os
		import platform
		
		self._ffi = cffi.FFI()
		path = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
		#path = '.'
		self._ffi.cdef(open(os.path.join(path, 'test.h'), 'r').read())

		self._lib = self._ffi.dlopen(
			os.path.join(path, f'test_{platform.machine()}.so')
		)
		# TODO: close self._lib


	def encode(self, X):
		# reset block to zero
		self._block.zero_()
		self._SumVec.zero_()

		n_samlpes, n_feat = X.shape
		self._add_cnt = 0
		
		for feat_idx in range(n_feat): 
			ngramm = self._ngrammencoding(X[0],feat_idx)
			if feat_idx >= self._ngramm-1:
				self._SumVec.add_(ngramm)
				self._add_cnt +=1

		return self._SumVec.type(t.float), self._add_cnt

	def _ngrammencoding(self, X, start):
		output = t.Tensor(self._D).type(t.int32)
		
		item = self._itemMemory[X[start]]

		self._lib.ngrammencoding(
			self._ffi.cast('int32_t * const', output.data_ptr()),
			self._D,
			self._ngramm,
			self._ffi.cast('int32_t * const', self._block.data_ptr()),
			self._ffi.cast('int32_t *', item.data_ptr())
		)

		return output

	def _circshift(self, dest, src, n):
		self._lib.circshift(
			self._ffi.cast('int32_t * const', dest.data_ptr()),
			self._ffi.cast('int32_t * const', src.data_ptr()),
			self._D,
			n
		)

	def _circshift_inplace(self, arr, n):
		self._lib.circshift_inplace(
			self._ffi.cast('int32_t * const', arr.data_ptr()),
			self._D,
			n
		)

	def clip(self):
		'''
		Clip sum of ngramms to 1-bit values
		'''

		# Add random vector to break ties
		if self._add_cnt % 2 == 0:
			self._SumVec.add_(t.randint(0,2,(self._D,), dtype=t.int32))
			self._add_cnt += 1
		
		self._SumVec = (self._SumVec > (self._add_cnt/2)).type(t.int32)
		self._add_cnt = 1

		return self._SumVec.type(t.float), self._add_cnt


class sng_encoder(hd_encoder):
	'''
	Sum n-gramm encoder
	'''

	def __init__(self, D, device, nitem=1, ngramm = 3):
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
		self._block = t.Tensor(self._ngramm,self._D).zero_().to(self._device)
		self._Y = t.Tensor(self._D).to(self._device)
		self._SumVec= t.Tensor(self._D).zero_().to(self._device)

		self._add_cnt = 0

		# item memory initialization 
		self._itemMemory = t.randint(0,2,(nitem,D)).to(self._device)


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


		n_samlpes,n_feat = X.shape
		self._add_cnt = 0
		
		for feat_idx in range(n_feat): 
			ngramm = self._ngrammencoding(X[0],feat_idx)
			if feat_idx >= self._ngramm-1:
				self._SumVec.add_(ngramm)
				self._add_cnt +=1

		return self._SumVec, self._add_cnt

	def clip(self):
		'''
		clip sum of ngramms to 1-bit values
		'''

		self._SumVec = self._threshold(self._SumVec, self._add_cnt)
		self._add_cnt = 1
			
		return self._SumVec, self._add_cnt

	def _ngrammencoding(self,X,start):
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
		for i in range(self._ngramm-1,0,-1): 
			self._block[i] = self._circshift(self._block[i-1],1)
		# write new first entry 
		self._block[0] = self._itemMemory[X[start]]

		# calculate ngramm of _block
		self._Y = self._block[0]
		
		for i in range(1,self._ngramm):
			self._Y = self._bind(self._Y,self._block[i])

		return self._Y


	def _circshift(self,X,n):
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

	def _bind(self,X1,X2): 
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
		return ((t.mul((-2*X1+1), (2*X2-1))+1)/2)

	def _threshold(self,X,cnt):
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
			X.add_(t.randint(0,2,(self._D,)).type(t.FloatTensor).to(self._device)) # add random vector 
			cnt += 1
		
		return (X > (cnt/2)).type(t.FloatTensor)

