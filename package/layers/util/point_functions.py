import torch
import torch.nn as nn

import _mapped_convolution_ext._knn as _knn


def knn(ref, query, k):
    '''
	ref: B X D X M 		Set of reference points to check against
	query: B X D X N 	Set of query points for which we find the near
	k: scalar			Number of matches to find

	@returns
	idx: B x K x N 		List of indices of the reference point set,
						ordered from closest to furthers in dimension 1. Indices range from [0, M]
	dist: B x K x N 	Distance associated with the corresponding indices 						element
	'''
    return _knn.knn_forward(ref, query, k)