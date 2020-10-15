# wls optimizer
import cv2
import numpy as np
from scipy.sparse import spdiags, linalg


def wls_optimization(inp, data_weight, guidance, l):
  err = 0.00001
  h,w,_ = guidance.shape
  k = h*w

  guidance =  cv2.cvtColor(guidance, cv2.COLOR_BGR2GRAY)

  dy = np.diff(guidance, 1, 0)
  dy = -l/(np.abs(dy)**2 + err)
  dy = np.pad(dy, ((0,1), (0,0)), 'constant')
  dy = dy.flatten()

  dx = np.diff(guidance, 1, 1)
  dx = -l/(np.abs(dx)**2 + err)
  dx = np.pad(dx, ((0,0), (0,1)), 'constant')
  dx = dx.flatten()

  B = [dx, dy]
  d = [-h, -1]
  tmp = spdiags(B, d, k, k)

  ea = dx
  we = np.pad([dx], ((h,0), (0,0)), 'constant')
  we = we[:-h]
  
  so = dy
  no = np.pad([dy], ((1,0), (0,0)), 'constant')
  no = no[:-1]

  D = -(ea+we+so+no)
  Asmooth = tmp + tmp.T.conj() + spdiags(D, 0, k, k)

  data_weight = data_weight - np.min(data_weight.flatten())
  data_weight = np.divide(data_weight, np.max(data_weight.flatten()) + err)

  rel_mask = (data_weight[0,:] < 0.6)
  in_row1 = np.min(inp, axis=0)
  data_weight = np.where(data_weight[0,:] < 0.6, 0.8, data_weight)
  inp[0, rel_mask] = in_row1[rel_mask]

  Adata = spdiags(data_weight.flatten(), 0, k, k)

  A = Adata + Asmooth
  b = Adata.dot(inp.flatten())
  out = linalg.spsolve(A,b)

  return np.reshape(out, (h,w))
