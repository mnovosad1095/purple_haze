from sklearn.neighbors import KDTree
import numpy as np
import numpy_groupies as npg
import cv2 
import matplotlib.pyplot as plt
from wls_opt import wls_optimization

def show_image_in_window(img):
  cv2.imshow("I",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def shift_to_airlight(image, A):
  shifted = image - A
  return shifted

def convert_to_spherical(image):
  out = np.zeros(image.shape)
  Bv = image[:,:,0]
  Gv = image[:,:,1]
  Rv = image[:,:,2]
  out[:,:,0] = np.sqrt(np.square(Rv)+np.square(Bv)+np.square(Gv))
  out[:,:,0] = out[:,:,0] / np.max(out[:,:,0])

  out[:,:,1] = np.arctan2(Bv,Rv)
  out[:,:,1] = out[:,:,1] / np.abs(np.max(out[:,:,1]))

  out[:,:,2] = np.arctan2(np.sqrt(np.square(Rv)+np.square(Bv)),Gv)
  out[:,:,2] = out[:,:,2] / np.abs(np.max(out[:,:,2]))
  
  return out

def create_tree_from_tessalation(filename):
  points = np.loadtxt(filename)
  return KDTree(points)


if __name__ == "__main__":
  image = cv2.imread("../images/pumpkins_input.png") 
  image_norm = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) 
  h,w,ncols = image.shape

  A = [0.72, 0.85, 0.81]
  shifted = shift_to_airlight(image_norm, A)
  
  spherical = convert_to_spherical(shifted)
  
  tree = create_tree_from_tessalation("TR1000.txt")
  radius = np.sqrt(shifted[:,:,0]**2 + shifted[:,:,1]**2 +shifted[:,:,2]**2 )

  _, idx = tree.query(spherical.reshape((h*w, ncols)))

  radius_max = npg.aggregate(idx.flatten(), spherical[:,:,0].flatten(), func="max")
  radius_max = np.reshape(radius_max[idx], (h,w))

  transmission_estimate = np.divide(spherical[:,:,0], radius_max)

  transmission_estimate = np.minimum(np.maximum(transmission_estimate, 0.1), 1.0)
  
  tlb = 1 - np.min(np.divide(image_norm.reshape((h*w,ncols)), A), axis=1)
  tlb = np.maximum(transmission_estimate.flatten(), tlb)
  tlb = tlb.reshape((h,w))

  radius_std = npg.aggregate(idx.flatten(), spherical[:,:,0].flatten(), func='std')
  radius_std = np.reshape(radius_std[idx], (h,w))

  t = wls_optimization(transmission_estimate, radius_std, image, 0.1)

  img_dehazed = np.zeros((h,w,ncols))
  leave_haze = 1.00
  for i in range(3):
    img_dehazed[:,:,i] = (image_norm[:,:,i] - (1-t*leave_haze)*A[i])/np.maximum(t,0.1)
  a = np.concatenate([image_norm, img_dehazed], axis=1)
  show_image_in_window(a)