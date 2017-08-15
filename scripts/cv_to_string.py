import cv2
import numpy as np
import tf.transformations as tft

r = np.array([ 0.06755048,  0.01013327,  0.00100022])
t = np.array([ 0.00139976,  0.03109518, -0.0147689 ])

def cv_to_matrix(r, t):
  rot, _ = cv2.Rodrigues(r)
  matrix = np.eye(4, 4)
  matrix[:3, :3] = rot
  matrix[:3, 3] = t
  return matrix

def matrix_to_stf(matrix):
  s = ' '.join([str(x) for x in matrix[:3, 3]])
  q = tft.quaternion_from_matrix(matrix)
  s += ' ' + ' '.join([str(x) for x in q])
  return s
