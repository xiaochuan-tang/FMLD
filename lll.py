import os
import cv2
path = '/home/aiml/MMLD/dataset/dataset_MMLD/test_hill_jpeg/2.jpeg'
im = cv2.imread(path,cv2.IMREAD_LOAD_GDAL)
print(im)