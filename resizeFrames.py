import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# In[]
dr = "datasets/scene2archer/trainB/c/"

files = os.listdir(dr)

for i in tqdm(range(len(files))):
    img = cv2.imread(dr+files[i])
    print(img.shape)
    img2 = cv2.resize(img, (512, 384))
    cv2.imwrite("kek{}.jpg".format(i), img2)
  
# In[]
  
# In[]
  
# In[]
  
# In[]
  
  