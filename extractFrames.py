import numpy as np
import cv2

# In[]
#filename = "data/videos/archers09e05.mkv"
filename = "data/videos/vicecity.mp4"

vidcap = cv2.VideoCapture(filename)

success, image = vidcap.read()
count = 0

skip = 250
i = 0

while success:
    success, image = vidcap.read()
    if i%skip == 0:
        image = image[200:650,340:940,:]
        cv2.imwrite("data/images/styles/vicecity/frame%d.jpg" % count, image)     # save frame as JPEG file      
        print('Read a new frame: ', success)
        count += 1
    i += 1
  
# In[]
  
# In[]
  
# In[]
  
# In[]
  
  