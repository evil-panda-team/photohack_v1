# -*- coding: utf-8 -*-
import numpy as np
import cv2 

# Find the largest contour and extract it
def extract_largest_blob(mask, area_threshold=1, num_blobs=1):

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=4)
    st = stats[:,-1][stats[:,-1] > area_threshold]
                    
    if nb_components == 1 or len(st) < 2:
        return None, None, None
    
    if nb_components == 2 or len(st) == 2:
        return mask
    
    #blob_index = np.argsort(stats[:,-1])[-(num_blobs+1):-1]
    
    return output == 1

# Filling holes
def fill_holes(mask):

    mask_floodfill = mask.astype('uint8').copy()
    h, w = mask.shape[:2]
    cv2.floodFill(mask_floodfill, np.zeros((h+2, w+2), np.uint8), (0,0), 255)

    out = mask | cv2.bitwise_not(mask_floodfill)
    
    return out.astype(np.bool)
