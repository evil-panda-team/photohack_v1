import scipy.stats as st
import dlib
import cv2
from imutils import face_utils
import numpy as np


def get_gradient(size_g, color=[0, 255, 255], sigma=1, mu=0):
    x, y = np.meshgrid(np.linspace(-1, 1, size_g), np.linspace(-1, 1, size_g))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    final = np.zeros((512, 512, 3))
    for i in range(3):
        final[:, :, i] = color[i]
        final[:, :, i] = final[:, :, i] * g
    final = final.astype(np.uint8)
    return final


def transparent_overlay(src, overlay, text=['Hello!'], font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0), pos_txt=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    scr_cp = src.copy()
    numsteps = 3
    # loop over all pixels and apply the blending equation
    for al in range(numsteps):
        scr_cp = src.copy()
        for i in range(h):
            for j in range(w):
                if x + i >= rows or y + j >= cols:
                    continue
                # read the alpha channel
                alpha = (1/numsteps)*al*float(overlay[i][j][3] / 255.0)
                scr_cp[x + i][y + j] = alpha * overlay[i][j][:3] + \
                    (1 - alpha) * scr_cp[x + i][y + j]
        cv2.imshow("Frame", scr_cp)
        cv2.waitKey(1)

    coeff = len(text)
    dy = int(60 * 1/coeff)
    y_pos = pos_txt[1]
    curr_img = scr_cp.copy()
    for l in text:
        curr_line = ''
        image_copy = np.zeros_like(img)
        for w in l:
            curr_line += w
            image_copy = curr_img.copy()
            cv2.putText(image_copy, curr_line, (pos_txt[0], y_pos), font,
                        1.5*1/coeff, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("Frame", image_copy)
            cv2.waitKey(50)
        curr_img = image_copy
        y_pos += dy

    return curr_img


def check_text(text, max_line_length=33, max_length=100):
    lines = []
    if len(text) > max_length:
        text = text[:100]
    splitted_text = text.split()
    current_line = ''
    for word in splitted_text:
        if len(current_line+word) <= max_line_length:
            current_line = current_line + ' ' + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img = cv2.imread('data/images/selfies/rafa.jpg', -1)
cloud = cv2.imread('data/images/clouds/cloud4.png', -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)
shape = predictor(gray, rects[0])
shape = face_utils.shape_to_np(shape)

# loop over the (x, y)-coordinates for the facial landmarks
# and draw them on the image
# for (x, y) in shape:
#     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
mouth_x, mouth_y = shape[54]

# cv2.circle(img, (mouth_x, mouth_y), 2, (255, 0, 0), -1)
cloud_w = asd.shape[0] - mouth_x
cloud_ratio = cloud.shape[0]/cloud.shape[1]
cloud_h = int(cloud_ratio*cloud_w)
cloud_resized = cv2.resize(cloud, (cloud_w, cloud_h))
cloud_pos_x = mouth_x
cloud_pos_y = mouth_y - int(cloud_h*(5/4))
cloud_center_x = cloud_pos_x + int(cloud_w/5)
lines = check_text(message, max_line_length=cloud_w//20)
cloud_center_y = cloud_pos_y + int(cloud_h/2)
text = 'Hello! How are you? I am fine, thank you! And what about you? Do you like chocolate? Yes, sure! Wow fefeff'
# text = 'Hello! '
# text = 'Hello!'
lines = check_text(text, max_line_length=25)
result = transparent_overlay(
    img, cloud_resized, text=lines, pos=(cloud_pos_x, cloud_pos_y), pos_txt=(cloud_center_x, cloud_center_y), font=cv2.FONT_HERSHEY_COMPLEX_SMALL)
cv2.imshow("Frame", result)
cv2.waitKey(1)

# img = get_gradient(512, color=[180, 70, 133], sigma=1, mu=0)
# cv2.imshow("Frame", img)
# cv2.waitKey(0)
