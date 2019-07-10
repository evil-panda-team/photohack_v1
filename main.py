# -*- coding: utf-8 -*-

# In[] Import necessary packages
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# In[]: INPUTS
#image = "selfie_002.png"
image = "pavel.jpg"
image = "rafa.jpg"

message = 'Hello! How are you? I am fine, thank you! And what about you? Do you like chocolate? Yes, sure! Wow fefeff'
message = 'Rauf is bald to you! Go to our club'
message = "Will we buy Go in such a way that there is still more and then buy more and so on until we die?"
message = "Здравствуйте всем, привет всем. Передаю привет балдёжной команде ивел панде."

# In[] LOAD IMAGE
selfie_path = "data/images/selfies/" + image
img = Image.open(selfie_path)
h,w = (384,512)
img = img.resize((h,w))
plt.imshow(img)

# In[] Binary segmentation
from deeplab_demo import DeepLabModel
seg_model_path = "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz"
seg_model = DeepLabModel(seg_model_path)
resized_img, seg_map = seg_model.run(img)
resized_img = np.array(resized_img)[:w, :h]
seg_map = seg_map[:w, :h]

# In[]
plt.imshow(resized_img)

# In[]
plt.imshow(seg_map)

# In[] MORPHOLOGICAL POSTRPOCESSING OF MASK
# Largest blob
from morph import extract_largest_blob, fill_holes

mask = extract_largest_blob(seg_map)
plt.imshow(mask)

# In[]
# Filling holes
mask = fill_holes(mask)
plt.imshow(mask)

# In[] BLACK BACKGROUD ADDITION
# Add background
mask_extended = np.zeros((512, 512), dtype = np.bool)
mask_extended[-512:,:384] = mask

seg_image = resized_img * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
seg_image_extended = np.zeros((512, 512, 3), dtype=np.uint8)
seg_image_extended[-512:,:384,:] = seg_image

# In[] STYLE BANK
import torch
import torchvision.transforms as transforms
import pandas as pd

import sys
sys.path.append('stylebank/')

from networks import StyleBankNet
import stylebank.util as util

content_img_transform = transforms.Compose([
	util.Resize(513),
	transforms.CenterCrop([513, 513]),
	transforms.ToTensor()
])
    
trans = transforms.ToPILImage()

styles = pd.read_fwf('stylebank/styles_2.txt')
styles = styles.values[:,0]

device = "cpu"
style = 'anime1'
style_idx = (styles == style).argmax()

model = StyleBankNet(1).to(device)
model.encoder_net.load_state_dict(torch.load("stylebank/weights_test_2/encoder_2.pth"))
model.decoder_net.load_state_dict(torch.load("stylebank/weights_test_2/decoder_2.pth"))
model.style_bank[0].load_state_dict(torch.load("stylebank/weights_test_2/bank_2/{}_2.pth".format(style_idx)))

x = content_img_transform(trans(seg_image_extended))
styled_image = model(x.expand((1,3,513,513)), util.get_sid_batch(list(range(1)), 1))

styled_image = styled_image[0].cpu().detach()
styled_image = styled_image.clamp(min=0, max=1)
styled_image = styled_image.cpu().numpy().transpose(1, 2, 0)
styled_image = styled_image[:w,:w,:]

#data = styled_image.astype(np.float32) / 1. # normalize the data to 0 - 1
styled_image = 255 * styled_image # Now scale by 255
styled_image = styled_image.astype(np.uint8)
plt.imshow(styled_image)

# In[] GRADIENT
def get_gradient(size_g, channel=0, sigma=1, mu=0):
    x, y = np.meshgrid(np.linspace(-1, 1, size_g), np.linspace(-1, 1, size_g))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    final = np.zeros((512, 512, 3))
    final[:, :, channel] = g
    final *= 255
    final = final.astype(np.uint8)
    return final

gradient = get_gradient(512)
plt.imshow(gradient)

# In[] TRANSPARENT OVERLAY
def transparent_overlay_2(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            # read the alpha channel
            alpha = float(overlay[i][j][3] / 255.0)
            src[x + i][y + j] = alpha * overlay[i][j][:3] + \
                (1 - alpha) * src[x + i][y + j]

    return src

kek = styled_image*np.repeat(mask_extended[:, :, np.newaxis], 3, axis=2)
kek = cv2.cvtColor(kek, cv2.COLOR_RGB2RGBA)
kek[~mask_extended] = 0

asd = transparent_overlay_2(gradient, kek)

plt.imshow(asd)

# In[] COLOR BACKGROUD ADDITION
#tmp = cv2.cvtColor(seg_image, cv2.COLOR_RGB2GRAY)
#_, alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
#r, g, b = cv2.split(seg_image)
#rgba = [r,g,b, alpha]
#dst = cv2.merge(rgba,4)
#
#trans_mask = dst[:,:,3] == 0
#color = [0,255,0,255]
#dst[trans_mask] = color
#
#img_backgrounded = np.ones((512, 512, 4), dtype = np.uint8)*color
#img_backgrounded[-512:,:384,:] = dst
#img_backgrounded = img_backgrounded[...,:3]
#img_backgrounded = img_backgrounded.astype(np.uint8)
#
#plt.imshow(img_backgrounded)

# In[] Facial landmarks detection:
from imutils import face_utils
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)
shape = predictor(gray, rects[0])
shape = face_utils.shape_to_np(shape)
mouth_left_xy = shape[48]
mouth_mid_xy = shape[66]
mouth_right_xy = shape[54]
mouth_x, mouth_y = mouth_right_xy
chin_xy = shape[8]

# In[] OBLACHKO:
import imageio

font_size=25
font_color=(0,0,0)
unicode_font = ImageFont.truetype("DejaVuSans.ttf", font_size)

def transparent_overlay(src, overlay, text=['Hello!'], font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0), pos_txt=(0, 0), scale=1):

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    scr_cp = src.copy()
    numsteps = 3

    # loop over all pixels and apply the blending equation
    with imageio.get_writer('canadian.gif', mode='I') as writer:
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
            writer.append_data(scr_cp)
            
        canadian_img = scr_cp.copy()

        moving_mouth = canadian_img[mouth_left_xy[1]:chin_xy[1],mouth_left_xy[0]:mouth_right_xy[0],:].copy()
        mm_h, mm_w = moving_mouth.shape[:2]

        step = 10
        i = range(1, mm_h//2, step)
        j = 0

        y_pos = pos_txt[1]
        curr_img = scr_cp.copy()
        for l in text:
            curr_line = ''
            image_copy = np.zeros_like(img)
            for w in l:
                curr_line += w
                image_copy = curr_img.copy()
                
                image_copy[mouth_left_xy[1]:chin_xy[1],mouth_left_xy[0]:mouth_right_xy[0],:] = 0
                image_copy[mouth_left_xy[1]+i[j]:chin_xy[1]+i[j],mouth_left_xy[0]:mouth_right_xy[0],:] = moving_mouth
                
                #
                pilimg = Image.fromarray(image_copy)
                draw = ImageDraw.Draw(pilimg)
                draw.text ((pos_txt[0], y_pos-font_size), curr_line, font=unicode_font, fill=font_color)
                #
#                cv2.putText(image_copy, curr_line, (pos_txt[0], y_pos), font,
#                            0.8, (0, 0, 0), 1, cv2.LINE_AA)
                writer.append_data(np.array(pilimg))
                j += 1
                if (j >= len(i)):
                    j = 0
            image_copy = curr_img.copy()
            curr_img = image_copy
#            y_pos += dy

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

cloud = cv2.imread('data/images/clouds/cloud4.png', -1)

cloud_w = asd.shape[0] - mouth_x
cloud_ratio = cloud.shape[0]/cloud.shape[1]
cloud_h = int(cloud_ratio*cloud_w)
cloud_resized = cv2.resize(cloud, (cloud_w, cloud_h))
cloud_pos_x = mouth_x
cloud_pos_y = mouth_y - int(cloud_h*(5/4))
cloud_center_x = cloud_pos_x + int(cloud_w/5)
lines = check_text(message, max_line_length=cloud_w//20)
cloud_center_y = cloud_pos_y + int(cloud_h/2)
transparent_overlay(asd, cloud_resized, text=lines, pos=(cloud_pos_x, cloud_pos_y), pos_txt=(cloud_center_x, cloud_center_y))