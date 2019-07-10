#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image, ImageFont, ImageDraw, ImageOps
from memes_phrasez import memes_generate
import indicoio
import imageio
import dlib
from imutils import face_utils
import sys

sys.path.append('stylebank/')
from src.img_to_ascii import get_ascii
import stylebank.util as util
from networks import StyleBankNet
import random
import os
import cv2
import shutil
import numpy as np
from deeplab_demo import DeepLabModel
from morph import extract_largest_blob, fill_holes
import torch
import torchvision.transforms as transforms
import pandas as pd
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

indicoio.config.api_key = 'cd048edbe759544dfcd983946d18b1cf'


def fcn(img_name, message, result_path = "bot/static/res/server"):
    img = cv2.imread(img_name)

    if os.path.exists(result_path):
        shutil.rmtree(result_path)

    os.makedirs(result_path )


    if len(message)<12:
        res_ascii = get_ascii(img.copy(), message=message, num_cols=80)
        cv2.imwrite( result_path  + "/result" + str(random.randrange(10000)) + ".jpg", res_ascii)

    stylebank(img_name, message, result_path + "/result" + str(random.randrange(10000)) + ".gif")#,use_text_api=False)

    memes_generate(img.copy(), message, result_path  + "/result" + str(random.randrange(10000)) + ".jpg")




def transparent_overlay(src, overlay, mouth_left_xy, mouth_right_xy, chin_xy, save_path, text=['Hello!'],
                        font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0), pos_txt=(0, 0), scale=1, ):
    
    font_size=25
    font_color=(0,0,0)
    unicode_font = ImageFont.truetype("DejaVuSans.ttf", font_size)

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    scr_cp = src.copy()
    numsteps = 3

    # loop over all pixels and apply the blending equation
    with imageio.get_writer(save_path, mode='I') as writer:
        for al in range(numsteps):
            scr_cp = src.copy()
            for i in range(h):
                for j in range(w):
                    if x + i >= rows or y + j >= cols:
                        continue
                    # read the alpha channel
                    alpha = (1 / numsteps) * al * float(overlay[i][j][3] / 255.0)
                    scr_cp[x + i][y + j] = alpha * overlay[i][j][:3] + \
                                           (1 - alpha) * scr_cp[x + i][y + j]
            writer.append_data(scr_cp)

        canadian_img = scr_cp.copy()

        moving_mouth = canadian_img[mouth_left_xy[1]:chin_xy[1],
                       mouth_left_xy[0]:mouth_right_xy[0], :].copy()
        mm_h, mm_w = moving_mouth.shape[:2]

        step = 10
        i = range(1, mm_h // 2, step)
        j = 0

        y_pos = pos_txt[1]
        curr_img = scr_cp.copy()
        for l in text:
            curr_line = ''
            image_copy = np.zeros_like(src)
            for w in l:
                curr_line += w
                image_copy = curr_img.copy()

                image_copy[mouth_left_xy[1]:chin_xy[1],
                mouth_left_xy[0]:mouth_right_xy[0], :] = 0
                image_copy[mouth_left_xy[1] + i[j]:chin_xy[1] + i[j],
                mouth_left_xy[0]:mouth_right_xy[0], :] = moving_mouth

                pilimg = Image.fromarray(image_copy)
                draw = ImageDraw.Draw(pilimg)
                draw.text ((pos_txt[0], y_pos-font_size), curr_line, font=unicode_font, fill=font_color)
                
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
        if len(current_line + word) <= max_line_length:
            current_line = current_line + ' ' + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


def get_gradient(size_g, color=[0, 255, 255], sigma=1, mu=0):
    x, y = np.meshgrid(np.linspace(-1, 1, size_g), np.linspace(-1, 1, size_g))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    final = np.zeros((512, 512, 3))
    for i in range(3):
        final[:, :, i] = color[i]
        final[:, :, i] = final[:, :, i] * g
    final = final.astype(np.uint8)
    return final


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


def stylebank(img_name, message, out, use_text_api=True):
    img = Image.open(img_name)
    h, w = (384, 512)
    img = img.resize((h, w))
    # plt.imshow(img)

    seg_model_path = "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz"
    seg_model = DeepLabModel(seg_model_path)
    resized_img, seg_map = seg_model.run(img)
    resized_img = np.array(resized_img)[:w, :h]
    seg_map = seg_map[:w, :h]

    mask = extract_largest_blob(seg_map)
    mask = fill_holes(mask)

    mask_extended = np.zeros((512, 512), dtype=np.bool)
    mask_extended[-512:, :384] = mask

    seg_image = resized_img * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    seg_image_extended = np.zeros((512, 512, 3), dtype=np.uint8)
    seg_image_extended[-512:, :384, :] = seg_image

    # sys.path.append('../stylebank/')

    content_img_transform = transforms.Compose([
        util.Resize(513),
        transforms.CenterCrop([513, 513]),
        transforms.ToTensor()
    ])

    trans = transforms.ToPILImage()
    
    if use_text_api:
        sentiment = indicoio.sentiment(message)
    else:
        blob = TextBlob(message,
                        analyzer=NaiveBayesAnalyzer())
        sentiment = blob.sentiment.p_pos            
        
    positive = ['cyberpunk3', 'hotlinemiami2', 'popart1']
    neutral = ['anime1', 'popart2', 'walkingdead2']
    negative = ['manga1', 'sincity1', 'mickeymouse1']

    styles = pd.read_fwf('stylebank/styles_2.txt')
    styles = styles.values[:, 0]

    if sentiment > 2/3:
        style = random.choice(positive)
    elif sentiment > 1/3:
        style = random.choice(neutral)
    else:
        style = random.choice(negative)
        
    style_idx = (styles == style).argmax()

    model = StyleBankNet(1).to("cuda")
    model.encoder_net.load_state_dict(torch.load(
        "stylebank/weights_test_2/encoder_2.pth"))
    model.decoder_net.load_state_dict(torch.load(
        "stylebank/weights_test_2/decoder_2.pth"))
    model.style_bank[0].load_state_dict(torch.load(
        "stylebank/weights_test_2/bank_2/{}_2.pth".format(style_idx)))

    x = content_img_transform(trans(seg_image_extended)).cuda()
    styled_image = model(x.expand((1, 3, 513, 513)),
                         util.get_sid_batch(list(range(1)), 1))

    styled_image = styled_image[0].cpu().detach()
    styled_image = styled_image.clamp(min=0, max=1)
    styled_image = styled_image.cpu().numpy().transpose(1, 2, 0)
    styled_image = styled_image[:w, :w, :]

    styled_image = 255 * styled_image  # Now scale by 255
    styled_image = styled_image.astype(np.uint8)
    #styled_image = seg_image_extended

    rgb = np.zeros(3, dtype=np.uint8)

    if sentiment > 0.8:
        rgb = [0, 255, 0]
    elif sentiment > 0.6:
        rgb = [0, 255, 255]
    elif sentiment > 0.4:
        rgb = [0, 0, 255]
    elif sentiment > 0.2:
        rgb = [255, 255, 0]
    else:
        rgb = [255, 0, 0]

    gradient = get_gradient(512, color=rgb)

    kek = styled_image * np.repeat(mask_extended[:, :, np.newaxis], 3, axis=2)
    kek = cv2.cvtColor(kek, cv2.COLOR_RGB2RGBA)
    kek[~mask_extended] = 0
    asd = transparent_overlay_2(gradient, kek)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)
    mouth_left_xy = shape[48]
    mouth_right_xy = shape[54]
    mouth_x, mouth_y = mouth_right_xy
    chin_xy = shape[8]

    cloud = cv2.imread('data/images/clouds/cloud4.png', -1)

    cloud_w = asd.shape[0] - mouth_x
    cloud_ratio = cloud.shape[0] / cloud.shape[1]
    cloud_h = int(cloud_ratio * cloud_w)
    cloud_resized = cv2.resize(cloud, (cloud_w, cloud_h))
    cloud_pos_x = mouth_x
    cloud_pos_y = mouth_y - int(cloud_h * (5 / 4))
    cloud_center_x = cloud_pos_x + int(cloud_w / 5)
    lines = check_text(message, max_line_length=cloud_w // 20)
    cloud_center_y = cloud_pos_y + int(cloud_h / 2)

    transparent_overlay(asd, cloud_resized, text=lines, pos=(cloud_pos_x, cloud_pos_y), pos_txt=(
        cloud_center_x, cloud_center_y), mouth_left_xy=mouth_left_xy, mouth_right_xy=mouth_right_xy, chin_xy=chin_xy,
                        save_path=out)
