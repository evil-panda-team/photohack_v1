import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps

def get_ascii(image, message=None, scale=1, num_cols=200, background='white'):
    if message == None:
        CHAR_LIST = '@%#*+=-:. '
    else:
        # CHAR_LIST = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
        CHAR_LIST = message + '.....'
    if background == "white":
        bg_code = 255
    else:
        bg_code = 0
    font = ImageFont.truetype(
        "fonts/DejaVuSansMono-Bold.ttf", size=10 * scale)
    num_chars = len(CHAR_LIST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    cell_width = width / num_cols
    cell_height = 2 * cell_width
    num_rows = int(height / cell_height)
    if num_cols > width or num_rows > height:
        print("Too many columns or rows. Use default setting")
        cell_width = 6
        cell_height = 12
        num_cols = int(width / cell_width)
        num_rows = int(height / cell_height)
    char_width, char_height = font.getsize("A")
    out_width = char_width * num_cols
    out_height = char_height * num_rows
    out_image = Image.new("L", (out_width, out_height), bg_code)
    draw = ImageDraw.Draw(out_image)
    for i in range(num_rows):
        line = "".join([CHAR_LIST[min(int(np.mean(image[int(i * cell_height):min(int((i + 1) * cell_height), height),
                                                        int(j * cell_width):min(int((j + 1) * cell_width),
                                                                                width)]) * num_chars / 255), num_chars - 1)]
                        for j in
                        range(num_cols)]) + "\n"
        draw.text((0, i * char_height), line, fill=255 - bg_code, font=font)

    if background == "white":
        cropped_image = ImageOps.invert(out_image).getbbox()
    else:
        cropped_image = out_image.getbbox()
    # cv2.imshow("Frame", np.array(out_image))
    # cv2.waitKey(0)
    return np.array(out_image)


if __name__ == '__main__':
    image = cv2.imread('data/images/selfies/rafa.jpg')
    get_ascii(image, 'Test message', num_cols=500)
