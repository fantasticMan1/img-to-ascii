import cv2
import sys
import os
import numpy as np
import pdb


CHARACTERS = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>',
    '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\',
    ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
    'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '{', '|', '}', '~', ' ',
]


spritesheet = None


# def make_sprite_sheet(char_res):
#     h, w = char_res

#     with open('IBMCourierCharmap.svg', 'r') as svg:
#         svg_text = svg.read()

#     svg2png(bytestring=svg_text, write_to='char_sprite.png',
#             output_width=32*w, output_height=15*h)


def char_img_from_spritesheet(c):
    global spritesheet
    n = CHARACTERS.index(c)
    h, w = spritesheet.shape
    h //= 15
    w //=32
    y = (n // 32) * h
    x = (n % 32) * w
    if c == ' ':
        return np.zeros((h,w))
    return spritesheet[y+2:y+h, x+1:x+w-1]


def match_character_to_region(c, img):
    """Returns a ranking of how well the character is matched by the image"""

    char_img = char_img_from_spritesheet(c)
    if char_img.shape != img.shape:
        char_img = cv2.resize(char_img, (img.shape[1], img.shape[0]))
    combined = (char_img + img) / 2
    mean, _ = cv2.meanStdDev(combined)
    return 255 - mean


def get_character_from_region(img):
    best_fit_char = 0
    max = 0
    for c in CHARACTERS:
        rank = match_character_to_region(c, img)
        if rank > max:
            best_fit_char = c
            max = rank
    return best_fit_char


def main(*args):
    img_path = args[1]
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
    else:
        raise RuntimeError(f'File not found: {img_path}')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_height, img_width = gray.shape

    FONT_HEIGHT_TO_WIDTH = 2
    char_width = img_width // 120
    char_height = int(char_width * FONT_HEIGHT_TO_WIDTH)

    mean, std_dev = cv2.meanStdDev(gray)
    _, thresh = cv2.threshold(gray, int(mean), 255, cv2.THRESH_BINARY)

    table = []
    for y in range(img_height // char_height):
        y *= char_height
        row = []
        for x in range(120):
            x *= char_width
            img_slice = thresh[y:y+char_height, x:x+char_width]
            row.append(get_character_from_region(img_slice))
        table.append(row)
    
    fname = img_path[:img_path.index('.')]  # Remove file extension
    with open(fname + '.txt', 'w') as f:
        lines = []
        for row in table:
            row.append('\n')
            lines.append(''.join(row))
        f.writelines(lines)

if __name__ == '__main__':
    spritesheet = cv2.imread('char_sprite.png', cv2.IMREAD_UNCHANGED)

    if len(sys.argv) > 1:
        main(*sys.argv)
