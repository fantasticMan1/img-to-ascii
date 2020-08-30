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

# TODO: Standardize size of spritesheet so that incoming image can be resized
# around the characters


def char_to_img(c):
    """Retrieves an image slice representing any ASCII character"""

    global spritesheet
    n = CHARACTERS.index(c)
    h, w = spritesheet.shape
    h //= 15  # Spritesheet is 15 characters tall
    w //= 32  # Spritesheet is 32 characters wide
    y = (n // 32) * h
    x = (n % 32) * w
    if c == ' ':  # Spritesheet has no ` ` character, so handle that here.
        return np.zeros((h,w))
    return spritesheet[y+2:y+h, x+1:x+w-1]


def match_character_to_region(c, img_region):
    """Returns a ranking of how well the character is matched by the image"""

    # TODO: find a better matching algorithm: one that increases the score for
    # the brightness inside the character region, and decreases the score for
    # all characters outside the character region

    char_img = char_to_img(c)
    if char_img.shape != img_region.shape:
        char_img = cv2.resize(char_img, (img_region.shape[1], img_region.shape[0]))
    combined = (char_img + img_region) / 2
    mean, _ = cv2.meanStdDev(combined)
    return 255 - mean


def get_character_from_region(img):
    """
    Finds the single ASCII character that best matches the image, according to
    match_character_to_region
    """

    best_fit_char = 0
    imax = 0
    for c in CHARACTERS:
        rank = match_character_to_region(c, img)
        if rank > imax:
            best_fit_char = c
            imax = rank
    return best_fit_char

def get_img_from_args(args):
    """
    Simple wrapper for cv2.imread which throws an error if the file is not
    found.
    """

    img_path = args[1]
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
    else:
        raise FileNotFoundError(f'{img_path}')
    return img

def get_filename_from_args(args):
    """Lops of the path and extension of the input image file."""
    # TODO: add filename argument

    img_path = args[1]
    path_end = img_path.rfind('/')
    extension_start = img_path.find('.')
    return img_path[path_end+1:extension_start]

def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def avg_threshold(img):
    """
    Uses cv2.threshold to convert all pixels above the average to 255, and all
    below to 0
    """

    mean, std_dev = cv2.meanStdDev(img)
    _, thresh = cv2.threshold(img, int(mean), 255, cv2.THRESH_BINARY)
    return thresh


def main(*args):
    img = get_img_from_args(args)

    gray = convert_to_grayscale(img)

    img_height, img_width = gray.shape

    char_width = img_width // 120  # TODO: adjustable page width
    char_height = 2 * char_width

    thresh = avg_threshold(gray)

    table = []
    for y in range(img_height // char_height):
        y *= char_height
        row = []
        for x in range(120):
            x *= char_width
            img_slice = thresh[y:y+char_height, x:x+char_width]
            row.append(get_character_from_region(img_slice))
        table.append(row)

    fname = get_filename_from_args(args)
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
