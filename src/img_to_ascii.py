import cv2
import sys
import os
import numpy as np
import logging
from datetime import datetime


logging.getLogger().setLevel(logging.INFO)


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
        return np.zeros((h,w), dtype=np.uint8)
    return spritesheet[y:y+h, x:x+w]

def match_character_to_region(c, img_region):
    """Returns a ranking of how well the character is matched by the image"""

    global spritesheet

    char_img = char_to_img(c)
    if char_img.shape != img_region.shape:
        # Resize the spritesheet so that the sections will overlay
        x_scaling_factor = img_region.shape[1] / char_img.shape[1]
        y_scaling_factor = img_region.shape[0] / char_img.shape[0]
        new_shape = (
            int(spritesheet.shape[1] * x_scaling_factor),
            int(spritesheet.shape[0] * y_scaling_factor),
        )
        logging.info('resizing spritesheet from %dx%d to %dx%d',
            spritesheet.shape[1], spritesheet.shape[0],
            new_shape[0], new_shape[1])
        spritesheet = cv2.resize(spritesheet, new_shape)

        return match_character_to_region(c, img_region)

    combined = (char_img + img_region) / 2
    mean, _ = cv2.meanStdDev(combined)
    return 255 - mean

def get_character_from_region(img):
    """
    Finds the single ASCII character that best matches the image, according to
    match_character_to_region
    """

    best_fit_char = ' '
    imax = None
    for c in CHARACTERS:
        rank = match_character_to_region(c, img)
        if imax is None or rank > imax:
            best_fit_char = c
            imax = rank
    return best_fit_char

def parse_input(args):
    """
    Simple wrapper for cv2.imread which throws an error if the file is not
    found.

    Returns a numpy array and the width in characters given in the CLI.
    """

    if len(args) < 2:
        raise RuntimeError('usage: `img_to_ascii.py <image filename> <width (optional)>')
    elif len(args) == 2:
        w = 120
    else:
        w = int(args[2])

    img_path = args[1]
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
    else:
        raise FileNotFoundError(f'{img_path}')
    return img, w

def get_filename_from_args(args):
    """Lops of the path and extension of the input image file."""

    img_path = args[1]
    path_end = img_path.rfind('/')
    extension_start = img_path.rfind('.')
    return img_path[path_end+1:extension_start]

def get_spritesheet():
    img = cv2.imread('char_sheet.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)

    # Add a halo to the characters so that neighboring pixels are more likely
    # to contribute to a match
    blurred = cv2.blur(img, (5, 5))
    img = img + cv2.resize(blurred, (img.shape[1], img.shape[0]))

    return img

def avg_threshold(img):
    """
    Uses cv2.threshold to convert all pixels above the average to 255, and all
    below to 0
    """

    mean, std_dev = cv2.meanStdDev(img)
    _, thresh = cv2.threshold(img, int(mean), 255, cv2.THRESH_BINARY)
    return thresh


def main(*args):
    img, width = parse_input(args)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_height, img_width = gray.shape

    char_width = img_width // width
    char_height = 2 * char_width

    logging.info('character width: %d, character height: %d.', char_width, char_height)

    blurred = cv2.GaussianBlur(gray, (0, 0), 2*char_width+1)
    gray = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)

    thresh = avg_threshold(gray)

    table = []
    for y in range(img_height // char_height):
        y *= char_height
        row = []
        for x in range(width):
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
    t0 = datetime.now()

    spritesheet = get_spritesheet()
    main(*sys.argv)

    t1 = datetime.now()
    logging.info('total time: %d seconds', (t1 - t0).total_seconds())
