import cv2
import argparse
import numpy as np
from pathlib import Path
from DoG import Difference_of_Gaussian

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='./input/2.png', help='path to input image')
parser.add_argument('--out_dir', default='./output', help='path to output directory')

args = parser.parse_args()


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)


# Read the image
img = cv2.imread(args.image_path, 0).astype(np.float32)

# Set the output directory
output_dir = Path(args.out_dir)
output_dir.mkdir(exist_ok=True)

# Use three thresholds (1,2,3) on the image and plot the key points.
for threshold in range(1,4):
    DoG = Difference_of_Gaussian(threshold)
    keypoints = DoG.get_keypoints(img)

    output_path = output_dir / f'TH{threshold}.png'
    plot_keypoints(img, keypoints, str(output_path))
