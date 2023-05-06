import cv2
import argparse
import numpy as np
from pathlib import Path
from DoG import Difference_of_Gaussian

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='./input/1.png', help='path to input image')
parser.add_argument('--out_dir', default='./output', help='path to output directory')
parser.add_argument('--threshold', default=3.0, type=float, help='threshold value for feature selection')

args = parser.parse_args()

# Read the image
img = cv2.imread(args.image_path, 0).astype(np.float32)

# Generate DoG images
DoG = Difference_of_Gaussian(threshold=args.threshold)
DoG_images = DoG.get_dog_images(img)

# Set up the output directory
output_dir = Path(args.out_dir)
output_dir.mkdir(exist_ok=True)

# Output the DoG images.
for filename, DoG_image in DoG_images.items():
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), DoG_image)
