import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian
from os import makedirs


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    ### TODO ###
    DoG = Difference_of_Gaussian(threshold=args.threshold)
    keypoints = DoG.get_keypoints(img)
    DoG_images = DoG.get_dog_images(img)

    makedirs('output', exist_ok=True)

    # Part.1-1: Visualize the DoG images of 1.png.
    for filename, DoG_image in DoG_images.items():
        cv2.imwrite(filename, DoG_image)



if __name__ == '__main__':
    main()