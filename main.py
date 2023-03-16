import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian
from os import makedirs


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():

    img1 = cv2.imread('./testdata/1.png', 0).astype(np.float32)
    img2 = cv2.imread('./testdata/2.png', 0).astype(np.float32)

    DoG = Difference_of_Gaussian(threshold=3.0)
    DoG_images = DoG.get_dog_images(img1)

    makedirs('output', exist_ok=True)

    # Part.1-1: Visualize the DoG images of 1.png.
    for filename, DoG_image in DoG_images.items():
        cv2.imwrite(filename, DoG_image)

    # Part.1-2: Use three thresholds (1,2,3) on 2.png and describe the difference.
    for threshold in range(1,4):
        DoG = Difference_of_Gaussian(threshold)
        keypoints = DoG.get_keypoints(img2)
        plot_keypoints(img2, keypoints, f'./output/TH{threshold}.png')



if __name__ == '__main__':
    main()