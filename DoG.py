import numpy as np
import cv2

class Difference_of_Gaussian(object):

    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):

        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        octaves = []
        base_image = np.copy(image)

        for _ in range(self.num_octaves):

            # Put the base image of each octave at first.
            octave = [base_image]

            for level in range(1,self.num_guassian_images_per_octave):

                sigma = self.sigma ** level

                # Push the blurred image into octave
                blurred_image = cv2.GaussianBlur(
                    src=base_image, ksize=(0,0), sigmaX=sigma, sigmaY=sigma)

                octave.append(blurred_image)

                # The base image in next octave is the half of the last blurred image
                if len(octave) == self.num_guassian_images_per_octave:

                    new_height, new_width = blurred_image.shape[0] // 2, blurred_image.shape[1] // 2
                    base_image = cv2.resize(
                        blurred_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

            octaves.append(octave)
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)

        # Store DoG image packaged in octave
        dog_arrays = []

        for octave in octaves:

            dog_list = [cv2.subtract(octave[i+1], octave[i]) for i in range(self.num_DoG_images_per_octave)]
            dog_array = np.array(dog_list)
            dog_arrays.append(dog_array)

        self.dog_arrays = dog_arrays

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        keypoints = np.zeros((0,2)).astype(np.int32)

        # Each dog_array is an octave
        for i, dog_array in enumerate(dog_arrays):
            for layer in range(dog_array.shape[0]):
                for row in range(dog_array.shape[1]):
                    for col in range(dog_array.shape[2]):

                        # Skip the DoG value less than threshold
                        if abs(dog_array[layer, row, col]) <= self.threshold:
                            continue

                        cube = dog_array[layer-1:layer+2, row-1:row+2, col-1:col+2]

                        # If the cube out of index, numpy won't raise an error but a incomplete ndarray
                        if cube.size != 27:
                            continue

                        local_max = np.max(cube)
                        local_min = np.min(cube)
                        cur_value = dog_array[layer, row, col]

                        if cur_value == local_max or cur_value == local_min:

                            # The keypoints found in i-th octave are scaled by 2^i
                            kp = np.array([row * 2**i, col * 2**i]).astype(np.int32)
                            keypoints = np.append(keypoints, [kp], axis=0)

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]

        return keypoints

    def get_dog_images(self, image):

        # self.dog_arrays is obtained from get_keypoints()
        if 'dog_arrays' not in self.__dict__:
            _ = self.get_keypoints(image)

        # Map real number to 0-255
        def normalize_image(img):
            Max = np.max(img)
            Min = np.min(img)
            norm = (img-Min) / (Max-Min) * 255
            return norm.astype(np.uint8)

        dog_images = dict()

        for octave in range(self.num_octaves):
            for dog in range(self.num_DoG_images_per_octave):
                norm = normalize_image(self.dog_arrays[octave][dog])
                dog_images.setdefault(f'DoG{octave+1}-{dog+1}.png', norm)

        return dog_images
