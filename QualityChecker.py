import numpy as np
from scipy.signal import convolve2d
import math
import config
from FeatureExtractor import FeatureExtractor
import cv2
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input as efficientnetb7_preprocess_input
import config


class QualityChecker:

    __classification_model = None

    @staticmethod
    def __generate_model():
        model = EfficientNetB7(input_tensor=Input(config.QUALITYCHECKER_IMSIZE))
        model = Model(inputs=model.input, outputs=model.get_layer("top_dropout").output)
        new_output_layer = Dense(1, activation="sigmoid", name="output")(model.output)
        model = Model(inputs=model.input, outputs=new_output_layer)
        return model

    @staticmethod
    def init():
        print("Initializing QualityChecker...")
        classifier = QualityChecker.__generate_model()
        classifier.load_weights(config.QUALITYCHECKER_WEIGHT_PATH)
        QualityChecker.__classification_model = classifier
        print("Done.")

    @staticmethod
    def good_quality(im):
        im_copy = im.copy()
        pred = QualityChecker.__classification_model.predict(efficientnetb7_preprocess_input(im_copy))
        return pred[0][0] > config.QUALITYCHECKER_THRESHOLD

'''
    @staticmethod
    def __estimate_noise(im):
        height, width = im.shape
        mask = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
        sigma = np.sum(np.sum(np.absolute(convolve2d(im, mask))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (width - 2) * (height - 2))
        print("Sigma:" + str(sigma))
        return sigma

    @staticmethod
    def __is_noisy(im):
        im = FeatureExtractor.rgb2grayscale(im)
        sigma = QualityChecker.__estimate_noise(im)
        return sigma > config.GAUSSIAN_SIGMA

    @staticmethod
    def __is_blurry(im):
        im = np.uint8(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        max_value = np.max(cv2.convertScaleAbs(cv2.Laplacian(im, 3)))
        print("Blur: " + str(max_value))
        return max_value < config.BLURRY_THRESHOLD

    @staticmethod
    def __is_dark(im):
        ycbcr_im = FeatureExtractor.rgb2ycbcr(im)
        y_channel = ycbcr_im[:, :, 0]
        mean = np.mean(y_channel)
        print("Dark: " + str(mean))
        return mean < config.DARK_THRESHOLD

    @staticmethod
    def check_quality(im):
        if QualityChecker.__is_noisy(im[0]) or QualityChecker.__is_blurry(im[0]) or QualityChecker.__is_dark(im[0]):
            return False
        else:
            return True

import numpy as np
from scipy.signal import convolve2d
import math
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image

def rgb2ycbcr(im):
      im_copy = im.copy()
      im_copy = Image.fromarray(im_copy.astype('uint8'))
      im_copy = im_copy.convert("YCbCr")
      y, cb, cr = im_copy.split()
      im_copy = np.dstack((y, cb, cr))
      return im_copy
def rgb2grayscale(im):
      im_copy = im.copy()
      im_copy = Image.fromarray(im_copy.astype('uint8'))
      im_copy = ImageOps.grayscale(im_copy)
      return np.array(im_copy)

def load_image(im_path):
      im = image.load_img(im_path)
      im = image.img_to_array(im)
      #im = np.expand_dims(im, axis=0)
      return im # returns a tensor 1x...


class QualityChecker:

    @staticmethod
    def __estimate_noise(im):
        height, width = im.shape
        mask = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
        sigma = np.sum(np.sum(np.absolute(convolve2d(im, mask))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (width - 2) * (height - 2))
        print("Sigma:" + str(sigma))
        return sigma

    @staticmethod
    def __is_noisy(im):
        im = FeatureExtractor.rgb2grayscale(im)
        sigma = QualityChecker.__estimate_noise(im)

    @staticmethod
    def __is_blurry(im):
        im = np.uint8(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        max_value = np.max(cv2.convertScaleAbs(cv2.Laplacian(im, 3)))
        print("Blur: " + str(max_value))

    @staticmethod
    def __is_dark(im):
        ycbcr_im = rgb2ycbcr(im)
        y_channel = ycbcr_im[:, :, 0]
        mean = np.mean(y_channel)
        print("Dark: " + str(mean))

    @staticmethod
    def check_quality(im):
        QualityChecker.__is_noisy(im)
        QualityChecker.__is_blurry(im)
        QualityChecker.__is_dark(im)

if __name__ == "__main__":
        QualityChecker.check_quality(load_image("./400139000931_158536.jpg"))
'''