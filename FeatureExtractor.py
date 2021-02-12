from PIL import Image, ImageOps
from skimage.feature.texture import local_binary_pattern
from scipy.stats import skew
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import cv2
import config
import numpy as np
from Segmenter import Segmenter
from NoMaskException import NoMaskException


class FeatureExtractor:
    """
    This class allows to extract several features of a given image
    """

    __feature_extractor_model = None

    @staticmethod
    def __generate_model():
        model = EfficientNetB7(include_top=False, input_shape=config.SEGMENTER_IM_SIZE)
        model = Model(inputs=model.input, outputs=model.output)
        layer = GlobalAveragePooling2D(name="top_g_a_pool")(model.output)
        layer = Dropout(rate=0.7, name="dropout")(layer)
        layer = Dense(config.CLASS_NUMBER, activation="softmax", name="output")(layer)
        model = Model(inputs=model.input, outputs=layer)
        model.load_weights(config.FE_WEIGHTS_PATH)
        model = Model(inputs=model.input, outputs=model.get_layer("top_g_a_pool").output)
        return model

    @staticmethod
    def init():
        """
        Initialize FeatureExtraction initializing Segmenter
        """
        print("Initializing FeatureExtractor...")
        FeatureExtractor.__feature_extractor_model = FeatureExtractor.__generate_model()
        Segmenter.init()
        print("Done.")

    @staticmethod
    def extract_neural_features(im):
        """
        Given an image (as np tensor), extract neural features from the last layer of a pretrained Efficientnetb7

        :param im: a np tensor 1x256x256x3
        :return: a np tensor of size 2560
        """
        im_copy = im.copy()
        im_copy = efficientnet_preprocess_input(im_copy)
        result = FeatureExtractor.__feature_extractor_model.predict(im_copy)
        return result[0]

    @staticmethod
    def rgb2grayscale(im):
        """
        Convert an image from RGB to grayscale

        :param im: a np array NxMx3
        :return: a np array NxM corresponding to the grayscale of the input
        """
        im_copy = im.copy()
        im_copy = Image.fromarray(im_copy.astype('uint8'))
        im_copy = ImageOps.grayscale(im_copy)
        return np.array(im_copy)

    @staticmethod
    # requires an RGB image and a 3d mask
    def compute_LBP_rgb(im, mask):
        """
        Given a segmented image and a segmentation mask compute LBP and return the corresponding normalized histogram
        with 256 bins

        :param im: an np array NxMx3 of a segmented image
        :param mask: a np array NxMx3 of the segmentation mask
        :return: a np array corresponding to the lbp normalized histogram with 256 bins
        """
        im_copy = im.copy()
        mask_copy = mask.copy()
        im_copy = FeatureExtractor.rgb2grayscale(im_copy)
        P, R = 8, 1
        dim = 2 ** P
        mask_copy = mask_copy[:, :, 0].astype(bool)
        codes = local_binary_pattern(im_copy, P, R, method="ror")
        # hist, _ = np.histogram(codes[mask], bins=np.arange(dim + 1), range=(0, dim))
        hist, _ = np.histogram(codes[mask_copy], bins=256, range=(0, 255))
        norm_hist = hist / np.sum(hist)
        return norm_hist

    @staticmethod
    # requires an image in the RGB space and a 0,1 mask in the double format
    def extract_statistics_ycbcr(im, mask):
        """
        Compute mean, std, skew and median of the Cb and Cr channels

        :param im: a np array corresponding to a segmented RGB image
        :param mask: a np array corresponding to a segmentation mask
        :return: mean, std, skew and median of the Cb and Cr channels
        """
        im_copy = im.copy()
        mask_copy = mask.copy()
        im_copy = FeatureExtractor.rgb2ycbcr(im_copy)
        cb = im_copy[:, :, 1]
        cr = im_copy[:, :, 2]
        mask_copy = mask_copy[:, :, 0].astype(bool)
        mean_cb = np.mean(cb[mask_copy])
        mean_cr = np.mean(cr[mask_copy])
        std_cb = np.std(cb[mask_copy])
        std_cr = np.std(cr[mask_copy])
        skew_cb = skew(cb[mask_copy])
        skew_cr = skew(cr[mask_copy])
        median_cb = np.median(cb[mask_copy])
        median_cr = np.median(cr[mask_copy])
        return np.array([mean_cb, mean_cr, std_cb, std_cr, skew_cb, skew_cr, median_cb, median_cr])

    @staticmethod
    def extract_statistics_rgb(im, mask):
        """
        Compute mean, std, skew and median of the RGB channels

        :param im: a np array corresponding to a segmented RGB image
        :param mask: a np array corresponding to a segmentation mask
        :return: mean, std, skew and median of the 3 RGB channels
        """
        im_copy = im.copy()
        mask_copy = mask.copy()
        r = im_copy[:, :, 0]
        g = im_copy[:, :, 1]
        b = im_copy[:, :, 2]
        mask_copy = mask_copy[:, :, 0].astype(bool)
        mean_r = np.mean(r[mask_copy])
        mean_g = np.mean(g[mask_copy])
        mean_b = np.mean(b[mask_copy])
        std_r = np.std(r[mask_copy])
        std_g = np.std(g[mask_copy])
        std_b = np.std(b[mask_copy])
        skew_r = skew(r[mask_copy])
        skew_g = skew(g[mask_copy])
        skew_b = skew(b[mask_copy])
        median_r = np.median(r[mask_copy])
        median_g = np.median(g[mask_copy])
        median_b = np.median(b[mask_copy])
        return np.array([mean_r, mean_g, mean_b, std_r, std_g, std_b, skew_r, skew_g, skew_b, median_r, median_g, median_b])

    @staticmethod
    def extract_sift_kp(im, mask):
        """
        Compute SIFT keypoints of a given image and return their SIFT descriptions

        :param im: a np array corresponding to a segmented RGB image
        :param mask: a np array corresponding to the segmentation mask
        :return: a np array containing the description of the keypoints found
        """
        im_copy = im.copy()
        mask_copy = mask.copy()
        mask_copy = mask_copy[:, :, 0].astype(np.uint8)
        sift = cv2.xfeatures2d.SIFT_create(400)
        gray = FeatureExtractor.rgb2grayscale(im_copy)
        kp, des = sift.detectAndCompute(gray, mask_copy)
        return des

    @staticmethod
    def extract_all_features(im):
        """
        Given an image (as np tensor) extract several features

        :param im: a np tensor 1x256x256x3
        :return: lbp, rgb histogram, ycbcr histogram, ycbcr statistics, rgb statistics, sift descriptors and neural features
        """
        im_copy = im.copy()
        segmented_im, mask = Segmenter.segment_image(im_copy)
        if np.sum(mask) == 0:
            raise NoMaskException()
        rgb_hist = FeatureExtractor.extract_rgb_hist(segmented_im, mask)
        ycbcr_hist = FeatureExtractor.extract_ycbcr_hist(segmented_im, mask)
        neural = FeatureExtractor.extract_neural_features(im_copy)
        lbp = FeatureExtractor.compute_LBP_rgb(segmented_im, mask)
        ycbcr_statistics = FeatureExtractor.extract_statistics_ycbcr(segmented_im, mask)
        rgb_statistics = FeatureExtractor.extract_statistics_rgb(segmented_im, mask)
        sift_kp = FeatureExtractor.extract_sift_kp(segmented_im, mask)
        return lbp, rgb_hist, ycbcr_hist, ycbcr_statistics, rgb_statistics, sift_kp, neural

    @staticmethod
    def rgb2ycbcr(im):
        """
        Convert an RGB image to the YCbCr space

        :param im: an np array of an RGB image
        :return: an np array corresponding to the YCbCr space of the input image
        """
        im_copy = im.copy()
        im_copy = Image.fromarray(im_copy.astype('uint8'))
        im_copy = im_copy.convert("YCbCr")
        y, cb, cr = im_copy.split()
        im_copy = np.dstack((y, cb, cr))
        return im_copy

    @staticmethod
    def extract_rgb_hist(im, mask):
        """
        Compute the normalized RGB histogram of the input image. The 3 histograms are concatenated

        :param im: an np array of a segmented image
        :param mask: an np array of a segmentation mask
        :return: a np array corresponding to the RGB histogram of the input image
        """
        im_copy = im.copy()
        mask_copy = mask.copy()
        mask_copy = mask_copy[:, :, 0].astype(bool)
        r = im_copy[:, :, 0]
        g = im_copy[:, :, 1]
        b = im_copy[:, :, 2]
        hist_r, _ = np.histogram(r[mask_copy], bins=128, range=(0, 255))
        norm_hist_r = hist_r / np.sum(hist_r)
        hist_g, _ = np.histogram(g[mask_copy], bins=128, range=(0, 255))
        norm_hist_g = hist_g / np.sum(hist_g)
        hist_b, _ = np.histogram(b[mask_copy], bins=128, range=(0, 255))
        norm_hist_b = hist_b / np.sum(hist_b)
        return np.array(list(norm_hist_r) + list(norm_hist_g) + list(norm_hist_b))

    @staticmethod
    def extract_ycbcr_hist(im, mask):
        """
        Compute the normalized YCbCr histogram of the input image. The 2 histograms (Cb and Cr )are concatenated

        :param im: an np array of a segmented image
        :param mask: an np array of a segmentation mask
        :return: a np array corresponding to the Cb and Cr histograms of the input image
        """
        im_copy = im.copy()
        mask_copy = mask.copy()
        im_copy = FeatureExtractor.rgb2ycbcr(im_copy)
        mask_copy = mask_copy[:, :, 0].astype(bool)
        cb = im_copy[:, :, 1]
        cr = im_copy[:, :, 2]
        hist_cb, _ = np.histogram(cb[mask_copy], bins=128, range=(0, 255))
        norm_hist_cb = hist_cb / np.sum(hist_cb)
        hist_cr, _ = np.histogram(cr[mask_copy], bins=128, range=(0, 255))
        norm_hist_cr = hist_cr / np.sum(hist_cr)
        return np.array(list(norm_hist_cb) + list(norm_hist_cr))

    @staticmethod
    def extract_bow(model, sift_kp_set):
        """
        Use the KMeans model given as input to compute the BOW histogram

        :param model: KMeans model from Indexer to perform BOW with SIFT
        :param sift_kp_set:
        :return: an np array of the normalized histogram of words
        """
        visual_words = model.predict(sift_kp_set)
        visual_words_hist, _ = np.histogram(visual_words, bins=100, range=(0, 99))
        norm_visual_words_hist = visual_words_hist / np.sum(visual_words_hist)
        return norm_visual_words_hist
