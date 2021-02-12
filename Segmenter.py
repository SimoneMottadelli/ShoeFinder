import segmentation_models as sm
import config
import numpy as np


class Segmenter:
    """
    This class allows to segment an image using a pretrained UNet with EfficientnetB7 as backbone
    """

    __segmentation_model = None
    __preprocess_input = None

    @staticmethod
    def init():
        """
        Initialize Segmenter building the segmentation model (UNet with Efficientnetb7 as backbone) and loading
        its corresponding weights
        """
        print("Initializing Segmenter...")
        model = sm.Unet(config.SEGMENTER_BACKBONE, encoder_weights='imagenet', input_shape=config.SEGMENTER_IM_SIZE)
        model.load_weights(config.SEGMENTER_WEIGHT_PATH)
        Segmenter.__segmentation_model = model
        Segmenter.__preprocess_input = sm.get_preprocessing(config.SEGMENTER_BACKBONE)
        print("Done.")

    @staticmethod
    def segment_image(im):
        """
        Segment an image (a np tensor 1x256x256x3) and return the segmented image and the segmentation mask

        :param im: a np tensor 1x256x256x3
        :return: the segmented image with black background with shape 256x256x3, the segmentation mask with shape 256x256x3
        """
        im_copy = im.copy()
        pred = Segmenter.__segmentation_model.predict(Segmenter.__preprocess_input(im_copy))
        mask_2d = pred[0][:, :, 0] > config.SEGMENTER_CONFIDENCE
        mask_3d = np.dstack([mask_2d]*3).astype('float')
        im = im[0]
        return (im * mask_3d).astype(np.uint8), mask_3d
