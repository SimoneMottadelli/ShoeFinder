import segmentation_models as sm
import config
import numpy as np


class Segmenter:

    __segmentation_model = None
    __preprocess_input = None

    @staticmethod
    def init():
        print("Initializing Segmenter...")
        model = sm.Unet(config.SEGMENTER_BACKBONE, encoder_weights='imagenet', input_shape=config.SEGMENTER_IM_SIZE)
        model.load_weights(config.SEGMENTER_WEIGHT_PATH)
        Segmenter.__segmentation_model = model
        Segmenter.__preprocess_input = sm.get_preprocessing(config.SEGMENTER_BACKBONE)
        print("Done.")

    @staticmethod
    def segment_image(im):
        im_copy = im.copy()
        pred = Segmenter.__segmentation_model.predict(Segmenter.__preprocess_input(im_copy))
        mask_2d = pred[0][:, :, 0] > config.SEGMENTER_CONFIDENCE
        mask_3d = np.dstack([mask_2d]*3).astype('float')
        im = im[0]
        return (im * mask_3d).astype(np.uint8), mask_3d
