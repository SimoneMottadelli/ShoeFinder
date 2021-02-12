from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input as efficientnetb7_preprocess_input
import config


class Classifier:

    __classification_model = None

    @staticmethod
    def __generate_model():
        model = EfficientNetB7(input_tensor=Input(config.CLASSIFIER_IM_SIZE))
        model = Model(inputs=model.input, outputs=model.get_layer("top_dropout").output)
        new_output_layer = Dense(1, activation="sigmoid", name="output")(model.output)
        model = Model(inputs=model.input, outputs=new_output_layer)
        return model

    @staticmethod
    def init():
        """
        Initialize Classifier building the classification model and loading its corresponding weights
        """
        print("Initializing Classifier...")
        classifier = Classifier.__generate_model()
        classifier.load_weights(config.CLASSIFIER_WEIGHT_PATH)
        Classifier.__classification_model = classifier
        print("Done.")

    @staticmethod
    def classify_image(im):
        """
        Given an image (as a np tensor), return True if it is a shoe, False otherwise

        :param im: a tensor 1x128x128x3
        :return: True if im is a shoe, False otherwise
        """
        im_copy = im.copy()
        pred = Classifier.__classification_model.predict(efficientnetb7_preprocess_input(im_copy))
        return pred[0][0] > config.CLASSIFIER_THRESHOLD
