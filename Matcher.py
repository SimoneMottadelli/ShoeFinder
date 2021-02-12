from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from FeatureExtractor import FeatureExtractor
import numpy as np
from Indexer import Indexer


class Matcher:
    """
    This class allows to compare a given image with the indexed images to retrieve the most similar ones. Uses
    FeatureExtractor to extract features
    """

    __data_structure = None
    __pca = None
    __scaler = None

    @staticmethod
    def __normalize_data(data, scaler=None):
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(data)
        return scaler.transform(data), scaler

    @staticmethod
    def __apply_pca(data, pca=None, value=0.99):
        if pca is None:
            pca = PCA(n_components=value)
            pca.fit(data)
        data = pca.transform(data)
        return data, pca

    @staticmethod
    def init(data_structure):
        """
        Initialize Matcher applying MinMax normalization and PCA on the data structure given by Indexer

        :param data_structure: The data structure containing image names and their features
        """
        print("Initializing Matcher...")
        data_structure_copy = data_structure.copy()
        data_structure_copy["features"], scaler = Matcher.__normalize_data(data_structure_copy["features"])
        # pca is applied only to the neural features
        tmp_features, pca = Matcher.__apply_pca(data_structure_copy["features"][:, 1016:])
        data_structure_copy["features"] = np.concatenate((tmp_features, data_structure_copy["features"][:, :1016]), axis=1)
        Matcher.__data_structure = data_structure_copy
        Matcher.__pca = pca
        Matcher.__scaler = scaler
        print(data_structure_copy["features"].shape)
        print("Done.")

    @staticmethod
    def get_most_similar(im):
        """
        Extract the features from a given image (an np array) and compute the similarity between it and all the others that
        are in the indexed data structure. Feature extraction is performed using FeatureExtractor.

        :param im: a np array 256x256x3 representing an image
        :return: an np array (sorted) with the most similar images that have been indexed. The first column contains scores, the second one contains image names
        """
        im_copy = im.copy()

        # extract features from FeatureExtractor
        lbp, rgb_hist, ycbcr_hist, ycbcr_statistics, rgb_statistics, sift_kp, neural = FeatureExtractor.extract_all_features(im_copy)
        im_features = list(lbp) + list(rgb_hist) + list(ycbcr_hist) +  list(ycbcr_statistics) + list(rgb_statistics) + list(FeatureExtractor.extract_bow(Indexer.kmeans_model, sift_kp)) + list(neural)

        # apply MinMax normalization using the train scaler and the PCA on the neural features
        im_features = np.expand_dims(np.array(im_features), axis=0)
        im_features, _ = Matcher.__normalize_data(im_features, Matcher.__scaler)
        tmp_features, _ = Matcher.__apply_pca(im_features[:, 1016:], Matcher.__pca)
        im_features = np.concatenate((tmp_features, im_features[:, :1016]), axis=1)

        # compute cosine similarity between the image and the other indexed images
        rsv = cdist(im_features, Matcher.__data_structure["features"], metric="cosine")
        rsv = np.append(rsv.transpose(), np.array([Matcher.__data_structure["im_names"]], dtype="object").transpose(), axis=1)

        # sort the results according to the higher score
        sorted_rsv = rsv[rsv[:, 0].argsort()]

        return sorted_rsv

    @staticmethod
    def retrieve_items(sorted_rsv, n=3):
        """
        Given the sorted list of similar images, retrieve the n most similar images belonging to different classes

        :param sorted_rsv: a sorted np array. The first column contains scores, the second one contains image names
        :param n: the number of images to retrieve
        :return: list of retrieved image names
        """
        images = sorted_rsv[:, 1]
        image_classes_dict = Indexer.extract_classes()
        classes_already_retrieved = []
        retrieved = []
        for image in images:
            class_id = image_classes_dict[image]
            if class_id not in classes_already_retrieved:
                retrieved.append(image)
                classes_already_retrieved.append(class_id)
                if len(retrieved) == n:
                    break
        return retrieved
