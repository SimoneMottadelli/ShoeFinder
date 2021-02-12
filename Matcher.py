from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from FeatureExtractor import FeatureExtractor
import numpy as np
from Indexer import Indexer


class Matcher:

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
        print("Initializing Matcher...")
        data_structure_copy = data_structure.copy()
        data_structure_copy["features"], scaler = Matcher.__normalize_data(data_structure_copy["features"])
        tmp_features, pca = Matcher.__apply_pca(data_structure_copy["features"][:, 1016:])
        data_structure_copy["features"] = np.concatenate((tmp_features, data_structure_copy["features"][:, :1016]), axis=1)
        Matcher.__data_structure = data_structure_copy
        Matcher.__pca = pca
        Matcher.__scaler = scaler
        print(data_structure_copy["features"].shape)
        print("Done.")

    @staticmethod
    def get_most_similar(im): #ricordarsi di mettere N
        im_copy = im.copy()
        lbp, rgb_hist, ycbcr_hist, ycbcr_statistics, rgb_statistics, sift_kp, neural = FeatureExtractor.extract_all_features(im_copy)
        im_features = list(lbp) + list(rgb_hist) + list(ycbcr_hist) +  list(ycbcr_statistics) + list(rgb_statistics) + list(FeatureExtractor.extract_bow(Indexer.kmeans_model, sift_kp)) + list(neural)
        im_features = np.expand_dims(np.array(im_features), axis=0)
        im_features, _ = Matcher.__normalize_data(im_features, Matcher.__scaler)
        tmp_features, _ = Matcher.__apply_pca(im_features[:, 1016:], Matcher.__pca)
        im_features = np.concatenate((tmp_features, im_features[:, :1016]), axis=1)
        rsv = cdist(im_features, Matcher.__data_structure["features"], metric="cosine")
        rsv = np.append(rsv.transpose(), np.array([Matcher.__data_structure["im_names"]], dtype="object").transpose(), axis=1)
        sorted_rsv = rsv[rsv[:, 0].argsort()]
        return sorted_rsv

    @staticmethod
    def retrieve_items(sorted_rsv, n=3):
        images = sorted_rsv[:, 1]
        image_classes_dict = Indexer.extract_classes()
        classes_already_retrieved = []
        retrieved = []
        for image in images:
            class_id = image_classes_dict[image]
            if class_id not in classes_already_retrieved:
                retrieved.append(image)
                classes_already_retrieved.append(class_id)
                if len(retrieved) == 3:
                    break
        return retrieved
