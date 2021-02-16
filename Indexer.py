import os
import pandas as pd
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import config
import pickle
import numpy as np
from FeatureExtractor import FeatureExtractor
from FeatureExtractionException import FeatureExtractionException


class Indexer:
    """
    This class is used to build the data structure that has to be used by Matcher. It also provides the utility
    function for loading images
    """

    kmeans_model = None

    @staticmethod
    def load_image(im_path, im_size=config.SEGMENTER_IM_SIZE):
        """
        Load and return an image

        :param im_path: the path of the image
        :param im_size: the desidered size
        :return: a tensor 1xNxMxK
        """

        im = image.load_img(im_path, target_size=im_size)
        im = image.img_to_array(im)
        im = np.expand_dims(im, axis=0)
        return im

    @staticmethod
    def extract_classes():
        """
        Read the dataset.csv file and build a python dictionary

        :return: a python dictionary: key -> image_name, value -> the correspondent class
        """

        train_csv = pd.read_csv(config.CSV_PATH, sep=";", header=None)
        filenames = list(train_csv[0])
        classes = list(train_csv[1])
        dictionary = {}
        for i in range(len(filenames)):
            dictionary[filenames[i]] = classes[i]
        return dictionary

    @staticmethod
    def build_data_structure(index_folder):
        """
        Build the data structure that has to be used by Matcher. If a predefined data structure and KMeans model exist,
        it will use them.

        :param index_folder: the path to the images to be indexed
        :return: the data structure that has to be used by Matcher
        """

        # if a predefined data structure and KMeans model exist, then load and use them
        if os.path.isfile(config.INDEXING_DATA_STRUCTURE_PATH) and os.path.isfile(config.INDEXING_KMEANS_PATH):
            print("Loading KMeans model and data_structure...")
            Indexer.kmeans_model = pickle.load(open(config.INDEXING_KMEANS_PATH, "rb"))
            data_structure = pickle.load(open(config.INDEXING_DATA_STRUCTURE_PATH, "rb"))
            print("Done.")
            return data_structure

        # build the data structure
        data_structure = {"im_names": [], "lbp": [], "rgb_hist": [], "ycbcr_hist": [], "rgb_statistics": [], "ycbcr_statistics": [], "neural": [], "sift_kp": [], "labels": []}
        files = os.listdir(index_folder)
        files.sort()
        image_classes_dict = Indexer.extract_classes()

        # from each image, extract the features using FeatureExtractor
        for file in tqdm(files):
            im = Indexer.load_image(index_folder + file)
            try:
                lbp, rgb_hist, ycbcr_hist, ycbcr_statistics, rgb_statistics, sift_kp, neural = FeatureExtractor.extract_all_features(im)
            except FeatureExtractionException:
                continue
            if sift_kp is None or len(sift_kp.shape) == 0:
                print("Found image without any sift descriptors: ", file)
                continue
            data_structure["rgb_hist"].append(rgb_hist)
            data_structure["ycbcr_hist"].append(ycbcr_hist)
            data_structure["lbp"].append(lbp)
            data_structure["ycbcr_statistics"].append(ycbcr_statistics)
            data_structure["rgb_statistics"].append(rgb_statistics)
            data_structure["sift_kp"].append(sift_kp)
            data_structure["neural"].append(neural)
            data_structure["im_names"].append(file)
            data_structure["labels"].append(image_classes_dict[file])

        # build the KMeans model for the BOW of SIFT descriptors
        kp_set = np.concatenate(data_structure["sift_kp"], axis=0)
        Indexer.kmeans_model = KMeans(n_clusters=100, random_state=123).fit(kp_set)

        # build the attribute containing the features for each image
        # data_structure["features"] is a matrix whose rows correspond to images and columns to features
        data_structure["features"] = []
        for i in range(len(data_structure["im_names"])):
            tmp = list(data_structure["lbp"][i])
            tmp += list(data_structure["rgb_hist"][i])
            tmp += list(data_structure["ycbcr_hist"][i])
            tmp += list(data_structure["ycbcr_statistics"][i])
            tmp += list(data_structure["rgb_statistics"][i])
            tmp += list(FeatureExtractor.extract_bow(Indexer.kmeans_model, data_structure["sift_kp"][i]))
            tmp += list(data_structure["neural"][i])
            data_structure["features"].append(tmp)
        data_structure["features"] = np.array(data_structure["features"])

        # remove useless attributes
        data_structure.pop('lbp', None)
        data_structure.pop('rgb_hist', None)
        data_structure.pop('ycbcr_hist', None)
        data_structure.pop('neural', None)
        data_structure.pop('ycbcr_statistics', None)
        data_structure.pop('rgb_statistics', None)
        data_structure.pop('sift_kp', None)

        return data_structure
