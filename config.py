"""
Configuration file
"""

# INDEXING
INDEXING_DATA_STRUCTURE_PATH = "./objects/indexing.obj"
CSV_PATH = "./dataset.csv"
DATASET_PATH = "./train/"
INDEXING_KMEANS_PATH = "./objects/kmeans_bow.obj"

# SEGMENTER
SEGMENTER_BACKBONE = 'efficientnetb7'
SEGMENTER_WEIGHT_PATH = './models/efficientnetb7_dice_0.878.hdf5'
SEGMENTER_CONFIDENCE = 0.99
SEGMENTER_IM_SIZE = (256, 256, 3)

# FEATURE EXTRACTION
FE_WEIGHTS_PATH = "./models/efficientnetb7_FE.hdf5"
CLASS_NUMBER = 586

# CLASSIFIER
CLASSIFIER_IM_SIZE = (128, 128, 3)
CLASSIFIER_WEIGHT_PATH = "./models/efficientnetb7_classifier_0.966.hdf5"
CLASSIFIER_THRESHOLD = 0.5

# QUALITY CHECKER
GAUSSIAN_SIGMA = 3.04
DARK_THRESHOLD = 22
BLURRY_THRESHOLD = 51
QUALITYCHECKER_IMSIZE = (256, 256, 3)
QUALITYCHECKER_THRESHOLD = 0.5
QUALITYCHECKER_WEIGHT_PATH = "./models/quality_classifier.hdf5"
