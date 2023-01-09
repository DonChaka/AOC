from os.path import join

from sklearn.svm import SVC


HOME_PATH_TRAIN = join('Images', 'RoadImages', 'train')
HOME_PATH_TEST = join('Images', 'RoadImages', 'test')

COLORED_DATASET_PATHS = {
    'red': join('Images', 'polishDatasetColorCoded', 'Red'),
    'blue': join('Images', 'polishDatasetColorCoded', 'Blue'),
    'yellow': join('Images', 'polishDatasetColorCoded', 'Yellow')
}

SHAPED_DATASET_PATHS = {
    'circle': join('Images', 'polishDatasetShapeCoded', 'Circle'),
    'square': join('Images', 'polishDatasetShapeCoded', 'Square'),
    'triangle': join('Images', 'polishDatasetShapeCoded', 'Triangle')
}

OUTPUT_SOURCE_IMAGE_SHAPE = (900, 1200, 3)
OUTPUT_SLICE_IMAGE_SHAPE = (40, 40, 3)

HOG_PARAMS = {
    'orientation': 8,
    'pixels_per_cell': (5, 5),
    'cells_per_block': (8, 8)
}

SIGN_RECTANGLE_THICKNESS = 2
SIGN_RECTANGLE_OFFSET = 5
SIGN_MIN_SIZE = 10
SIGN_MATCH_DISTANCE_EPSILON = 0.05

CLASSIFIER_MODEL_CLASS = SVC
CLASSIFIER_SEARCH_BEST_MODEL = False
CLASSIFIER_SEARCH_BEST_MODEL_PARAMS = {
    'kernel': ["rbf", 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [3, 6]
}
CLASSIFIER_PARAMS = {
    'degree': 3,
    'gamma': 'scale',
    'kernel': 'poly'
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'criterion': 'entropy',
    'max_depth': 100,
    'min_samples_split': 2
}

FONT_SIZE = 1
FONT_THICKNESS = 2
