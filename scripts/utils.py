import os
import numpy as np
import yaml

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from data_generator import DataGenerator

def prepare_dataset(classes_path_list, validation_ratio=0.2, random_state=0):
    '''
        @arg classes_path_list: a list of paths for each class.
        @arg validation_ratio: int, if it is not zero, the data will be split to train and validation sets
        @arg random_state: a random seed for shuffling the data
    ''' 
    image_classes_list = []
    for class_dir in classes_path_list:
        image_classes_list.append([os.path.join(class_dir, name) for name in os.listdir(class_dir)])

    X_train, y_train, X_val, y_val = [], [], [], []
    train_class_ratio = []
    if validation_ratio != 0:
        for i in range(len(image_classes_list)):
            Xi_train, Xi_val = train_test_split(image_classes_list[i], test_size=validation_ratio, random_state=random_state)
            X_train += Xi_train
            y_train += [i] * len(Xi_train)
            train_class_ratio.append(len(Xi_train))
            X_val += Xi_val
            y_val += [i] * len(Xi_val)
    else:
        for i, Xi_train in enumerate(image_classes_list):
            X_train += Xi_train
            y_train += [i] * len(Xi_train)
            train_class_ratio.append(len(Xi_train))

    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_val, y_val = shuffle(X_val, y_val, random_state=random_state)

    train_class_ratio = np.array(train_class_ratio)
    train_class_ratio = train_class_ratio / train_class_ratio.sum()

    return X_train, y_train, X_val, y_val, train_class_ratio

def load_config_file(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            loadedConfigs = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return loadedConfigs

