import os
import datetime
import random
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50

from utils import prepare_dataset, load_config_file
from data_generator import DataGenerator


class Classifier:
    def __init__(self, config):
        self.__load_configurations(config)
        self.model = self.__build_model()
        self.model.compile(
            optimizer=Adam(),
            loss='binary_crossentropy',
            metrics=['binary_accuracy'],
        )
        if self.pretrained_weights_path is not None:
            self.model.load_weights(self.pretrained_weights_path)
            print('pretrained weights were loaded')

    def __load_configurations(self, config_path):
        config = load_config_file(config_path)
        self.target_image_size = config['target_image_size']
        self.label_map = config['label_map']

        self.train_data_path = config['train_data_path']
        self.test_data_path = config['test_data_path']
        self.val_ratio = config['val_ratio']

        self.train_generator_config = config['train_generator_config']
        self.val_generator_config = config['val_generator_config']
        self.test_generator_config = config['test_generator_config']

        self.train_epochs = config['train_epochs']
        self.save_weights_dir = config['save_weights_dir']
        self.evaluate_after_training = config['evaluate_after_training']

        self.pretrained_weights_path = config.get('pretrained_weights_path', None)
        self.classification_threshold = config['classification_threshold']

    def __build_model(self):
        input_shape = (self.target_image_size[0], self.target_image_size[1], 3)

        # avoid downloading imagenet's weights if weights were provided
        weights = 'imagenet' if self.pretrained_weights_path is not None else None

        resNet_model = ResNet50(
            include_top=False, weights=weights, input_shape=input_shape)
        resNet_model.trainable = False

        input_image = layers.Input(input_shape)
        x = resNet_model(input_image, training=False)
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(1000, activation='relu',
                         kernel_initializer='he_normal')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(500, activation='relu',
                         kernel_initializer='he_normal')(x)
        output = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[input_image], outputs=[output])

        return model

    def __create_generators(self):
        # set seeds for consistancy
        random.seed = 0
        np.random.seed = 0

        train_imgaes_path = [os.path.join(
            self.train_data_path, label) for label in self.label_map]
        test_images_path = [os.path.join(
            self.test_data_path, label) for label in self.label_map]

        X_train, y_train, X_val, y_val, train_classes_ratio = prepare_dataset(
            train_imgaes_path, validation_ratio=self.val_ratio)
        X_test, y_test, _, _, test_classes_ratio = prepare_dataset(
            test_images_path, validation_ratio=0)

        train_gen = DataGenerator(X_train, y_train, self.train_generator_config)
        val_gen = DataGenerator(X_val, y_val, self.val_generator_config)
        test_gen = DataGenerator(X_test, y_test, self.test_generator_config)

        return train_gen, val_gen, test_gen, train_classes_ratio

    def train(self, save_weights_dir=None):
        # check if save_weights_dir was provided, if not use the default one (the one in the config file)
        if save_weights_dir is None:
            save_weights_dir = self.save_weights_dir

        assert os.path.exists(save_weights_dir), 'the directory for saving weights does not exit'

        train_gen, val_gen, test_gen, train_class_ratio = self.__create_generators()

        # accounting for the implance in the number of samples of each class
        train_class_weights = {
            0: train_class_ratio[1],
            1: train_class_ratio[0]
        }

        history = self.model.fit(x=train_gen,
                                 epochs=self.train_epochs,
                                 validation_data=val_gen,
                                 workers=10,
                                 class_weight=train_class_weights
                                 )
        model_name = 'model{}.h5'.format(
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model.save_weights(os.path.join(save_weights_dir, model_name))

        if self.evaluate_after_training:
            self.evaluate()

        return history

    def evaluate(self, weights=None):
        if weights is not None:
            self.model.load_weights(weights)

        train_gen, val_gen, test_gen, train_class_ratio = self.__create_generators()

        print('evaluating...')

        train_history = self.model.evaluate(train_gen, verbose=0)
        val_history = self.model.evaluate(val_gen, verbose=0)
        test_history = self.model.evaluate(test_gen, verbose=0)

        print('accuracy on the train set: {:.4f}'.format(train_history[1]))
        print('accuracy on the validation set: {:.4f}'.format(val_history[1]))
        print('accuracy on the test set: {:.4f}'.format(test_history[1]))

    def classify(self, images_dir, weights=None):
        if weights is not None:
            self.model.load_weights(weights)

        image_names_list = os.listdir(images_dir)
        image_paths_list = [os.path.join(images_dir, img)
                            for img in image_names_list]
        config = {
            'target_image_size': self.target_image_size,
            'resize_fill_mode': 'nearest',
        }
        batch_size = 32
        gen = DataGenerator(image_paths_list, None, batch_size, config)

        predicted_classes_dict = {}
        img_idx = 0
        for idx in range(gen.__len__()):
            x_batch = gen.__getitem__(idx)
            y_pred_batch = self.model(x_batch, training=False)
            for y_pred in y_pred_batch:
                y_pred_thresholded = int(y_pred < self.classification_threshold)
                pred_class = self.label_map[y_pred_thresholded]
                predicted_classes_dict[image_names_list[img_idx]] = pred_class
                img_idx += 1
        return predicted_classes_dict
