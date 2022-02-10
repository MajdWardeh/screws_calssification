import sys
import os
from pathlib import Path
import datetime
import random
import numpy as np
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50

from data_preparation import prepare_dataset_for_binary_classification, DataGenerator

class Classifier:
    def __init__(self, weights=None):
        self.target_image_size = (90, 335)
        self.weights_updated = False if weights is None else True
        self.model = self.__build_model()
        if not weights is None:
            self.__weights_check_and_load(weights)
        self.model.compile(
                    optimizer= Adam(), 
                    loss='binary_crossentropy', 
                    metrics=['binary_accuracy'],
                )
      
    def __build_model(self):
        input_shape = (self.target_image_size[0], self.target_image_size[1], 3)
        weights = 'imagenet' if not self.weights_updated else None
        resNet_model = ResNet50(include_top=False, weights=weights, input_shape=input_shape)
        resNet_model.trainable = False

        input_image = layers.Input(input_shape)
        x = resNet_model(input_image, training=False)
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(1000, activation='relu',  kernel_initializer='he_normal')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(500, activation='relu',  kernel_initializer='he_normal')(x)
        output = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[input_image], outputs=[output])

        return model
    
    def __createGenerators(self, val_size=0.2):
        ## set seeds for consistancy
        random.seed = 0
        np.random.seed = 0

        train_base_dir = '/home/majd/screws_classification/screws_set/train'
        class1_train_dir = os.path.join(train_base_dir, '1')
        class2_train_dir = os.path.join(train_base_dir, '2')

        test_base_dir = '/home/majd/screws_classification/screws_set/test'
        class1_test_dir = os.path.join(test_base_dir, '1')
        class2_test_dir = os.path.join(test_base_dir, '2')

        train_config = {
            'target_image_size': self.target_image_size,
            'random_horizontal_flip': 0.5,
            'random_vertical_flip': 0.5,
            'random_shift_prob': 0.3,
            'horizontal_shift_amount': 0.5/10.,
            'vertical_shift_amount': 0.5/10.,
            'random_zoom_prob': 1,
            'zoom_range': (0.95, 1.05),
            'random_brightness_shift': True,
            'brightness_shift_std': 0.15,
            'fill_mode': 'nearest',
            'resize_fill_mode': 'nearest',
        }
        val_config = {
            'target_image_size': self.target_image_size,
            'resize_fill_mode': 'nearest',
        }
        test_config = val_config

        train_batch_size = 32
        val_batch_size = 32
        test_batch_size = 32

        X_train, y_train, X_val, y_val, train_classes_ratio = prepare_dataset_for_binary_classification(class1_train_dir, class2_train_dir, split=True, test_size=val_size)
        X_test, y_test, test_classes_ratio = prepare_dataset_for_binary_classification(class1_test_dir, class2_test_dir, split=False)

        trainGen = DataGenerator(X_train, y_train, train_batch_size, train_config)
        valGen = DataGenerator(X_val, y_val, val_batch_size, val_config)
        testGen = DataGenerator(X_test, y_test, test_batch_size, test_config)

        return trainGen, valGen, testGen, train_classes_ratio

    def train(self, save_weights_dir='./weights'):
        Path(save_weights_dir).mkdir(parents=True, exist_ok=True)

        self.trainGen, self.valGen, self.testGen, self.train_class_ratio = self.__createGenerators()

        ## accounting for the implance in the number of samples of each class
        self.train_class_weights = {
            0: self.train_class_ratio[1],
            1: self.train_class_ratio[0]
        }

        history = self.model.fit(x=self.trainGen,
                            epochs=3000,
                            validation_data=self.valGen,
                            workers=10,
                            class_weight=self.train_class_weights
                            )
        model_name = 'model{}.h5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model.save_weights(os.path.join(save_weights_dir, model_name))
        return history

    def __weights_check_and_load(self, weights=None):
        if not weights is None:
            print('loading weights from: {}'.format(weights)) 
            self.model.load_weights(weights)
            self.weights_updated = True
        elif not self.weights_updated:
            raise Exception('trained weights are not loaded')

    def evaluate(self, weights=None):
        self.__weights_check_and_load(weights)

        self.trainGen, self.valGen, self.testGen, self.train_class_ratio = self.__createGenerators()

        print('evaluating...')

        train_history = self.model.evaluate(self.trainGen, verbose=0)
        val_history = self.model.evaluate(self.valGen, verbose=0)
        test_history = self.model.evaluate(self.testGen, verbose=0)

        print('accuracy on the train set: {:.4f}'.format(train_history[1]))
        print('accuracy on the validation set: {:.4f}'.format(val_history[1]))
        print('accuracy on the test set: {:.4f}'.format(test_history[1]))
    
    def classify(self, image_dir, weights=None):
        self.__weights_check_and_load(weights)
        image_names_list = os.listdir(image_dir) 
        image_paths_list = [os.path.join(image_dir, img) for img in image_names_list]
        config = {
            'target_image_size': self.target_image_size,
            'resize_fill_mode': 'nearest',
        }
        batch_size=32
        gen = DataGenerator(image_paths_list, None, batch_size, config)

        predicted_classes_dict = {}
        img_idx = 0
        for idx in range(gen.__len__()):
            x_batch = gen.__getitem__(idx)
            y_pred_batch = self.model(x_batch, training=False)
            for y_pred in y_pred_batch:
                pred_class = '1' if y_pred < 0.5 else '2'
                predicted_classes_dict[image_names_list[img_idx]] = pred_class
                img_idx += 1
        return predicted_classes_dict

def train():
    classifier = Classifier()
    classifier.train('/home/majd/screws_classification/weights')

def evaluate():
    weights = '/home/majd/screws_classification/weights/model1.h5'
    classifier = Classifier(weights) # give it the pre-trained weights so it won't download imagenet weights
    classifier.evaluate()

def main():
    # train()
    evaluate()

if __name__ == "__main__":
    main()