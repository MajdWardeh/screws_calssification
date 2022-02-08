import sys
import os
import random
import numpy as np
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50

from data_preparation import prepare_dataset_for_binary_classification, DataGenerator
      
def build_model(input_shape):

    resNet_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
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

        
def createGenerators(target_image_size, val_size=0.2):
    ## set seeds for consistancy
    random.seed = 0
    np.random.seed = 0

    train_base_dir = './screws_set/train'
    class1_train_dir = os.path.join(train_base_dir, '1')
    class2_train_dir = os.path.join(train_base_dir, '2')

    test_base_dir = './screws_set/test'
    class1_test_dir = os.path.join(test_base_dir, '1')
    class2_test_dir = os.path.join(test_base_dir, '2')

    train_config = {
        'target_image_size': target_image_size,
        'random_horizontal_flip': 0.3,
        'random_vertical_flip': 0.3,
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
        'target_image_size': target_image_size,
        'resize_fill_mode': 'nearest',
    }
    test_config = val_config

    train_batch_size = 64
    val_batch_size = 64
    test_batch_size = 64

    X_train, y_train, X_val, y_val, train_class_ratio = prepare_dataset_for_binary_classification(class1_train_dir, class2_train_dir, split=True, test_size=val_size)
    X_test, y_test, test_class_ratio = prepare_dataset_for_binary_classification(class1_test_dir, class2_test_dir, split=False)

    trainGen = DataGenerator(X_train, y_train, train_batch_size, train_config)
    valGen = DataGenerator(X_val, y_val, val_batch_size, val_config)
    testGen = DataGenerator(X_test, y_test, test_batch_size, test_config)

    return trainGen, valGen, testGen, train_class_ratio


def main(train):
    target_image_size = (90, 335)
    input_shape = (target_image_size[0], target_image_size[1], 3)
    model = build_model(input_shape)
    
    trainGen, valGen, testGen, train_class_ratio = createGenerators(target_image_size)

    train_class_weights = {
        0: train_class_ratio[1],
        1: train_class_ratio[0]
    }

    model.compile(
                optimizer= Adam(), 
                loss='binary_crossentropy', 
                metrics=['binary_accuracy'],
            )

    if train:

        history = model.fit(x=trainGen,
                            epochs=3000,
                            validation_data=testGen,
                            workers=10,
                            class_weight=train_class_weights
                            )
        model.save_weights('./weights/first_model.h5')
        model.evaluate(testGen)

    else:
        model.load_weights('./weights/model1.h5')

        train_history = model.evaluate(trainGen, verbose=0)
        val_history = model.evaluate(valGen, verbose=0)
        test_history = model.evaluate(testGen, verbose=0)

        print('accuracy on the train set: {:.4f}'.format(train_history[1]))
        print('accuracy on the validation set: {:.4f}'.format(val_history[1]))
        print('accuracy on the test set: {:.4f}'.format(test_history[1]))

if __name__ == "__main__":
    main(train=False)