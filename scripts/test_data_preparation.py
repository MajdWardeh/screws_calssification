import os
import numpy as np

from data_generator import DataGenerator
from utils import prepare_dataset, load_config_file

def main():
    base_dir = '/home/majd/screws_classification/screws_set/train'
    class1_dir = os.path.join(base_dir, '1')
    class2_dir = os.path.join(base_dir, '2')

    X_train, y_train, _, _, train_sample_weight = prepare_dataset([class1_dir, class2_dir], validation_ratio=0)

    target_size = (90, 335)
    brightness_std = 0.12
    config = {
        'target_image_size': target_size,
        'batch_size': 1,
        'resize_fill_mode': 'nearest',
        'random_horizontal_flip': 1.,
        'random_vertical_flip': 1.,
        'random_shift_prob': 1,
        'horizontal_shift_amount': 0.05,
        'vertical_shift_amount': 0.05,
        'random_zoom_prob': 1,
        'zoom_range': (0.95, 1.05),
        'random_brightness_shift': True,
        'brightness_shift_std': 0.12,
        'fill_mode': 'nearest',
    }
    dataGen = DataGenerator(X_train, y_train, config=config)
    for i in range(dataGen.__len__()):
       x_batch, y_batch = dataGen.__getitem__(i)
       assert x_batch[0].shape == (target_size[0], target_size[1], 3)
       assert x_batch[0].dtype == np.float32
       assert y_batch[0].dtype == np.float32
       brightness_max_value =  x_batch[0].max()
       assert brightness_max_value <= 1 and brightness_max_value >= (1-5*brightness_std), 'assertion on brightness_max_value={} has failed'.format(brightness_max_value)
    print('tests passed!')

if __name__=='__main__':
    main()