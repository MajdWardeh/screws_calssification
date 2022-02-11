The code in the scripts directory has four main files:
1. classifier.py is responsible for creating, training, and evaluating the screw classifier.
2. data_generator.py is responsible for data generation and augmentation.
3. inference.py this code classifies images in an INPUT_DIR and moves them to the classified class subdirectory in an OUT_DIR directory.
4. utils.py this file has helper functions for dataset_preparation and loading YAML files.

For detailed explanation regarding the data preprocessing and augmentation, the network architecture, and the achieved accuracies, please read the provided report.

To test the code, a docker file is provided. First, you need to build it.

# Building the docker image
```bash
# change directory this one
cd /to/this/dir

# build the docker image
sudo docker build -t majdimage .
```

# Evaluating accuracy:
```bash
# run the docker container:
sudo docker run majdimage 
```

# Training the classifier:
```bash
# run the container with a virtual TTY bash terminal:
sudo docker run -it majdimage bash

# to train the classifier, you need to provide a config file
python3 train.py --train --config_file ../config_files/config1.yaml
```

# Inferencing:
```bash
# to perform inferencing on a directory, you need to bind the INPUT_DIR and OUTPUT_DIR from the host to the container, then run the docker container in a virtual TTY bash terminal:
in_dir_host=/path/to/in/dir
in_dir_container=/screws_classification/in_dir

out_dir_host=/path/to/out/dir
out_dir_container=/screws_classification/out_dir

sudo docker run -v $in_dir_host:$in_dir_container -v $out_dir_host:$out_dir_container -it majdimage bash

# run inferencing.py
python3 inference.py --input_dir /screws_classification/in_dir --output_dir /screws_classification/out_dir --config_file ../config_files/config1.yaml
```
