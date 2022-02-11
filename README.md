The code is in the scripts directory, it has three files:
1. classifier.py responsible for creating, training, and evaluating the screw classifier.
2. data_generator.py is responsible for data generation and augmentation.
3. inference.py this code classifies images in an INPUT_DIR and moves them to the classified class subdirectory in an OUT_DIR directory.
4. utils.py this file has helper functions for dataset_preparation and loading YAML files.

To test the code, a docker file is provided. First, you need to build the docker file.

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
python3 train.py --trian --config_file ../config_files/config1.yaml
```

# Inferencing:
```bash
# to perform inferencing on a directory, you need to bind the INPUT_DIR and OUTPUT_DIR from the host to the container, then run the docker container in a virtual TTY bash terminal:
in_dir_host=/path/to/in/dir
in_dir_container=/home/majd/screws_classification/in_dir

out_dir_host=/path/to/out/dir
out_dir_container=/home/majd/screws_classification/out_dir

sudo docker run -v $in_dir_host:$in_dir_container -v $out_dir_host:$out_dir_container -it majdimage bash

# run inferencing.py
python3 inference.py --input_dir /home/majd/screws_classification/in_dir --output_dir /home/majd/screws_classification/out_dir --config_file ../config_files/config1.yaml
```