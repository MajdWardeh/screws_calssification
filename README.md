The code is in the scripts directory, it has three files:
1. data_preparation.py responsbile for data generation and augmentation.
2. classifier.py responsible for defining, training, and evalutating of the classifier.
3. inference.py this code classifies images in an INPUT_DIR and moves them to the classified class subdirectory in an OUT_DIR directory.

To test the code, a docker file is provided. First, you need to build the docker file:
```bash
# change directory this one
cd /to/this/dir

# build the docker image
sudo docker build -t majdimage .

# Evalutating accuracy:

# define some bash variables
host_weights="/home/majd/screws_classification/weights"
container_weights="/home/majd/screws_classification/weights"

host_screws_set="/home/majd/screws_classification/screws_set"
container_screws_set="/home/majd/screws_classification/screws_set"

# run classifier.py file, by runing the docker container:
sudo docker run -v $host_weights:$container_weights -v $host_screws_set:$container_screws_set majdimage 

# to do inferencing on a directory, you need to bind the INPUT_DIR and OUTPUT_DIR from the host to the container, then run the docker container in a virtual TTY bash terminal:

in_dir_host=/path/to/in/dir
in_dir_container=/home/majd/screws_classification/in_dir

out_dir_host=/path/to/out/dir
out_dir_container=/home/majd/screws_classification/out_dir

sudo docker run -v $host_weights:$container_weights -v $in_dir_host:$in_dir_container -v $out_dir_host:$out_dir_container -it majdimage bash

# run the inferencying
python3 inference.py --input_dir /home/majd/screws_classification/in_dir --output_dir /home/majd/screws_classification/out_dir

# you can also provide a dir for the weights, but first, you need to bind the weights dir from the host to the container.

weights_host=/path/to/host/weights
weights_container=/home/majd/screws_classification/weights_dir

sudo docker run -v $host_weights:$container_weights -v $in_dir_host:$in_dir_container -v $out_dir_host:$out_dir_container -v $weights_host:$weights_container -it majdimage bash

python3 inference.py --input_dir /home/majd/screws_classification/in_dir --output_dir /home/majd/screws_classification/out_dir --weights /home/majd/screws_classification/weights_dir


# you can also evaluate the accuracies from the TTY terminal by running:
python3 classifiy.py

```