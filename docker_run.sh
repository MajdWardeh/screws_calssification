#!/bin/bash

cd /home/majd/repos/screws_classification

container_pwd="/home/majd/screws_classification"
host_weights_dir=$(pwd)/weights

sudo docker run -v $(pwd)/weights:$container_pwd/weights -v $(pwd)/screws_set:$container_pwd/screws_set -it myimage bash

