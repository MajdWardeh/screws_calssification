#!/bin/bash

host_base="/home/majd/repos/screws_classification"
container_base="/home/majd/screws_classification"

host_weights=$host_base/weights
container_weights=$container_base/weights

host_screws_set=$host_base/weights
container_screws_set=$container_base/screws_set

sudo docker run -v $host_weights:$container_weights -v $host_screws_set:$container_screws_set -it myimage bash

