FROM python:3.7

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir /screws_classification

WORKDIR /screws_classification

COPY requirements.txt requirements.txt

RUN pip install  -r requirements.txt

COPY . .

WORKDIR /screws_classification/scripts

CMD [ "python3", "train.py", "--evaluate", "--config_file", "../config_files/config1.yaml" ]
