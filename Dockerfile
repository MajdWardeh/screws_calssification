FROM python:3.7

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# RUN adduser --disabled-password majd
RUN useradd -ms /bin/bash majd
USER majd

RUN mkdir /home/majd/screws_classification

WORKDIR /home/majd/screws_classification

COPY requirements.txt requirements.txt

RUN pip install --user -r requirements.txt

COPY scripts scripts

RUN mkdir weights screws_set 

CMD [ "python", "scripts/Network.py" ]