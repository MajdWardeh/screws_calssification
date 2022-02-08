FROM python:3.7

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --user --no-cache-dir -r requirements.txt

COPY Network.py .
COPY data_generator.py .

CMD [ "python", "./Network.py" ]