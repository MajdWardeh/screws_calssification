FROM python:3.7

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# COPY ./scripts/Network.py .
# COPY ./scripts/data_generator.py .
# COPY file1 .
CMD ["pwd"]
# CMD [ "python", "./Network.py" ]