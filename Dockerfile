FROM ubuntu:16.04

MAINTAINER Sara Alizadeh and Shayan Liaghat "sara72alizadeh@gmail.com" "shayanliaghat@gmail.com"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev && \
    pip install --upgrade pip

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD [ "application.py" ]