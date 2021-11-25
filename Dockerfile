FROM python:3.8

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update \
  && pip install -U pip \
  && pip install -r requirements.txt

ENTRYPOINT [ "streamlit", "run", "app.py" ]
