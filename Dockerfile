FROM python:3.10.14-slim-bullseye
RUN apt update -y
WORKDIR /app

COPY . /app
RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]