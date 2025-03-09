FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy opencv-python

ENV DISPLAY=:99

CMD ["python", "Color_Detection.py"]
