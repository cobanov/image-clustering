FROM nvcr.io/nvidia/pytorch:21.08-py3

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade -r requirements.txt
