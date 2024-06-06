FROM nvcr.io/nvidia/pytorch:23.12-py3
#22.12
#huggingface/transformers-pytorch-gpu:4.35.2
#docker pull huggingface/transformers-pytorch-gpu:4.35.2


COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt


WORKDIR /app

