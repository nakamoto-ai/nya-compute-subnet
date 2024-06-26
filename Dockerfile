FROM nvcr.io/nvidia/pytorch:23.12-py3
#22.12
#huggingface/transformers-pytorch-gpu:4.35.2
#docker pull huggingface/transformers-pytorch-gpu:4.35.2

RUN apt-get update && \
    apt-get install -y tmux npm curl && \
    npm install pm2@latest -g

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

RUN chmod +x /app/scripts/entrypoint.sh

# update PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app:/app/src"

WORKDIR /app

