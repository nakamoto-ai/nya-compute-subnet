# Running a Miner

## Overview

1. Miners complete the computation tasks received from the subnet.
2. Miners compete to *correctly* complete the tasks as quickly as possible. Responding with incorrect results will result in being penalized through blacklisting.
3. Miners are rewarded based on the proportion of the total computation they complete.
4. Correctness is determined by sending redundant compute to all miners. Correctness is objective, the subnet does not utilize YUMA consensus.
5. You are encouraged to customize the miner's code to improve performance and efficiency on your specific hardware; but avoid taking shortcuts. 


## Minimum Hardware Requirements

```
24gb GPU VRAM, tested on NVIDIA RTX A5000
```


## Running a Miner <a name="miner" />

### Docker Compose (Recommended)

1. Install [Docker](https://docs.docker.com/engine/install/).
2. (Optional) If you have an NVIDIA GPU, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
3. Build the Docker image; note that this step could take a while as it downloads the necessary dependencies and builds the image.
    ```bash
    docker compose build
    ```
4. Launching the miner:
   1. If you have an NVIDIA GPU, run the following command:
        ```bash
        docker compose up nya-miner-gpu
        ```
   2. If you do not have a GPU, run the following command:
       ```bash
       docker compose up nya-miner-cpu
       ```
5. Register the miner with the subnet:
    ```bash
    comx --testnet module register nya-miner [KEYNAME] --ip [IP] --port 9910 --netuid 23
    ```

### Virtual Environment

```bash
git clone https://github.com/nakamoto-ai/nya-compute-subnet.git
cd nya-compute-subnet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PORT=1914 
export KEY_NAME=nya-miner 
export DEVICE_MAP=auto 

python src/miner.py --port $PORT --device_map $DEVICE_MAP --keyfile $KEY_NAME


comx --testnet module register nya-miner [KEYNAME] --ip [IP] --port $PORT --netuid 23

```

### Troubleshooting

If your miner is not receiving any requests, ensure your miner is registered on the subnet. Furthermore, try the command below to ensure the API endpoint is accessible. 

```bash
curl -X POST MINER_IP:MINER_PORT/method/compute
```
This command sends a request to your miner, below is the expected output:
```angular2html
INFO:     MINER_IP:PORT - "POST /method/compute HTTP/1.1" 400 Bad Request
```
