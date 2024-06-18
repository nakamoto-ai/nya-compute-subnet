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

## Factors to Consider

1. **Hardware**: The subnet is compute-intensive. Miners with more powerful hardware will be able to complete more tasks and earn more rewards.
2. **Uptime**: Miners must be online to receive tasks and submit results. Miners that are offline will not receive tasks and will not earn rewards.
3. **Optimization**: Miners are rewarded based on the proportion of the total computation they complete. If you can optimize your miner to complete tasks more quickly, you will earn more rewards. Avoid taking shortcuts that compromise correctness, we perform several checks to ensure the correctness of the results. You will not receive any tasks if the validator detects incorrect results.


## Running a Miner <a name="miner" />

### Docker Compose (Recommended)

1. Install [Docker](https://docs.docker.com/engine/install/).
2. (Optional) If you have an NVIDIA GPU, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
3. Build the Docker image; note that this step could take a while as it downloads the necessary dependencies and builds the image.
    ```bash
    docker compose build
    ```
4. Review the environment variables and ports set in the `compose.yaml` file. Once everything is in order, launch the miner using the command below:
   ```bash
   docker compose up nya-miner-gpu
   ```

5. Register the miner with the subnet, replacing `[KEYNAME]`, `[IP]`, and `[PORT]` with values set in the `compose.yaml` file.
    ```bash
    comx module register nya-miner [KEYNAME] --ip [IP] --port [PORT] --netuid 8
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

comx module register nya-miner $KEY_NAME --ip [IP] --port $PORT --netuid 8

```

### Troubleshooting

If your miner is not receiving any requests, ensure your miner is registered on the subnet. Furthermore, try the command below to ensure the API endpoint is accessible. 

```bash
curl -X POST MINER_IP:MINER_PORT/method/compute
```
This command sends a request to your miner, below is the expected output:
```bash
INFO:     MINER_IP:PORT - "POST /method/compute HTTP/1.1" 400 Bad Request
```
