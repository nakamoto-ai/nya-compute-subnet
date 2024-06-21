# Running a Miner

## Overview

1. Miners complete the computation tasks received from the subnet.
2. Miners compete to *correctly* complete the tasks as quickly as possible. Responding with incorrect results will
   result in being penalized through reduced task allocation and 0 rewards.
3. Miners are rewarded based on the proportion of the total computation they complete correctly.
4. Correctness is determined by sending redundant compute to all miners. Correctness is objective, rather than
   subjective.
5. You are encouraged to customize the miner's code to improve performance and efficiency on your specific hardware; but
   avoid taking shortcuts. If you have a GPU with >24gb VRAM, try increasing the batch size to improve performance.

## Minimum Hardware Requirements

```
24gb GPU VRAM, tested on NVIDIA RTX A5000, A40
```

## Factors to Consider

1. **Hardware**: The subnet is compute-intensive. Miners with more powerful hardware will be able to complete more tasks
   and earn more rewards.
2. **Uptime**: Miners must be online to receive tasks and submit results. Miners that are offline will not receive tasks
   and will not earn rewards.
3. **Optimization**: Miners are rewarded based on the proportion of the total computation they complete. If you can
   optimize your miner to complete tasks more quickly, you will earn more rewards. Avoid taking shortcuts that
   compromise correctness, we perform several checks to ensure the correctness of the results. You will not receive any
   tasks if the validator detects incorrect results.

## Running a Miner <a name="miner" />

### Docker Compose (Recommended)

1. Install [Docker](https://docs.docker.com/engine/install/).
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
3. Build the Docker image; note that this step could take a while as it downloads the necessary dependencies and builds
   the image.
    ```bash
    docker compose build
    ```
4. You need a communex key to receive rewards. If you don't have one, you can generate one using the command below:
    ```bash
    docker compose run nya-miner-gpu comx key create [KEY_NAME]
    ```
   Replace `[KEY_NAME]` with your desired key name.
5. Review the environment variables and ports set in the `compose.yaml` file. Ensure the key name is set to the key you
   generated in the previous step.
   ```bash
    environment:
      # Setting the environment variables used in run_miner.sh script
      - PORT=1920
      - DEVICE=auto
      - KEY_NAME=[KEY_NAME] # change this to your key name
    ports:
      - "1920:1920"
   ```
   Replace `[KEY_NAME]` with the key name you generated in step 4.
6. Launch the miner using the command below:
   ```bash
   docker compose up nya-miner-gpu
   ```
7. Register the miner with the subnet, replacing `[KEY_NAME]`, `[IP]`, and `[PORT]` with values set in the `compose.yaml`
   file.
    ```bash
    comx module register [MINER_NAME] [KEY_NAME] --ip [IP] --port [PORT] --netuid 8
    ```

### Virtual Environment

1. Install [Python 3.8+](https://www.python.org/downloads/).
2. Set the following environment variables:
    ```bash
    export PORT=1914 
    export KEY_NAME=[KEY_NAME] 
    export DEVICE_MAP=auto 
    ```
   Replace `[KEY_NAME]` with your desired key name.
3. Run the following commands to setup and run the miner:
   ```bash
   git clone https://github.com/nakamoto-ai/nya-compute-subnet.git
   cd nya-compute-subnet
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   
   python src/miner.py --port $PORT --device_map $DEVICE_MAP --keyfile $KEY_NAME
   ```

4. Register the miner with the subnet, replacing `[KEY_NAME]`, `[IP]`, and `[PORT]` with values set in the environment
   variables.
    ```bash
    comx module register [MINER_NAME] [KEY_NAME] --ip [IP] --port [PORT] --netuid 8
    ```

### Troubleshooting

If your miner is not receiving any requests, ensure your miner is registered on the subnet. Furthermore, try the command
below to ensure the API endpoint is accessible.

```bash
curl -X POST MINER_IP:MINER_PORT/method/compute

```
The expected output is:

```bash
{"error":{"code":400,"message":"Missing header: X-Key"}}
```

Your miner should also report receiving a bad request.

```bash
INFO:     MINER_IP:PORT - "POST /method/compute HTTP/1.1" 400 Bad Request
```
