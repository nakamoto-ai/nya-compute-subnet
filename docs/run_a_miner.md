# Running a Miner

## Overview

1. Miners complete the computation tasks received from the subnet.
2. Miners compete to *correctly* complete the tasks as quickly as possible. Responding with incorrect results will result in being penalized through blacklisting.
3. Miners are rewarded based on the proportion of the total computation they complete.
4. Correctness is determined by sending redundant compute to all miners. Correctness is objective, the subnet does not utilize YUMA consensus.
5. You are encouraged to customize the miner's code to improve performance and efficiency on your specific hardware; but avoid taking shortcuts. 

## Running a Miner <a name="miner" />

### Docker (Recommended)

1. Install [Docker](https://docs.docker.com/engine/install/).
2. (Optional) If you have an NVIDIA GPU, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
3. Build the Docker image:
```bash
docker build -t nya-container -f Dockerfile .
```
4. Run the Docker container:
```bash
docker run -p 8000:8000 -p 9910:9910 -it -v $(pwd)/src:/app/src -v $(pwd)/.cache:/root/.cache -v ${HOME}/.commune:/root/.commune nya-container bash
```
5. Inside the container, run the miner:
```bash
python src/miner.py --name nya --keyfile [KEYNAME] 
```
6. Register the miner with the subnet:
```bash
comx --testnet module register nya-miner [KEYNAME] --ip [IP] --port 9910 --netuid 23
```
