import datetime
import logging
import os
import random

import pickle

import torch
from substrateinterface import Keypair  # type: ignore

import src.utils as utils
from src.workload import Workload
from src.validator import ComputeValidator

# from reward import Reward
time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info(f"Running {__file__}")


def get_dummy_workers_info(self, nu):
    modules_addresses = self.get_addresses(self.client, self.net_uid)
    modules_keys = self.client.query_map_key(self.net_uid)

    # TODO: set this to True once initial testing is complete
    check_validator_is_registered = False
    if check_validator_is_registered and self.key.ss58_address not in modules_keys.values():
        logger.error(f"Validator key {self.key.ss58_address} is not registered in subnet, aborting ...")
        return

    known_ip_list = [
        "69.30.85.204",  # mo-gpu
        "69.30.85.150",  # mo-gpu-2
        "69.30.85.81",  # naga-gpu
        # "194.163.148.58",  # mo-cpu
        # "209.145.51.226",  # Franz
        # "207.244.247.121",  # kyril
        # "85.190.242.134",  # naga
        # "194.163.150.107",  # bobby
        # "89.187.159.49",
        # "104.143.3.153",
    ]

    worker_info_dict: dict[str, dict[str, any]] = {}
    modules_filtered_address = utils.get_ip_port(modules_addresses)
    for module_id in modules_keys.keys():
        module_addr = modules_filtered_address.get(module_id, None)
        if not module_addr:
            continue
        if module_addr[0] not in known_ip_list:
            continue
        module_dict = {"id": module_id,
                       "ip": module_addr[0],
                       "port": module_addr[1],
                       "key": modules_keys[module_id],
                       }

        worker_info_dict[modules_keys[module_id]] = module_dict

    return worker_info_dict


def get_miner_info():
    # create worker_info_dict with dummy data till we have the actual data
    worker_info_dict = {random.randint(-500, -100): "info" for _ in range(11)}

    workload_dummy_results = {workload: torch.rand((10, 10))
                              for workload in workload_set
                              }

    worker_results_dict = {
        worker_id: {
            "results": torch.stack([workload_dummy_results[workload_id] for workload_id in
                                    worker_info_to_workload_info[
                                        worker_id]]) if random.random() > 0.2 else torch.ones((64, 10, 10)),
            "checksum": "checksum",
            "handshake": "handshake"}
        for worker_id in worker_info_dict.keys()
    }

    worker_results_dict[worker_id] = {
        "results": torch.rand((64, 10, 10)),
        "checksum": "checksum",
        "handshake": "handshake"
    }


# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)  # add assertion here

def test_workload():
    processed_id_list = []
    workload = Workload(max_token=1024, processed_id_list=processed_id_list)

    task_list = workload.get_workload(processed_id_list, 10)

    assert len(task_list) == 10

    for task in task_list:
        assert task["token_count"] <= 1024
        assert "text" in task
        assert "id" in task

    processed_id_list = [task["id"] for task in task_list]

    task_list = workload.get_workload(processed_id_list, 10)

    assert len(task_list) == 10

    for task in task_list:
        assert task["token_count"] <= 1024
        assert "text" in task
        assert "id" in task
        assert task["id"] not in processed_id_list


def load_data():
    dir = "data"
    input_dict = {}
    for file in os.listdir(dir):
        with open(os.path.join(dir, file), "rb") as f:
            input_dict[file] = pickle.load(f)

    input_dict = dict(sorted(input_dict.items(), key=lambda x: x[0]))

    return input_dict


def test_workload_in_validator():
    input_data_dict = load_data()

    compute_validator = ComputeValidator(key=None,
                                         net_uid=None,
                                         client=None)

    for key, value in input_data_dict.items():
        print(f"Processing {key}")
        compute_validator.validate_workloads(**value)


def main():
    test_workload_in_validator()
    test_workload()


if __name__ == '__main__':
    # unittest.main()
    main()
