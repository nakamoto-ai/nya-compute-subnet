import argparse
import asyncio
import base64
import datetime
import logging
import math
import os
import pickle
import random
import time
import traceback
from collections import Counter

import numpy as np
import pandas as pd
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from communex._common import get_node_url  # pylint: disable=W0611
from communex.client import CommuneClient
from communex.compat.key import classic_load_key
from communex.module.client import ModuleClient
from communex.module.module import Module
from substrateinterface import Keypair  # type: ignore

import utils
from workload import Workload

# from reward import Reward
time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

logging.basicConfig(filename=f"/logs/nya_validator_{time_str}.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

logger.info(f"Running {__file__}")


# counter = 0


# def sign_request(data, private_key_path="keys/private.pem"):
#     with open(private_key_path, 'r') as key_file:
#         private_key = RSA.import_key(key_file.read())
#     data_str = data
#     if not isinstance(data_str, str):
#         data_str = str(data)
#
#     hash_value = SHA256.new(data_str.encode('utf-8'))
#     signature = pkcs1_15.new(private_key).sign(hash_value)
#     return base64.b64encode(signature).decode('utf-8')


class ComputeValidator(Module):
    out_dir: str = "out"
    validated_out_dir: str = "validated"
    worker_completed_workload: dict[int, int] = {}

    # submit 2 weights per hour
    weight_submit_interval = 1800

    worker_time_per_workload: dict[int, float] = {}

    def __init__(
            self,
            key: Keypair,
            net_uid: int,
            client: CommuneClient,
            interval: int = 30,
            min_workload_per_worker: int = 16,
            all_workers_redundancy_ratio: float = 0.5,
            three_workers_redundancy_ratio: float = 0.5,
            call_timeout: int = 300,
            fine_web_max_token: int = 1024,
            use_dynamic_interval: bool = True,
    ) -> None:
        super().__init__()
        logger.info("Initializing Compute Validator ... ")

        self.client = client
        self.key = key
        self.net_uid = net_uid
        self.interval = interval
        self.use_dynamic_interval = use_dynamic_interval
        self.call_timeout = call_timeout

        self.min_workload_per_worker = min_workload_per_worker
        self.all_workers_redundancy_ratio = all_workers_redundancy_ratio
        self.three_workers_redundancy_ratio = three_workers_redundancy_ratio

        self.processed_id_list = []
        self.processed_id_arrow_path = os.path.join(self.out_dir, "processed_id.arrow")
        if os.path.exists(self.processed_id_arrow_path):
            self.processed_id_list = pd.read_parquet(self.processed_id_arrow_path)["id"].tolist()

        self.workload = Workload(max_token=fine_web_max_token, processed_id_list=self.processed_id_list)

        self.fine_web_max_token = fine_web_max_token

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.last_validation_time = time.time()

    @staticmethod
    def get_addresses(client: CommuneClient, netuid: int) -> dict[int, str]:
        """
        Retrieve all module addresses from the subnet.

        Args:
            client: The CommuneClient instance used to query the subnet.
            netuid: The unique identifier of the subnet.

        Returns:
            A dictionary mapping module IDs to their addresses.
        """
        module_addresses = client.query_map_address(netuid)
        return module_addresses

    def get_workers_info(self):

        modules_addresses = self.get_addresses(self.client, self.net_uid)
        modules_keys = self.client.query_map_key(self.net_uid)

        # TODO: set this to True once initial testing is complete
        check_validator_is_registered = True
        if check_validator_is_registered and self.key.ss58_address not in modules_keys.values():
            logger.error(f"Validator key {self.key.ss58_address} is not registered in subnet, aborting.")
            return {}

        worker_info_dict: dict[str, dict[str, any]] = {}
        modules_filtered_address = utils.get_ip_port(modules_addresses)
        for module_id in modules_keys.keys():
            module_addr = modules_filtered_address.get(module_id, None)
            if not module_addr:
                continue
            # if module_addr[0] not in known_ip_list:
            #     continue
            module_dict = {"id": module_id,
                           "ip": module_addr[0],
                           "port": module_addr[1],
                           "key": modules_keys[module_id],
                           }

            worker_info_dict[modules_keys[module_id]] = module_dict

        return worker_info_dict

    def get_workload(self, worker_info_dict):
        worker_workload_dict = {worker_id: [] for worker_id in worker_info_dict.keys()}
        worker_count = len(worker_info_dict)

        worker_capacity_dict = {}
        time_out_by_half = self.call_timeout * 0.4
        for worker_id in worker_info_dict.keys():
            if worker_id not in self.worker_time_per_workload:
                worker_capacity_dict[worker_id] = self.min_workload_per_worker
            else:
                worker_capacity = time_out_by_half / self.worker_time_per_workload[worker_id]
                worker_capacity_dict[worker_id] = math.floor(worker_capacity)

        for worker_id, capacity in worker_capacity_dict.items():
            # if capacity > 2 * self.min_workload_per_worker:
            #     worker_capacity_dict[worker_id] = 2 * self.min_workload_per_worker
            if capacity < self.min_workload_per_worker:
                worker_capacity_dict[worker_id] = self.min_workload_per_worker

        # logger.info(f"Worker capacity:\n{worker_capacity_dict}")
        median_capacity = np.median(list(worker_capacity_dict.values()))

        all_workers_redundancy_count = math.floor(median_capacity * self.all_workers_redundancy_ratio)
        three_workers_redundancy_count = math.floor(median_capacity * self.three_workers_redundancy_ratio)

        redundant_count = all_workers_redundancy_count + three_workers_redundancy_count

        remaining_capacity_dict = {worker_id: max(worker_capacity_dict[worker_id] - redundant_count, 0)
                                   for worker_id in worker_capacity_dict.keys()}

        remaining_workload_count = sum(remaining_capacity_dict.values())

        # remaining_workload_count = (self.min_workload_per_worker - all_workers_redundancy_count -
        #                             three_workers_redundancy_count) * worker_count

        # assert remaining_workload_count >= 0, "Remaining workload should be greater than 0"

        total_workload_count = (remaining_workload_count + all_workers_redundancy_count +
                                three_workers_redundancy_count * math.ceil(worker_count / 3))

        total_workload_count = int(total_workload_count * 1.5)  # increase the workload by 50% in case we need more

        logger.info(f"Gathering {total_workload_count} workloads ... ")

        task_workload = self.workload.get_workload(self.processed_id_list, total_workload_count)


        logger.info(f"Assigning workloads ... ")
        # shuffle the workload
        random.shuffle(task_workload)

        remaining_workload = task_workload

        all_workers_shared_workload = remaining_workload[:all_workers_redundancy_count]
        remaining_workload = remaining_workload[all_workers_redundancy_count:]

        all_worker_redundancy_workload_ids = [workload["id"] for workload in all_workers_shared_workload]

        # allocate the shared workload for all workers
        for worker_id in worker_info_dict.keys():
            worker_workload = [workload.copy() for workload in all_workers_shared_workload]
            worker_workload_dict[worker_id].extend(worker_workload)

        three_way_redundancy_workload_ids = []

        if worker_count >= 3:
            three_way_redundancy_workload = remaining_workload[
                                            :three_workers_redundancy_count * math.ceil(worker_count / 3)]
            remaining_workload = remaining_workload[three_workers_redundancy_count * math.ceil(worker_count / 3):]

            for workload in three_way_redundancy_workload:
                workers_with_available_capacity = [worker_id for worker_id, workload_list in
                                                   worker_workload_dict.items() if
                                                   len(workload_list) < worker_capacity_dict[worker_id]
                                                   ]

                # if less than 3 workers are available, break
                if len(workers_with_available_capacity) < 3:
                    break

                # if more than 3 workers have less than 3 workload, then randomly select 3 workers
                if len(workers_with_available_capacity) > 3:
                    workers_with_available_capacity = random.sample(workers_with_available_capacity, 3)

                three_way_redundancy_workload_ids.append(workload["id"])

                for worker_id in workers_with_available_capacity:
                    worker_workload_dict[worker_id].append(workload.copy())

        while any([worker_capacity_dict[worker_id] - len(l) > 0 for worker_id, l in
                   worker_workload_dict.items()]) and len(
            remaining_workload) > 0:
            workload = remaining_workload.pop()
            for worker_id in worker_info_dict.keys():
                if len(worker_workload_dict[worker_id]) < worker_capacity_dict[worker_id]:
                    worker_workload_dict[worker_id].append(workload.copy())
                    break

        # shuffle each worker's workload
        for worker_id in worker_info_dict.keys():
            workload = worker_workload_dict[worker_id]
            random.shuffle(workload)
            worker_workload_dict[worker_id] = workload

        # record and remove index from all the workloads, we will use this to track the processed workload
        worker_info_to_workload_info = {}
        for worker_id, workload in worker_workload_dict.items():
            workload_ids = [work["id"] for work in workload]
            worker_info_to_workload_info[worker_id] = workload_ids
            workload = list(map(lambda x: x["text"], workload))
            worker_workload_dict[worker_id] = workload
            # for work in workload:
            #     if "index" in work:
            #         work.pop("index")

        workload_redundancy_dict = {"all": all_worker_redundancy_workload_ids,
                                    "three": three_way_redundancy_workload_ids}

        return worker_workload_dict, worker_info_to_workload_info, worker_capacity_dict, workload_redundancy_dict

    async def get_worker_result(
            self,
            worker_info_dict: dict[str, any],
            workload,
    ) -> dict[str, any]:
        """
        Prompt a miner module to generate an answer to the given question.

        Args:
            worker_info_dict: The IP address and key of the miner module.
            workload: The workload to be processed by the miner module.

        Returns:
            The generated answer from the miner module, or None if the miner fails to generate an answer.
        """
        worker_key = worker_info_dict["key"]
        module_ip, module_port = worker_info_dict["ip"], worker_info_dict["port"]
        client = ModuleClient(module_ip, int(module_port), self.key)

        # signature = sign_request(workload)

        params = {
            "task": workload,
        }

        failed = False
        reason = "Successful"
        size_estimate = len(pickle.dumps(params))
        worker_result = {}

        current_time = time.perf_counter()
        try:
            worker_result = await client.call("compute",
                                              worker_key,
                                              params,
                                              timeout=self.call_timeout,
                                              )
        except Exception as e:
            # logger.error(f"Error! address: {module_ip}:{module_port}, error: {e}")
            failed = True

            reason = str(type(e).__name__)

        print(f"{module_ip} - {type(worker_result)}")
        if isinstance(worker_result, list):
            worker_result = {}

        elapsed_time = time.perf_counter() - current_time
        worker_result_size = len(pickle.dumps(worker_result))
        worker_result["server_elapsed_time"] = elapsed_time
        worker_result["workload_size_mb"] = size_estimate / 10 ** 6
        worker_result["result_size_mb"] = worker_result_size / 10 ** 6
        if failed:
            worker_result["failure_reason"] = reason

        logger.debug(f"Worker summary. status: {'failed' if failed else 'success'},"
                     f"ip: {module_ip}, sent_size: {size_estimate / 10 ** 6:.2f} MB, "
                     f"received_size: {worker_result_size / 10 ** 6:.2f} MB, "
                     f"elapsed_time: {elapsed_time:.2f} seconds")

        return worker_result

    async def process_workload(self, worker_id, worker_info_dict, workload):
        logger.debug(
            f"Sending workload to worker {worker_id[:10]} at {worker_info_dict['ip']}:{worker_info_dict['port']}")
        worker_result = await self.get_worker_result(worker_info_dict, workload)
        return worker_id, worker_result

    async def gather_results(self, worker_info_to_workload_dict, workers_info_dict):

        process_workload_task_list = [
            self.process_workload(worker_id, workers_info_dict[worker_id], workload)
            for worker_id, workload in worker_info_to_workload_dict.items()
        ]
        results = await asyncio.gather(*process_workload_task_list)
        worker_info_to_result_dict = {worker_id: result for worker_id, result in results}

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future_to_worker_id = {
        #         executor.submit(self.process_workload, worker_id, worker_info_dict[worker_id], workload): worker_id
        #         for worker_id, workload in worker_info_to_workload_dict.items()
        #     }
        #
        #     for future in concurrent.futures.as_completed(future_to_worker_id):
        #         worker_id, result = future.result()
        #         worker_info_to_result_dict[worker_id] = result

        return worker_info_to_result_dict

    def run_gather_results(self, worker_info_to_workload_dict, workers_info_dict):
        return asyncio.run(self.gather_results(worker_info_to_workload_dict, workers_info_dict))

    @staticmethod
    def filter_unreliable_workloads(workload_to_worker_result, unreliable_workers):
        worker_list, result = workload_to_worker_result

        reliable_workers = [worker_id for worker_id in worker_list if worker_id not in unreliable_workers]

        return len(reliable_workers) > 0

    def validate_workloads(self,
                           worker_info_to_workload_info,
                           worker_info_to_result_dict,
                           worker_capacity_dict,
                           workload_redundancy_dict
                           ):
        # TODO add a public and private handshake
        # verification steps
        # 1. all the required fields and nothing but the required fields are present
        # 2. the fields are of the correct data type, format, and shape
        # 3. checksum verification
        # 4. handshake verification
        # 5. shared compute redundancy verification
        # 6. three-way voting compute redundancy verification

        # global counter
        # data_tuple = (
        #     worker_info_to_workload_info, worker_info_to_result_dict, worker_capacity_dict, workload_redundancy_dict)
        #
        # with open(f"data/validate_workloads_input_data_{counter}.pkl", "wb") as f:
        #     pickle.dump(data_tuple, f)
        # counter += 1

        required_fields = [
            "probabilities",
            "probabilities_index",
            "server_elapsed_time",
        ]

        # Create a copy of the original dictionary to validate
        validated_workloads = worker_info_to_result_dict.copy()

        for worker_info in list(validated_workloads.keys()):
            result = validated_workloads[worker_info]
            # Find missing fields by checking against the required fields
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                # Remove the invalid result
                validated_workloads.pop(worker_info)
                # Set the failure reason including the specific missing fields
                if "failure_reason" not in worker_info_to_result_dict[worker_info]:
                    missing_fields_str = ", ".join(missing_fields)
                    worker_info_to_result_dict[worker_info]["failure_reason"] = (
                        f"Missing required fields: {missing_fields_str}"
                    )

        def filter_result_len_capacity(result_dict, capacity):
            probabilities = result_dict["probabilities"]
            probabilities_index = result_dict["probabilities_index"]
            equal_length = len(probabilities) == len(probabilities_index)
            match_capacity = len(probabilities) == capacity

            return equal_length, match_capacity

        for worker_info in list(validated_workloads.keys()):
            result = validated_workloads[worker_info]
            # Get the boolean values for each condition
            equal_length, match_capacity = filter_result_len_capacity(result, worker_capacity_dict[worker_info])
            # if not (equal_length and match_capacity):
            if not equal_length:
                validated_workloads.pop(worker_info)
                if "failure_reason" not in worker_info_to_result_dict[worker_info]:
                    # Determine which conditions failed
                    failure_reasons = []
                    if not equal_length:
                        failure_reasons.append("equal_length is False")
                    if not match_capacity:
                        failure_reasons.append("match_capacity is False")
                    # Join the failure reasons into a single string
                    worker_info_to_result_dict[worker_info]["failure_reason"] = (
                            "Invalid result length: " + ", ".join(failure_reasons)
                    )

        for worker_info, result in validated_workloads.items():
            result["probabilities"] = [np.array(prob) for prob in result["probabilities"]]
            result["probabilities_index"] = [np.array(prob_index) for prob_index in result["probabilities_index"]]

        workload_id_counter = Counter(
            [workload_id for workload_list in worker_info_to_workload_info.values() for workload_id in workload_list])

        workload_set = set(
            [workload for workload_list in worker_info_to_workload_info.values() for workload in workload_list])

        unique_tasks = len(workload_set)

        workload_to_worker_result = {}

        for worker_info, workload_info in worker_info_to_workload_info.items():

            if worker_info not in validated_workloads:
                continue

            worker_results = validated_workloads[worker_info]
            for i, workload_id in enumerate(workload_info):
                if workload_id not in workload_to_worker_result:
                    workload_to_worker_result[workload_id] = {}
                worker_result_dict = {
                    "probabilities": worker_results["probabilities"][i],
                    "probabilities_index": worker_results["probabilities_index"][i],
                }
                workload_to_worker_result[workload_id][worker_info] = worker_result_dict

        workload_to_worker_result = dict(
            sorted(workload_to_worker_result.items(), key=lambda x: len(x[1]), reverse=True))

        # def is_close_to_group(matrix, group, atol=1e-06):
        #     for mat in group:
        #         if np.allclose(matrix, mat, atol=atol):
        #             return True
        #     return False

        workload_to_worker_results_grouped = {}

        invalid_responses = []

        for workload_id, worker_results in workload_to_worker_result.items():

            # each group will have a list of worker info and one result
            groups = []
            for worker_id, worker_result in worker_results.items():
                if (len(worker_result["probabilities_index"].shape) == 1 or
                        worker_result["probabilities"].shape == (1, 1)):
                    if worker_id not in invalid_responses:
                        logger.info(f"Identified out of memory response from worker {worker_id}, skipping ... ")
                        invalid_responses.append(worker_id)
                    if "failure_reason" not in worker_info_to_result_dict[worker_id]:
                        worker_info_to_result_dict[worker_id]["failure_reason"] = "Out of memory"
                    continue
                if worker_result["probabilities"].dtype not in [np.float32, np.float64, np.float16]:
                    if worker_id not in invalid_responses:
                        logger.debug(
                            f"Invalid dtype: {worker_result['probabilities'].dtype} from worker {worker_id}, skipping ... ")
                        invalid_responses.append(worker_id)
                    if "failure_reason" not in worker_info_to_result_dict[worker_id]:
                        worker_info_to_result_dict[worker_id]["failure_reason"] = "Invalid dtype"

                    continue

                added = False
                for worker_id_list, group_result in groups:
                    # if is_close_to_group(worker_result, group_result):
                    shape_equal = worker_result["probabilities"].shape == group_result["probabilities"].shape
                    if not shape_equal:
                        # logger.info(f"Shape mismatch, worker: {worker_result['probabilities'].shape}, ")
                        continue
                    # lenient threshold since quantization can cause small differences
                    logit_all_close = np.allclose(worker_result["probabilities"],
                                                  group_result["probabilities"],
                                                  atol=1e-01,
                                                  rtol=1e-02)

                    total_cells = np.size(worker_result["probabilities_index"][:, :1])
                    equal_cells = np.sum(
                        worker_result["probabilities_index"][:, :1] == group_result["probabilities_index"][:, :1])

                    # only check the first 3 elements, softmax is top-heavy
                    # logit_index_equal = np.equal(worker_result["probabilities_index"][:, :1],
                    #                                 group_result["probabilities_index"][:, :1])
                    logit_index_equal = equal_cells / total_cells > 0.98

                    if logit_all_close and logit_index_equal:
                        worker_id_list.append(worker_id)
                        added = True
                        break
                    else:
                        max_abs_diff = np.max(
                            np.abs(worker_result["probabilities"] - group_result["probabilities"]))
                        max_rel_diff = np.max(
                            np.abs(worker_result["probabilities"] - group_result["probabilities"]) / group_result[
                                "probabilities"])

                        logger.info(
                            f"Mismatch, max_abs_diff: {max_abs_diff:.2f}, max_rel_diff: {max_rel_diff:.2f}, equal_ratio: {equal_cells / total_cells:.2f}")

                if not added:
                    groups.append(([worker_id], worker_result))
            workload_to_worker_results_grouped[workload_id] = groups

        workload_to_worker_results_grouped = dict(
            sorted(workload_to_worker_results_grouped.items(), key=lambda x: len(x[1]), reverse=True))

        # now, we check if any worker
        # failed_worker_list = []
        # for workload_id, groups in workload_to_worker_results_grouped.items():
        #     all_workers = [ww for worker_list, _ in groups for ww in worker_list]
        #     worker_count = len(all_workers)
        #     required_majority = math.ceil(worker_count / 2)
        #     if worker_count == 3:
        #         required_majority = 2
        #
        #     for worker_list, _ in groups:
        #         if len(worker_list) < required_majority:
        #             failed_worker_list.extend(worker_list)
        #
        # failed_worker_list = list(set(failed_worker_list))

        worker_win_dict = {worker_id: 0 for worker_id in validated_workloads.keys()}
        worker_loss_dict = {worker_id: 0 for worker_id in validated_workloads.keys()}

        majority_validated_workloads = {}

        for workload_id, groups in workload_to_worker_results_grouped.items():
            all_workers = [ww for worker_list, _ in groups for ww in worker_list]
            worker_count = len(all_workers)
            required_majority = math.ceil(worker_count / 2)

            # find the majority group
            majority_group_index, majority_group = next(
                filter(lambda x: len(x[1][0]) >= required_majority, enumerate(groups)),
                (None, None))

            if majority_group_index is not None:
                majority_validated_workloads[workload_id] = majority_group

                for i in range(len(groups)):
                    is_majority = True if i == majority_group_index else False

                    if is_majority:
                        for worker_id in groups[i][0]:
                            worker_win_dict[worker_id] += 1
                    else:
                        for worker_id in groups[i][0]:
                            worker_loss_dict[worker_id] += 1

                # check i
            else:
                # no majority group
                pass

        unreliable_workers = [worker_id for worker_id, losses in worker_loss_dict.items() if
                              losses > 0]

        validated_workloads = {workload_id: majority_validated_workloads[workload_id]
                               for workload_id in majority_validated_workloads.keys()
                               if self.filter_unreliable_workloads(majority_validated_workloads[workload_id],
                                                                   unreliable_workers)}

        return validated_workloads

    def save_results(self, validated_workloads):
        validated_workload_list = [{"id": workload_id,
                                    "probabilities": result[1]["probabilities"].tolist(),
                                    "probabilities_index": result[1]["probabilities_index"].tolist(),
                                    }
                                   for workload_id, result in validated_workloads.items()]

        validated_workload_df = pd.DataFrame(validated_workload_list)

        validated_workload_id_set = set([workload_id for workload_id in validated_workloads.keys()])

        self.processed_id_list.extend(list(validated_workload_id_set))

        processed_list_df = pd.DataFrame({"id": self.processed_id_list})
        processed_list_df.to_parquet(self.processed_id_arrow_path)

        current_time_file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        out_dir = os.path.join(self.out_dir, self.validated_out_dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        path = os.path.join(out_dir, f"{current_time_file}.parquet")
        validated_workload_df.to_parquet(path)

    def record_worker_performance(self,
                                  workers_info_dict,
                                  validated_workloads,
                                  worker_info_to_result_dict,
                                  worker_capacity_dict):
        worker_validated_counter = Counter(
            [worker_id for worker_list, _ in validated_workloads.values() for worker_id in worker_list])

        worker_validated_counter = dict(sorted(worker_validated_counter.items(), key=lambda x: x[0], reverse=False))

        for worker_id, validated_workload_count in worker_validated_counter.items():
            if worker_id not in self.worker_completed_workload:
                self.worker_completed_workload[worker_id] = 0
            self.worker_completed_workload[worker_id] += validated_workload_count

        worker_metrics_list = []

        for worker_info, result_dict in worker_info_to_result_dict.items():
            metrics = {}
            metrics["worker_id"] = worker_info[:10]
            metrics["address"] = workers_info_dict[worker_info]["ip"] + ":" + str(
                workers_info_dict[worker_info]["port"])
            metrics["capacity"] = worker_capacity_dict[worker_info]
            metrics["validated"] = worker_validated_counter.get(worker_info, 0)
            metrics["completion_ratio"] = metrics["validated"] / metrics["capacity"]
            metrics["elapsed_time"] = result_dict["server_elapsed_time"]
            # metrics["elapsed_time"] = result_dict.get("elapsed_time", None)
            # metrics["time_per_workload"] = metrics["server_elapsed_time"] / metrics["capacity"]
            effective_time_per_workload = 0
            if metrics["validated"] > 0:
                effective_time_per_workload = metrics["elapsed_time"] / metrics["validated"]
            metrics["time_per_workload"] = effective_time_per_workload

            metrics["failure_reason"] = result_dict.get("failure_reason", None)

            if "device_info" in result_dict and isinstance(result_dict["device_info"], list):
                name_list = []
                for device in result_dict["device_info"]:
                    name_list.append(device.get("name", None))
                metrics["device_name"] = ", ".join(name_list)

            worker_metrics_list.append(metrics)

        worker_metrics_list = sorted(worker_metrics_list, key=lambda x: (x["validated"],
                                                                         x["elapsed_time"]),
                                     reverse=True)
        worker_metrics_df = pd.DataFrame(worker_metrics_list)

        logger.info(f"Worker Metrics: \n{worker_metrics_df.to_string()}")

        self.worker_time_per_workload = {}

        for worker_id, workload_count in worker_validated_counter.items():
            time_elapsed = int(worker_info_to_result_dict[worker_id]["server_elapsed_time"])
            if workload_count > 0:
                self.worker_time_per_workload[worker_id] = time_elapsed / workload_count

    def submit_weights(self):
        logger.info("Submitting weights ... ")

        workers_info_dict = self.get_workers_info()
        total_completed_workload = sum(self.worker_completed_workload.values())

        # worker_weight_dict = {worker_id: count / total_completed_workload
        #                       for worker_id, count in self.worker_completed_workload.items()}

        # uids = list(worker_weight_dict.keys())
        uids = [workers_info_dict[worker_info]["id"] for worker_info in self.worker_completed_workload.keys()]
        weights = list(self.worker_completed_workload.values())

        logger.info(f"Submitting weights: {self.worker_completed_workload}")
        try:
            self.client.vote(key=self.key, uids=uids, weights=weights, netuid=self.net_uid)
        except Exception as e:
            logger.error(f"Error submitting weights: {e}")

    def validate_step(
            self,
    ) -> None:
        """
        Perform a validation step.
        """

        logger.info(f"Starting validation step.")

        workers_info_dict = self.get_workers_info()

        if len(workers_info_dict) == 0:
            logger.error("No workers found, aborting ...")
            return
        logger.info(f"Fetched {len(workers_info_dict)} workers ... ")

        try:
            worker_info_to_workload_dict, worker_info_to_workload_info, worker_capacity_dict, workload_redundancy_dict = (
                self.get_workload(
                    workers_info_dict))
        except Exception as e:
            logger.error(f"Error getting workload: {e}")
            return
        logger.info(f"Generated workload.")

        worker_info_to_result_dict = self.run_gather_results(worker_info_to_workload_dict,
                                                             workers_info_dict)
        logger.info(f"Gathered results.")

        validated_workloads = self.validate_workloads(worker_info_to_workload_info,
                                                      worker_info_to_result_dict,
                                                      worker_capacity_dict,
                                                      workload_redundancy_dict
                                                      )
        logger.info(f"Validated workloads.")

        self.save_results(validated_workloads)
        logger.info(f"Saved results.")

        self.record_worker_performance(workers_info_dict,
                                       validated_workloads,
                                       worker_info_to_result_dict,
                                       worker_capacity_dict)
        logger.info(f"Recorded worker performance.")

        logger.info(f"Validation step completed.")

    def validation_loop(self) -> None:
        while True:
            logger.info("Validation loop started...")
            start_time = time.perf_counter()
            self.validate_step()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            if time.time() - self.last_validation_time > self.weight_submit_interval:
                # submit weights
                self.last_validation_time = time.time()
                self.submit_weights()
                self.worker_completed_workload = {}

            if self.use_dynamic_interval:
                interval = random.randint(1500, 4500)
            else:
                interval = self.interval

            logger.info(f"Validation loop took {elapsed_time:.2f} seconds, sleeping for {interval} seconds.")
            # if elapsed_time < 30:
            #     logger.warning("Validation loop took less than 30 seconds, sleeping for 30 seconds.")
            #     time.sleep(30)
            time.sleep(interval)


def main():
    logger.info("Loading validator configuration ... ")

    parser = argparse.ArgumentParser(description="Load validator configuration.")
    parser.add_argument('--name', type=str, default='nya-compute-validator', help='Validator name')
    parser.add_argument('--keyfile', type=str, default='nya-validator', help='Path to keyfile')
    parser.add_argument('--interval', type=int, default=1, help='Interval in seconds')
    parser.add_argument('--testnet', type=bool, default=False, help='Is testnet (True/False)')
    parser.add_argument('--subnet_uid', type=int, default=8, help='Subnet UID, default is 23 for testnet')
    args = parser.parse_args()

    logger.info(f"Validator configuration: {args}")

    validator_config = {
        "name": args.name,
        "keyfile": args.keyfile,
        "interval": args.interval,
        "testnet": args.testnet,
    }

    use_testnet = True if validator_config.get("testnet") else False

    if use_testnet:
        logger.debug("Connecting to TEST network ... ")
    else:
        logger.debug("Connecting to Main network ... ")

    commute_client = CommuneClient(get_node_url(use_testnet=use_testnet))
    subnet_uid = args.subnet_uid
    keypair = classic_load_key(validator_config.get("keyfile"))
    validator = ComputeValidator(
        key=keypair,
        net_uid=subnet_uid,
        client=commute_client,
        interval=validator_config.get("interval"),
        call_timeout=1000
    )

    logger.info("Running validator ... ")
    validator.validation_loop()


if __name__ == '__main__':
    main()
