import argparse
import logging
import time

from datasets import Dataset
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import src.miner as miner

disable_progress_bar()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Running {__file__}")


def test_mining(debug: bool = False, batch_size: int = 1, repeat_n_times=1):
    dataset = load_dataset("allenai/c4", "en", streaming=True)
    compute_miner = miner.NyaComputeMiner(device_map="auto", debug=debug, batch_size=batch_size)

    train_set = dataset["train"]

    # add index number to each row in train_set
    max_time = 5
    workload_amount = 5

    for _ in range(repeat_n_times):
        start_time = time.perf_counter()
        train_set_list = list(train_set.take(workload_amount))

        task = [r["text"] for r in train_set_list]

        result = compute_miner.compute(task)
        end_time = time.perf_counter()
        time_elapsed = end_time - start_time

        time_per_task = time_elapsed / workload_amount
        workload_amount = int(max_time / time_per_task)
        logger.info(f"Time per task: {time_per_task}")


def test_encoding(debug: bool = False, batch_size: int = 1):
    dataset = load_dataset("allenai/c4", "en", streaming=True)
    compute_miner = miner.NyaComputeMiner(device_map="auto", debug=debug, batch_size=batch_size)

    train_set = dataset["train"]

    # add index number to each row in train_set
    max_time = 5
    workload_amount = 5
    skip = 0
    while True:
        start_time = time.perf_counter()
        train_set_list = list(train_set.skip(skip).take(workload_amount))
        skip += workload_amount

        task = [r["text"] for r in train_set_list]

        task_dict_list = [{"text": t} for t in task]

        input_data = Dataset.from_list(task_dict_list)

        encoded = input_data.map(compute_miner.batch_encode,
                                 batched=True,
                                 remove_columns=["text"],
                                 batch_size=compute_miner.batch_size,
                                 )
        data_loader = DataLoader(encoded, compute_miner.batch_size)

        batch_len = set([len(batch["input_ids"]) for batch in data_loader])
        logger.info(f"Batch len: {batch_len}")


def main():
    parser = argparse.ArgumentParser(description="Load validator configuration.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the model.")

    args = parser.parse_args()
    debug = args.debug
    batch_size = args.batch_size

    test_mining(debug, batch_size * 2)
    test_mining(debug, batch_size)

    # test_encoding(debug, batch_size)


if __name__ == '__main__':
    main()
