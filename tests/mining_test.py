import logging
import time

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

import src.miner as miner

disable_progress_bar()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Running {__file__}")


def main():
    dataset = load_dataset("allenai/c4", "en", streaming=True)
    compute_miner = miner.NyaComputeMiner(device_map="auto", debug=True)

    train_set = dataset["train"]

    # add index number to each row in train_set
    max_time = 150
    workload_amount = 32
    while True:
        start_time = time.perf_counter()
        train_set_list = list(train_set.take(workload_amount))

        task = [r["text"] for r in train_set_list]

        result = compute_miner.compute(task)
        end_time = time.perf_counter()
        time_elapsed = end_time - start_time

        time_per_task = time_elapsed / workload_amount
        workload_amount = max_time / time_per_task
        logger.info(f"Time per task: {time_per_task}")


if __name__ == '__main__':
    main()