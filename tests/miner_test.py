import unittest

from datasets import load_dataset

import src.miner as miner


class MyTestCase(unittest.TestCase):

    def test_correct_input(self):
        compute_miner = miner.NyaComputeMiner(device_map="auto", debug=True)

        dataset = load_dataset("allenai/c4", "en", streaming=True)
        train_set = dataset["train"]

        # add index number to each row in train_set

        workload_amount = 32

        train_set_list = list(train_set.take(workload_amount))

        task = [r["text"] for r in train_set_list]

        result = compute_miner.compute(task)


if __name__ == '__main__':
    unittest.main()
