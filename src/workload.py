from datasets import load_dataset


class Workload:
    def __init__(self, max_token: int, processed_id_list: list):
        # c4_dataset = load_dataset("allenai/c4", "en", streaming=True)["train"]
        fine_web_dataset = load_dataset("HuggingFaceFW/fineweb",
                                        name="CC-MAIN-2024-10",
                                        split="train",
                                        streaming=True)

        # # add index number to each row in train_set
        # def add_index(x, index):
        #     x["index"] = index
        #     return x

        # indexed_train_set = fine_web_dataset.map(add_index, with_indices=True,
        #                                          remove_columns=["timestamp", "url"]
        # )

        if max_token is not None:
            fine_web_dataset = fine_web_dataset.filter(lambda x: x["token_count"] <= max_token)

        self.dataset = fine_web_dataset

        self.remaining_data = self.dataset.filter(lambda x: x["id"] not in processed_id_list)

    def get_workload(self, processed_id_list, workload_amount):
        self.remaining_data = self.dataset.filter(lambda x: x["id"] not in processed_id_list)
        returned_task = list(self.remaining_data.take(workload_amount))
        return returned_task
