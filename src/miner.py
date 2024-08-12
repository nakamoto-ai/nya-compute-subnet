import argparse
import base64
import datetime
import logging
import time
from communex.compat.key import classic_load_key
from keylimiter import TokenBucketLimiter
from communex.module.server import ModuleServer
from torch.cuda import synchronize

import torch
import uvicorn
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from communex.module import endpoint
from communex.module.module import Module
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

disable_progress_bar()

time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

logger.info(f"Running {__file__}")


def verify_signature(data, signature, public_key_path="keys/public.pem"):
    with open(public_key_path, 'r') as key_file:
        public_key = RSA.import_key(key_file.read())

    data_str = data
    if not isinstance(data, str):
        data_str = str(data)
    hash_value = SHA256.new(data_str.encode('utf-8'))
    signature = base64.b64decode(signature)
    try:
        pkcs1_15.new(public_key).verify(hash_value, signature)
        return True
    except (ValueError, TypeError):
        return False


class NyaComputeMiner(Module):
    version = "0.1"

    def __init__(self,
                 batch_size: int = 64,
                 device_map: str = "auto",
                 debug: bool = False
                 ):
        super().__init__()

        if debug:
            model_name = "distilbert/distilbert-base-uncased"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = self.model.to("cuda").half()
        else:
            model_name = "google/t5-small"
            commit_hash = "main"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=commit_hash)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.batch_size = batch_size

        if device_map == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA is not available. Aborting.")
            raise ValueError("CUDA is not available.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

        device_info_list = []
        attr_name_list = ["name", "gcnArchName", "is_integrated", "is_multi_gpu_board", "major",
                          "max_threads_per_multi_processor", "minor", "multi_processor_count", "total_memory", ]
        for i in range(torch.cuda.device_count()):
            device_dict = {}
            device_properties = torch.cuda.get_device_properties(i)
            for attr_name in attr_name_list:
                if hasattr(device_properties, attr_name):
                    device_dict[attr_name] = getattr(device_properties, attr_name)
            device_info_list.append(device_dict)

        self.device_info = device_info_list

        logger.info(f"Initialized {self.__class__.__name__}, using device: {self.device}")

    def batch_encode(self, batch):
        batch = dict(batch)
        try:
            # Handle different formats of input batch
            if isinstance(batch, list):
                texts = [item for sublist in batch for item in sublist.get("text", [])]
            elif isinstance(batch, dict) and "text" in batch:
                texts = batch["text"]
            else:
                raise ValueError(f"Unexpected batch format: {batch}")

            # Use tokenizer to encode texts
            encoded = self.tokenizer(
                texts,
                max_length=1024,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            )

            # Move encoded tensors to the correct device
            for key, tensor in encoded.items():
                encoded[key] = tensor.to(self.device)

            # Prepare decoder input ids
            decoder_start_token_id = self.model.config.decoder_start_token_id
            decoder_input_ids = torch.full(
                (encoded["input_ids"].shape[0], 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=self.device
            )

            encoded["decoder_input_ids"] = decoder_input_ids

            # Debugging: Print tensor shapes
            print(f"input_ids shape: {encoded['input_ids'].shape}")
            print(f"attention_mask shape: {encoded['attention_mask'].shape}")
            print(f"decoder_input_ids shape: {encoded['decoder_input_ids'].shape}")

            return encoded

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise

    @endpoint
    def compute(self, task: list[str]):
        try:
            input_data = Dataset.from_dict({"text": task})
            encoded = input_data.map(self.batch_encode, batched=True, remove_columns=["text"])

            if "decoder_input_ids" not in encoded.column_names:
                raise ValueError("Expected column 'decoder_input_ids' not found in the dataset")

            encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids"])

            data_loader = DataLoader(encoded, batch_size=self.batch_size)

            all_probabilities = []
            all_probabilities_index = []

            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"]
                )

                logits = output.logits.detach().cpu().numpy()
                probabilities = logits.tolist()

                all_probabilities.extend(probabilities)
                all_probabilities_index.extend(list(range(len(probabilities))))

            result = {
                'probabilities': all_probabilities,
                'probabilities_index': all_probabilities_index
            }

        except Exception as e:
            logger.error(f"Error during computation: {e}")
            raise

        return result


def main():
    parser = argparse.ArgumentParser(description="Load validator configuration.")
    parser.add_argument("--name", help="Miner name.", default="nya compute miner")
    parser.add_argument("--keyfile", help="Name of the key file", default="nya-miner")
    parser.add_argument("--ip", help="IP address to bind the server to.", default="0.0.0.0")
    parser.add_argument("--port", help="Port to bind the server to.", default=9910)
    parser.add_argument("--device_map", help="Device to run the model on.", default="auto")
    parser.add_argument("--batch_size", help="Batch size for the model.", default=1, type=int)
    parser.add_argument("--subnetuid", help="Subnet UID to bind the server to.", default=23)
    parser.add_argument("--testnet", help="Use testnet.", default=True)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Aborting.")
        raise ValueError("CUDA is not available.")

    logger.info(f"Miner configuration: {args}")

    batch_size = int(args.batch_size)

    miner = NyaComputeMiner(
        batch_size=batch_size,
        device_map=args.device_map,
    )

    port = args.port

    if not isinstance(port, int):
        if isinstance(port, str) and port.isdigit():
            port = int(port)
    # Handle if it's a string but not a number
    else:
        logger.error("Port must be an integer. Aborting.")
        raise ValueError("Port must be an integer.")

    try:
        key = classic_load_key(args.keyfile)
    except FileNotFoundError:
        logger.error(f"Key file {args.keyfile} not found. Aborting.")
        raise FileNotFoundError(f"Key file {args.keyfile} not found.")

    refill_rate = 1  # Set your desired refill rate

    bucket = TokenBucketLimiter(30, refill_rate)  # Initialize your TokenBucketLimiter

    if args.testnet:
        logger.info("Using testnet")

    # Initialize the ModuleServer with the miner and other configurations
    server = ModuleServer(
        miner,
        key,
        limiter=bucket,
        subnets_whitelist=[args.subnetuid],
        use_testnet=args.testnet
    )

    app = server.get_fastapi_app()  # Get FastAPI app from server

    uvicorn.run(app, host=args.ip, port=port)  # Run the server with uvicorn


if __name__ == "__main__":
    main()
