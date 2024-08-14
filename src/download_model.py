from transformers import AutoModel, AutoTokenizer

model_name = "distilbert/distilbert-base-uncased"

AutoModel.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)
