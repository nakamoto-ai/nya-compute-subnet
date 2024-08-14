# Running a Validator

```bash

git clone https://github.com/nakamoto-ai/nya-compute-subnet.git
cd nya-compute-subnet

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run The Validator
python src/validator.py --name <vali-name> --keyfile <key>

# Run the Validator w/ PM2
pm2 start "src/validator.py --name <vali-name> --keyfile <key>" --name nya-vali

```