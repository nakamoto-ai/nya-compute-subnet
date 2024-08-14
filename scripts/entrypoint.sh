#!/bin/bash

echo "Running entrypoint.sh in $(pwd)"

pm2 start "python -m src.validator" --name "nya-validator"

pm2 logs