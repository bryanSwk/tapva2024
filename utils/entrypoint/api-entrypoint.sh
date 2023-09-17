#!/bin/bash

set -x

source ~/.bashrc

gunicorn src.inference_fastapi:app -b 0.0.0.0:4000 -w 1 -k uvicorn.workers.UvicornWorker --timeout 1000