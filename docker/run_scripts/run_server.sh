#!/bin/bash

PORT=${1:-4000}
uvicorn main:app --host 0.0.0.0 --port $PORT
