#!/bin/bash

tritonserver --model-repository=/models/yolo-onnx-cpu --http-port=8000 --grpc-port=8001
