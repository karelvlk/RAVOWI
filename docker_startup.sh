#!/bin/bash
echo "docker starting nginx ..."
nginx

# echo "Starting performance tests"
# echo "-------------------------"
# python perf_test.py --num_requests 512 --concurrent_requests 2 --batch_size 1
# echo "-------------------------"
# python perf_test.py --num_requests 512 --concurrent_requests 4 --batch_size 1
# echo "-------------------------"
# python perf_test.py --num_requests 8192 --concurrent_requests 8 --batch_size 1
# echo "-------------------------"
# echo "Performance tests finished"

echo "docker starting AI backend server ..."
uvicorn main:app --host 0.0.0.0 --port 8080
