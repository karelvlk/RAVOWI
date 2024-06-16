FROM nvcr.io/nvidia/tritonserver:22.09-py3
COPY ./models /models
COPY ./triton_startup.sh /triton_startup.sh
ENTRYPOINT [ "sh", "/triton_startup.sh" ] 
