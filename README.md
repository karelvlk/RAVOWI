# RAVOWI: Real-time Analysis and Validation Of Webcam Images

This application is a product of my Bachelor's thesis at the Faculty of Mathematics and Physics, Charles University. The thesis project exemplifies the practical application and integration of advanced AI methodologies in real-time image processing and analysis. 

The complete thesis is located in the `doc/cs/` directory and is available only in the Czech language.

---

# Meteval (MetSnap): Real-time Image Analysis and Validation App

Meteval (also called MetSnap) represents a state-of-the-art demonstration application crafted to exhibit the most advanced research methodologies in real-time analysis and validation of webcam images. This application stems from an extensive exploration within a bachelor's thesis, meticulously detailing the employed methods and the rationale behind their selection.

## Initial Setup

Follow these instructions to seamlessly deploy and run the project on your local environment for development and testing pursuits.

### System Requirements

Before initiating, ensure your system meets the following prerequisites:

- Docker installed (^4.28.0)
  - Supported versions:
    - Docker Desktop (^4.28.0)
    - Docker Engine (^26.0.0)
    - Docker Compose (^2.26.0)
- A compatible operating system and CPU architecture:
  - Linux, Windows with x86_64 (amd64) CPU
  - macOS with Apple Silicon (arm64) CPU
- At least 20GB of available disk space
- At least 4GB of RAM

### Quickstart Guide

#### For Linux or Windows Users:

Execute the following command to build and launch the application:

```sh
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up
```

#### For Apple Silicon Mac Users:

Utilize this command to accommodate the specific architecture:

```sh
docker compose -f docker-compose.mac.yml build
docker compose -f docker-compose.mac.yml up
```

**⚠️ Note ⚠️**: The process involves downloading Docker Images totaling approximately 5GB.

### Execution

Upon the successful construction of Docker containers via the Docker Compose tool, the application's client interface will be accessible at `http://localhost:3000`.

## Technical Insights

The application operates entirely on CPU, with all deep learning models being facilitated by the NVIDIA Triton Inference Server. These models are in ONNX format for broad compatibility. An alternative execution path exists for leveraging TensorRT format on CUDA GPUs, which necessitates direct model compilation on the specific GPU in use. The conversion from ONNX to TensorRT formats can be efficiently performed using the `trtexec` utility.
