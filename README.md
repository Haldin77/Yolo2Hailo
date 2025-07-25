# YOLOv8‑Seg `best.pt` → HEF Hailo‑8

**Deployment on Raspberry Pi 5 + Hailo‑8 / Hailo‑8L**
*Version: July 2025*

Project carried out at the **CRIStAL laboratory — University of Lille**.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Docker Installation](#docker-installation)
3. [Running the Hailo AI Suite Container](#running-the-hailo-ai-suite-container)
4. [Exporting the YOLOv8‑Seg Model to ONNX (outside Docker)](#exporting-the-yolov8‑seg-model-to-onnx-outside-docker)
5. [Compiling the Model to HEF (inside the Container)](#compiling-the-model-to-hef-inside-the-container)
6. [Raspberry Pi 5 Usage Guide](#raspberry-pi-5-usage-guide)

---

## System Requirements

* **OS:** Ubuntu 20.04 / 22.04 (64‑bit)
* **RAM:** ≥ 16 GB (32 GB recommended)
* **Docker:** `docker.io` 20.10.07 or `docker-ce` 20.10.6

---

## Docker Installation

### 1. Add your user to the `docker` group

```bash
sudo usermod -aG docker ${USER}
# then log out and log back in
```

### 2. (Optional) NVIDIA support for emulation/quantization

Configure the production repository:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

(Optional) Enable experimental packages:

```bash
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Update the package list:

```bash
sudo apt-get update
```

Install the NVIDIA Container Toolkit packages:

```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

sudo systemctl restart docker
```

* Compatible GPU: Pascal/Turing/Ampere (Titan X Pascal, GTX 1080 Ti, RTX 2080 Ti, RTX A4000…)
* NVIDIA driver 525

---

## Running the Hailo AI Suite Container

1. **Download** the software suite and PCIe driver from the Developer Zone.
2. **Install** the PCIe driver:

   ```bash
   sudo dpkg -i <pcie_driver>.deb
   sudo reboot
   ```
3. **Unzip** and start the container:

   ```bash
   unzip hailo_ai_sw_suite_<version>.zip
   ./hailo_ai_sw_suite_docker_run.sh
   ```
4. **Verify** Hailo packages and CLI help:

   ```bash
   pip list | grep hailo
   hailo -h
   ```
5. **Script options:**

   * `--help` display help
   * `--hailort-enable-service` enable the HailoRT multi‑process service
   * `--resume` re‑attach to an existing container
   * `--override` delete and recreate the container

---

## Exporting the YOLOv8‑Seg Model to ONNX (outside Docker)

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv && source venv/bin/activate
   ```
2. Install dependencies:

   ```bash
   pip install "ultralytics>=8.2" onnx onnxsim
   ```
3. Export the model:

   ```bash
   yolo task=segment mode=export \
        model=best.pt \
        format=onnx \
        imgsz=640 \
        opset=12
   mv best.onnx bestcrack.onnx
   ```

---

## Compiling the Model to HEF (inside the Container)

### Project Structure

Place the following files in a folder (e.g. `hailo_project/`):

```
hailo_project/
├── bestcrack.onnx
└── calib_imgs/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### One‑line Compilation

```bash
cd hailo_project
# From the host:
./hailo_ai_sw_suite_docker_run.sh --resume
# Or if you are already inside the container:
hailo_docker

hailomz compile yolov8m_seg \
  --ckpt=bestcrack.onnx \
  --hw-arch=hailo8 \
  --calib-path=calib_imgs \
  --classes=1 \
  --performance
```

> **Note:** Replace `--hw-arch=hailo8` with `hailo8l` when using a Hailo‑8L, and adjust `--classes` to match your dataset.

### Expected Outputs

* `yolov8m_seg.hef`
* `yolov8m_seg.runtime.json`
* `compile_report.html`

---

## Raspberry Pi 5 Usage Guide

For detailed deployment instructions on Raspberry Pi 5, see the file [models/README.md](./models/README.md).

---

*End of README.*
