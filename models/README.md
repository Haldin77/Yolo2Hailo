# Guide: Using a HEF YOLOv8‑Seg Model with DeGirum PySDK and Hailo‑8 on Raspberry Pi 5

This guide builds on the tutorial **“Using DeGirum PySDK, DeGirum Tools, and Hailo Hardware”** and explains how to:

1. Prepare your Raspberry Pi 5 equipped with a Hailo‑8 or Hailo‑8L accelerator.
2. Install the DeGirum tool‑chain.
3. Organize a local *model zoo* and deploy your compiled model (`yolov8n_seg`).

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Organizing the `models/` Folder](#organizing-the-models-folder)
5. [Running & Configuring Jupyter Notebooks](#running--configuring-jupyter-notebooks)
6. [Example Inference Script](#example-inference-script)
7. [Additional Resources](#additional-resources)

---

## Introduction

**DeGirum** provides a set of tools that simplify the development and deployment of **Edge AI** applications:

* **DeGirum PySDK** – the core Python library for integrating AI inference into your apps.
* **DeGirum Tools** – utilities for benchmarking, streaming, and interacting with the DeGirum *model zoo*.

Because these tools are **hardware‑agnostic**, you can create flexible and scalable solutions that run on various platforms, including **Hailo‑8** and **Hailo‑8L**.

---

## Prerequisites

| Component                         | Version / Notes                                           |
| --------------------------------- | --------------------------------------------------------- |
| **Hailo Tools**                   | Installed and configured (see official Hailo docs)        |
| **HailoRT Multi‑Process Service** | Enabled: `sudo systemctl enable --now hailort.service`    |
| **Hailo Runtime**                 | Versions supported by PySDK: **4.19.0 / 4.20.0 / 4.21.0** |
| **Python**                        | ≥ 3.9 (`python3 --version`)                               |
| **Raspberry Pi OS**               | 64‑bit, fully updated                                     |

---

## Installation

> **Tip:** The steps below mirror the DeGirum `hailo_examples` repository.

### 1. Clone the example repository

```bash
git clone https://github.com/DeGirum/hailo_examples.git
cd hailo_examples
```

### 2. Create a virtual environment

```bash
python3 -m venv degirum_env
source degirum_env/bin/activate  # Linux / macOS
# On Windows:
# degirum_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Add the venv to Jupyter

```bash
python -m ipykernel install --user --name=degirum_env --display-name "Python (degirum_env)"
```

### 5. Verify the installation

```bash
python test.py
```

The script will:

1. Print system information.
2. Detect Hailo hardware.
3. Load a sample model and run an inference.

If everything passes, your environment is ready ! ✅

---

## Organizing the `models/` Folder

For the following call to work:

```python
model = dg.load_model(
    model_name="yolov8n_seg",
    inference_host_address="@local",
    zoo_url="/home/robucar/hailo_examples/models",
    device_type=["HAILORT/HAILO8"],
)
```

the directory structure must look like this:

```
/home/robucar/hailo_examples/models/
└── yolov8n_seg/
    ├── yolov8n_seg.hef        # Compiled graph for Hailo
    ├── yolov8n_seg.json       # DeGirum configuration
    └── crack.json             # Label map (1 class: "crack")
```

### File responsibilities

| File                  | Purpose                                                    |
| --------------------- | ---------------------------------------------------------- |
| **yolov8n\_seg.hef**  | Network binary compiled via Hailo Dataflow Compiler.       |
| **yolov8n\_seg.json** | DeGirum config (pre/post‑process, device, path to `.hef`). |
| **crack.json**        | Label dictionary (numeric key → class name).               |

> **Important:** The folder name **and** the filename prefix must match the `model_name` supplied to `dg.load_model`.

#### Minimal `yolov8n_seg.json` example

```json
{
  "ConfigVersion": 11,
  "Checksum": "<sha256_of_hef>",
  "DEVICE": [{
    "DeviceType": "HAILO8",
    "RuntimeAgent": "HAILORT",
    "SupportedDeviceTypes": "HAILORT/HAILO8"
  }],
  "PRE_PROCESS": [{
    "InputType": "Image",
    "InputN": 1,
    "InputH": 640,
    "InputW": 640,
    "InputC": 3,
    "InputPadMethod": "letterbox",
    "InputResizeMethod": "bilinear",
    "InputQuantEn": true
  }],
  "MODEL_PARAMETERS": [{
    "ModelPath": "yolov8n_seg.hef"
  }],
  "POST_PROCESS": [{
    "OutputPostprocessType": "SegmentationYoloV8",
    "LabelsPath": "crack.json",
    "OutputNumClasses": 1,
    "OutputConfThreshold": 0.3,
    "SigmoidOnCLS": true
  }]
}
```

#### Automatic config generation script

```python
import json, hashlib
hef = "yolov8n_seg.hef"
checksum = hashlib.sha256(open(hef, "rb").read()).hexdigest()
config = {
  "ConfigVersion": 11,
  "Checksum": checksum,
  "DEVICE": [{
      "DeviceType": "HAILO8",
      "RuntimeAgent": "HAILORT",
      "SupportedDeviceTypes": "HAILORT/HAILO8"
  }],
  "PRE_PROCESS": [{
      "InputType": "Image",
      "InputN": 1,
      "InputH": 640,
      "InputW": 640,
      "InputC": 3,
      "InputPadMethod": "letterbox",
      "InputResizeMethod": "bilinear",
      "InputQuantEn": true
  }],
  "MODEL_PARAMETERS": [{"ModelPath": "yolov8n_seg.hef"}],
  "POST_PROCESS": [{
      "OutputPostprocessType": "SegmentationYoloV8",
      "LabelsPath": "crack.json",
      "OutputNumClasses": 1,
      "OutputConfThreshold": 0.3,
      "SigmoidOnCLS": true
  }]
}
with open("yolov8n_seg.json", "w") as f:
    json.dump(config, f, indent=2)
print("yolov8n_seg.json created ✔")
```

---

## Running & Configuring Jupyter Notebooks

1. Start Jupyter:

   ```bash
   jupyter lab  # or jupyter notebook
   ```
2. In the browser, select the **“Python (degirum\_env)”** kernel for your notebooks.
3. The notebooks under `hailo_examples/notebooks/` cover:

   * Latency/throughput benchmarking.
   * Video streaming.
   * Interaction with the DeGirum *model zoo*.

---

## Example Inference Script

```python
import degirum as dg

model = dg.load_model(
    model_name="yolov8n_seg",
    inference_host_address="@local",
    zoo_url="/home/robucar/hailo_examples/models",
    device_type=["HAILORT/HAILO8"],
)

result = model("images/test.jpg")
print(result)
```

Add a simple latency benchmark:

```python
import time
start = time.time()
for _ in range(30):
    _ = model("images/test.jpg")
print("Avg latency (ms):", (time.time()-start)/30*1000)
```

---

## Additional Resources

* **DeGirum PySDK Documentation:** [https://docs.degirum.com/pysdk](https://docs.degirum.com/pysdk)
* **`hailo_examples` Repository:** [https://github.com/DeGirum/hailo\_examples](https://github.com/DeGirum/hailo_examples)
* **Hailo Documentation:** [https://docs.hailo.ai](https://docs.hailo.ai)

---

*End of guide.*
