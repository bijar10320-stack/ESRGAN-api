# ESRGAN Image Super-Resolution

This repository contains a FastAPI + PyTorch implementation of **ESRGAN / RRDBNet** for image super-resolution. Users can send a low-resolution image to the API, and the model will output a high-resolution version.

---

## Features
- FastAPI backend for serving the model via an API
- RRDBNet / ESRGAN model for upscaling images
- Simple Python script for testing single images
- Before/after image comparison

---

## Demo Images

**Low-Resolution (Input) vs Super-Resolution (Output)**

| Input | Output |
|-------|--------|
| ![Input](images/LR_OIP.jpg) | ![Output](images/SR_OIP.jpg) |

> Replace `images/lr_example.jpg` and `images/sr_example.jpg` with your own before/after images.

---

## How to Test Locally

1. Clone the repo:
```bash
git clone <your-repo-url>
cd <repo-folder>
