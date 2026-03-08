# FoodVision – Deep Learning Food Image Inference Platform

FoodVision is an end-to-end **Machine Learning inference platform** designed to automatically analyze and classify food images into **101 different food categories**.

The system integrates a **deep learning model**, a **web-based user interface**, and a **secure RESTful API service**. It demonstrates a complete mini **MLOps pipeline**, including model serving, API integration, and containerized deployment.

The platform allows users to upload food images and obtain predictions along with confidence scores in real time.

---

# 1. System Architecture

The system is composed of several core components:

**AI Engine**
- PyTorch
- Torchvision

**Model Architecture**
- Convolutional Neural Network (CNN)
- EfficientNet-B2
- Transfer Learning

**Backend Framework**
- Django
- Django REST Framework (DRF)

**Frontend**
- HTML5
- CSS3 (Minimal card-based UI)
- Vanilla JavaScript

**Deployment Environment**
- Docker
- Gunicorn
- Hugging Face Spaces

---

# 2. Core Features

### Web Interface
A clean and minimal interface allows users to upload images using a **Drag & Drop mechanism**.

Client-side validation automatically checks:

- file size (maximum 5MB)
- file format

This prevents unnecessary requests and reduces server load.

---

### REST API Service

FoodVision exposes a **prediction endpoint** so external systems can send images and receive classification results in JSON format.

This makes the model usable for:

- web applications
- mobile apps
- research experiments
- AI microservices

---

### JWT Authentication

API access is protected using **JSON Web Tokens (JWT)**.

Each client must first obtain an **access token**, which must be included in subsequent inference requests.

This ensures:

- authentication
- secure API access
- controlled model usage

---

### Lazy Model Loading

The deep learning model is loaded into memory **only once when the server starts**.

This design significantly reduces inference latency because:

- model weights are not reloaded for each request
- GPU/CPU initialization happens once

---

# 3. Local Development Setup

## Requirements

- Python 3.10 or Python 3.11
- Git

---

## Step 1 – Clone the repository

```bash
git clone https://github.com/your-username/FoodVisionBig.git
cd FoodVisionBig
```

## Step 2 – Create a virtual environment

```python
python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate
```

## Step 3 – Install dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4 – Configure environment variables
Create a *.env* file in the root directory:
```python
DJANGO_SECRET_KEY=your_random_secret_key
DJANGO_DEBUG=True
```

## Step 5 – Run the development server

```python
python manage.py collectstatic --noinput
python manage.py runserver
```

The API will be available at `http://localhost:8000/`.