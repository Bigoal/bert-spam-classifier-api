# Spam Classifier

An end-to-end spam classification system that classifies text messages as spam or ham using both classical NLP techniques (TF-IDF + NLTK) and modern transformer-based models (BERT). The project features a FastAPI web application deployed on huggingface for real-time predictions.

This project includes:

- **TF-IDF + NLTK** for classical text preprocessing and classification
- **BERT** fine-tuning for modern language understanding
- **FastAPI** for model inference and web deployment
- **Docker** for containerized deployment
- **Render** for deployment of the web application

---
## Live Demo

You can try the application here:

🔗 https://huggingface.co/spaces/Bigoal1/spam-bert-classifier-api

---

## Project Overview

The goal of this project is to classify a text message as:

- **spam**
- **ham**

The project was built using two approaches:

1. **Classical NLP**
   - Text preprocessing with NLTK
   - TF-IDF vectorization
   - Machine learning classification

2. **Transformer-based NLP**
   - BERT fine-tuning
   - Saved trained model for inference
   - Integrated into a FastAPI web app

The currently deployed web application uses the saved **BERT model** for predictions.

---

## Features

- Text preprocessing with NLTK
- TF-IDF-based baseline model
- Fine-tuned BERT model
- FastAPI web application
- HTML-based prediction page
- Confidence score returned with predictions
- Docker support for deployment
- huggingface deployment

---

## Tech Stack

### NLP / Machine Learning
- Python
- NLTK
- scikit-learn
- TF-IDF
- PyTorch
- Hugging Face Transformers
- BERT

### Web Application
- FastAPI
- HTML

### Deployment
- Docker
- huggingface

---

## Model Performance

### BERT Model Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.990135 |
| F1 Score | 0.962457 |
| Precision | 0.979167 |
| Recall | 0.946309 |

---

## Project Structure

```bash
spam-classifier/
│
├── static/
│   └── photo.jpg
│
├── spam_classifier/        # Saved BERT model and tokenizer
├── notebooks/
│   ├── NLTK.ipynb
│   └── BERT.ipynb
│
├── main.py                 # FastAPI app
├── Dockerfile
├── requirements.txt
└── README.md
