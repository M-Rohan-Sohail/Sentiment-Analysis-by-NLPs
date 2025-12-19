# Sentiment Analysis on Real-World Product Reviews (TensorFlow)

This project implements a **binary sentiment analysis system** (Positive / Negative) on a **real-world product reviews dataset** using **TensorFlow and Keras**.  
The focus of this project is **practical NLP implementation**, including data cleaning, text vectorization, model training, and real-time inference.

---

## 📌 Project Overview

- Dataset: Product reviews with text summaries and full reviews
- Task: Binary sentiment classification (Positive / Negative)
- Framework: TensorFlow (Keras)
- Approach: N-gram based text vectorization + neural network

This project was built to **understand NLP fundamentals end-to-end**, starting from raw CSV data to a working sentiment prediction model.
DataSet Link **https://www.kaggle.com/datasets/mansithummar67/171k-product-review-with-sentiment-dataset**
---

## 🧠 Key Concepts Covered

- Real-world CSV data handling & encoding issues
- Text cleaning and normalization
- Label preprocessing and validation
- N-gram based tokenization
- Word embeddings
- Pooling strategies for NLP
- Binary classification
- Model evaluation and inference

---

## 📂 Dataset Description

The dataset contains the following relevant columns:

- **Summary** – Short review title
- **Review** – Full review text
- **Sentiment** – `positive`, `negative`, or `neutral`

Neutral samples are removed to convert the task into **binary classification**.

---

## 🧹 Data Preprocessing

Steps applied before training:

- Removed missing values
- Converted sentiment labels to lowercase
- Filtered only `positive` and `negative` samples
- Combined `Summary` and `Review` into a single text field
- Lowercased text
- Removed punctuation and special characters
- Removed empty text rows
- Mapped labels:
  - `positive → 1`
  - `negative → 0`

---

## 🔤 Text Vectorization

Text is converted into numerical form using:

- `TextVectorization` layer
- Vocabulary size: **10,000**
- Output sequence length: **120**
- **Bigrams (n-grams = 2)** to better capture sentiment patterns like  
  _"not good"_, _"very bad"_, etc.

```python
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=120,
    ngrams=2
)
