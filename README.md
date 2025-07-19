# 🛡️ Hate Speech Detection Web App

> 📌 A lightweight Streamlit-based web application that detects hate speech in real time using a trained machine learning pipeline. The system leverages Scikit-learn’s `SGDClassifier` and `TfidfVectorizer` with class-balancing techniques to ensure robust predictions on textual data.

---

## 🛠️ Built With

- **Pandas** – Data manipulation and analysis  
- **Scikit-learn** – Machine Learning (SGDClassifier, TF-IDF, pipeline)  
- **Streamlit** – Frontend web app for interactive UI  
- **Joblib** – Model persistence  
- **Numpy** – Numerical operations  
- **re (Regular Expressions)** – Pattern matching for text preprocessing and polite word replacement  

---

## ✅ Features

- Real-time hate speech detection via web interface  
- Balanced training through upsampling to handle class imbalance  
- Clean, modular ML pipeline using `Pipeline` API  
- Downloadable model and vectorizer with `joblib`  
- Lightweight and easy to run via `Streamlit`  

---

## 💬 Politeness Enhancer

Offensive keywords are detected and replaced with softer or more respectful alternatives using a customizable dictionary and regular expressions.


---
✅ Install Dependencies
```
pip install streamlit scikit-learn joblib numpy
```
🚀 Running the App
```
streamlit run app.py
