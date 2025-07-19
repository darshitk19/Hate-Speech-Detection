# 🛡️ Hate Speech Detection & Politeness Enhancer Web App

A Streamlit-powered web application that detects hate speech and enhances politeness in user-provided text. Built using a trained Scikit-learn pipeline with `SGDClassifier` and `TfidfVectorizer`.

---

## 🧠 Key Features

### ✅ Hate Speech Detection
Predicts whether each sentence contains hate speech using a trained ML pipeline.

### 💬 Politeness Enhancer
Offensive keywords are detected and replaced with respectful alternatives using a customizable replacement dictionary.

### 🧾 Multi-Sentence Support
Input multiple sentences separated by periods or newlines. Each sentence is analyzed independently.

### 📊 Word Contribution Insight
Identifies which words contributed most to classifying a sentence as hate or not.

### 🛠️ Real-time Predictions
Fully interactive Streamlit UI for testing and viewing results instantly.

---

## 🛠️ Built With

- **Streamlit** – For interactive web UI  
- **Scikit-learn** – Machine learning model (`SGDClassifier`, `TfidfVectorizer`)  
- **Joblib** – Model serialization  
- **NumPy** – Numerical operations  
- **Regex** – Text parsing and sanitization  



---
✅ Install Dependencies
```
pip install streamlit scikit-learn joblib numpy
```
🚀 Running the App
```
streamlit run app.py
