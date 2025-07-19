# 🛡️ Hate Speech Detection Web App

> 📌 The app uses a Scikit-learn pipeline with `TfidfVectorizer` and `SGDClassifier` to perform hate speech detection on textual data. The model is trained using balanced data via upsampling to handle class imbalance. The entire machine learning pipeline—from preprocessing to prediction—is modular and easy to maintain. It supports seamless integration of different vectorizers (e.g., `CountVectorizer`) or classifiers (e.g., `LogisticRegression`, `RandomForestClassifier`) for experimentation. A lightweight and interactive `Streamlit` interface enables users to test the model in real time, making the solution practical and user-friendly for exploratory use.



## 🛠️ Built With

- **Pandas** – Data manipulation and analysis
- **Scikit-learn** – Machine Learning (SGDClassifier, train/test split, TF-IDF, pipeline)
- **Streamlit** – Frontend web app for interactive UI
- **Joblib** – Model persistence
- **Numpy** – Numerical operations


## ✅ Features

- Real-time text input prediction
- Trained ML pipeline (SGD + TF-IDF)
- Simple, intuitive Streamlit UI
- Exported & reloaded model using `joblib`

---
✅ Install Dependencies
```
pip install streamlit scikit-learn joblib numpy
```
🚀 Running the App
```
streamlit run app.py
