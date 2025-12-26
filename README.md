# 🇳🇬 Nigeria Wikipedia Sentiment Analysis

A **Streamlit-based Sentiment Analysis application** that predicts the sentiment of user-input text using a **Random Forest Classifier** trained on **Nigeria-related Wikipedia content**.  
The app combines **machine learning predictions**, **TextBlob sentiment scores**, and **visual analytics** for better interpretability.

---

## 📌 Project Overview

This project demonstrates how **Natural Language Processing (NLP)** and **Machine Learning** can be applied to analyze sentiment in textual data.  
The model is trained on text extracted from **Wikipedia articles about Nigeria**, making it domain-specific and context-aware.

Users can input any sentence and instantly receive:
- Sentiment prediction (Positive / Negative)
- Prediction confidence
- Word cloud visualization
- TextBlob polarity and subjectivity insights

---

## 🚀 Features

### 🔍 Machine Learning Sentiment Prediction
- Uses a **Random Forest Classifier**
- Text transformed using **TF-IDF Vectorization**
- Outputs sentiment label with confidence scores

### ☁️ Word Cloud Visualization
- Generates a word cloud from user input
- Highlights dominant words in the sentence

### 🧠 Dual Sentiment Analysis
- **ML-based sentiment** (Random Forest)
- **Rule-based sentiment** (TextBlob)

### 📊 Visual Analytics
- Probability bar chart for sentiment confidence
- Polarity & subjectivity visualization using Seaborn

### 🖥️ Interactive UI
- Built using **Streamlit**
- Simple, clean, and user-friendly interface

---

## 🏗️ Tech Stack

- **Frontend / App Framework**: Streamlit  
- **Machine Learning**: Scikit-learn (Random Forest)  
- **NLP**: TextBlob, NLTK  
- **Vectorization**: TF-IDF  
- **Visualization**: Matplotlib, Seaborn, WordCloud  
- **Model Persistence**: Joblib  

---

## 📁 Project Structure

```bash
├── app.py                            # Streamlit application
├── Nigeria.ipynb                     # Data processing & model training notebook
├── random_forest_sentiment_model.pkl # Trained Random Forest model
├── tfidf_vectorizer.pkl              # TF-IDF vectorizer
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

⚙️ Installation & Setup
1️⃣ Clone the Repository

```
git clone https://github.com/your-username/nigeria-sentiment-analysis.git
cd nigeria-sentiment-analysis
```

2️⃣ Create a Virtual Environment (Recommended)
```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```
3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
4️⃣ Run the Streamlit App
```
streamlit run app.py
```

## 🧪 Model & NLP Pipeline

### 📚 Data Source
- Wikipedia articles related to **Nigeria**

### 🧹 Preprocessing
- Text cleaning
- Tokenization
- Stopword removal

### 🧩 Feature Extraction
- TF-IDF Vectorization

### ⚖️ Class Imbalance Handling
- SMOTE (Synthetic Minority Oversampling Technique)

### 🤖 Model
- Random Forest Classifier

### 📈 Evaluation
- Prediction probabilities
- Sentiment confidence visualization

---

## 📊 Output Interpretation

- **Sentiment Label**: Positive / Negative  
- **Prediction Confidence (%)**
- **TextBlob Polarity**
  - Range: `-1` (Negative) to `+1` (Positive)
- **TextBlob Subjectivity**
  - Range: `0` (Objective) to `1` (Subjective)

---

## 🎯 Project Objective

To build a **domain-specific sentiment analysis system** that:

- Applies machine learning techniques to real-world textual data  
- Compares ML-based predictions with rule-based sentiment analysis  
- Provides clear, visual, and interpretable results through an interactive UI  

---
 🌐 Live Demo

Streamlit Application
https://nigeriawiki-32vbvf2bapprnpuktfqqphj.streamlit.app/
---

## 👨‍💻 Author

**Sainath**  
B.Tech – Computer Science & Engineering (Data Science)

- NLP & Machine Learning  
- Streamlit Applications  
- Data Analytics  

---

## 📜 License

This project is developed for **academic and educational purposes**.  
The Wikipedia content used follows **Wikipedia’s content usage policies**.

