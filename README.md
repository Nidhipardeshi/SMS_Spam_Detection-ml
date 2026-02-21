# SMS Spam Detection using Machine Learning

An end-to-end Machine Learning project that classifies SMS messages as **SPAM** or **HAM** using NLP techniques and a trained ML model.

The project includes model training, evaluation, and a Flask web application for real-time prediction.

---

##  Problem Statement

To build an automated system that detects spam SMS messages using Natural Language Processing and Machine Learning techniques.

---

##  Project Structure

SMS_Spam_Detection/
│
├── dataset/
│     └── spam.csv
│
├── notebooks/
│     └── eda.ipynb
│
├── models/
│     ├── spam_model.pkl
│     └── vectorizer.pkl
│
├── templates/
│     └── index.html
│
├── src/
│     └── train.py
│
├── app.py
├── requirements.txt
└── README.md

---

##  Model Details

- Feature Extraction: TF-IDF Vectorizer  
- Algorithm: Multinomial Naive Bayes / Logistic Regression  
- Text Preprocessing:
  - Lowercasing
  - Removing punctuation
  - Stopword removal

---

##  How to Run

### 1️ Install Dependencies
pip install -r requirements.txt

### 2️ Train the Model
cd src  
python train.py  

### 3️ Run Web App
python app.py  

Open:  
http://127.0.0.1:5000  

---

## 📈 Workflow

Data Cleaning → Feature Engineering → Model Training → Evaluation → Flask Deployment

---

## 🔮 Future Improvements

- Deep Learning models (LSTM / BERT)
- Cloud deployment
- Confidence score display

---

## 👩‍💻 Author

Nidhi Pardeshi
