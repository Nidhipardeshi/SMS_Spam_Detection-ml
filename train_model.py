import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


# ==============================
# 1. Ensure model folder exists
# ==============================

if not os.path.exists("model"):
    os.makedirs("model")


# ==============================
# 2. Load Dataset
# ==============================

data_path = r"data/spam.csv"
df = pd.read_csv(data_path, encoding="latin-1")

print("Dataset Loaded Successfully")
print("Columns:", df.columns)


# ==============================
# 3. Clean Dataset
# ==============================

# If dataset has v1, v2 format
if "v1" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

# Remove missing values
df["message"] = df["message"].fillna("")
df["label"] = df["label"].fillna("")

# Remove duplicates
df = df.drop_duplicates()

print("Dataset shape after cleaning:", df.shape)


# ==============================
# 4. Label Encoding
# ==============================

df["label"] = df["label"].map({"ham": 0, "spam": 1})


# ==============================
# 5. Train-Test Split
# ==============================

X = df["message"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 6. TF-IDF Vectorization
# ==============================

vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Vectorization Done")


# ==============================
# 7. Model Training
# ==============================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)

print("Model Training Completed")


# ==============================
# 8. Evaluation
# ==============================

y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 9. Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ==============================
# 10. ROC Curve
# ==============================

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()


# ==============================
# 11. Save Model & Vectorizer
# ==============================

pickle.dump(model, open("model/spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/tfidf.pkl", "wb"))

print("\nModel & Vectorizer Saved in /model folder")
print("Training Completed Successfully")