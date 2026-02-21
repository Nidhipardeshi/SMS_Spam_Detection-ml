from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]

    if prediction == 1:
        result = "Spam"
    else:
        result = "Ham (Not Spam)"

    return render_template(
        "index.html",
        prediction_text=f"Prediction: {result}",
        original_message=message
    )


if __name__ == "__main__":
    app.run(debug=True)