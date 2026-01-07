from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model pipeline (TF-IDF + Model)
model = joblib.load("pipeline.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    text_input = ""

    if request.method == "POST":
        text_input = request.form.get("text", "")

        if text_input.strip():
            # Prediksi label
            pred = model.predict([text_input])[0]

            # Confidence (karena pakai MultinomialNB)
            prob = model.predict_proba([text_input]).max()
            confidence = round(prob * 100, 2)

            prediction = "valid" if pred == 1 else "hoax"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        text=text_input
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


