from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load Model
pipe = pipeline(
    "text-classification",
    model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa",
    return_all_scores=False
)

# Mapping label → value
mapping = {
    "Negative": 1,
    "Neutral": 2,
    "Positive": 3
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sentiment API is running on Railway"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if "text" not in data:
        return jsonify({"error": "Text is required"}), 400

    raw = pipe(data["text"])[0]

    sentiment = raw["label"]
    value = mapping.get(sentiment, 0)

    return jsonify({
        "sentiment": sentiment,
        "value": value,
        "score": raw["score"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
