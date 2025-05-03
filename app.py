from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
with open("model1.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler1.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if not file:
        return "No file uploaded", 400

    df = pd.read_csv(file)

    # Drop unnecessary columns if present
    df = df.drop(columns=[col for col in ['nameOrig', 'nameDest', 'isFlaggedFraud'] if col in df.columns])

    # Encode type if needed
    if df['type'].dtype == 'object':
        df['type'] = df['type'].astype('category').cat.codes

    # Scale features
    df_scaled = scaler.transform(df)

    # Make predictions
    predictions = model.predict(df_scaled)
    prediction_labels = ['Fraud' if p == 1 else 'Not Fraud' for p in predictions]

    # Add predictions to DataFrame and save to CSV
    df['Prediction'] = prediction_labels
    df.to_csv("result.csv", index=False)

    # Display as HTML table
    result_html = df.to_html(classes='table table-striped', index=False)
    return render_template("result.html", titles="Fraud Prediction Table", tables=result_html)

if __name__ == "__main__":
    app.run(debug=True)
