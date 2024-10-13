from flask import Flask, request, jsonify, render_template
import joblib  # Or pickle if you prefer
import numpy as np

app = Flask(__name__)

# Load the saved KMeans model
model = joblib.load('kmeans_model.pkl')

@app.route('/')
def home():
    # Renders the homepage (index.html)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    age = request.form['age']
    income = request.form['income']
    spending_score = request.form['spending_score']
    
    # Convert inputs to the format required by the model
    input_data = np.array([[age, income, spending_score]], dtype=float)
    
    # Predict the cluster
    prediction = model.predict(input_data)
    
    # Send back the result
    return jsonify({"cluster": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
