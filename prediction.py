import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the pre-trained model and scaler
with open("best_random_forest_model.pkl", 'rb') as f:
    model = pickle.load(f)

with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Define the column names used during model training
columns = [
    'Age', 
    'Diabetes', 
    'BloodPressureProblems', 
    'AnyTransplants', 
    'AnyChronicDiseases', 
    'Height',
    'Weight', 
    'HistoryOfCancerInFamily', 
    'NumberOfMajorSurgeries'
]

# Sample home route
@app.route("/")
def hello_world():
    return "<p>Hello, Welcome to our Insurance premium price prediction Application</p>"

# Health check route
@app.route("/ping", methods=["GET"])
def ping():
    return "<p>Ping successful!</p>"

# Prediction route
@app.route("/predict", methods=['POST'])
def prediction():
    try:
        # Get the input JSON from the request
        ins_req = request.get_json()
        
        # Extract input features from the JSON payload
        query = [
            ins_req["Age"], 
            ins_req["Diabetes"], 
            ins_req["BloodPressureProblems"], 
            ins_req["AnyTransplants"], 
            ins_req["AnyChronicDiseases"], 
            ins_req["Height"],
            ins_req["Weight"], 
            ins_req["HistoryOfCancerInFamily"], 
            ins_req["NumberOfMajorSurgeries"]
        ]
        
        # Convert query list to a DataFrame with proper column names
        query_df = pd.DataFrame([query], columns=columns)

        # Scale the input data using the fitted scaler
        scaled_input = scaler.transform(query_df)

        # Convert the scaled input into a numpy array and reshape for prediction
        features_array = np.array(scaled_input).reshape(1, -1)

        # Function to calculate prediction confidence intervals
        def predict_confidence(model, X, confidence=0.95):
            # Get predictions from each tree in the random forest
            tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
            mean_prediction = tree_predictions.mean(axis=0)
            lower_percentile = (1 - confidence) / 2 * 100
            upper_percentile = (1 + confidence) / 2 * 100
            lower_bound = np.percentile(tree_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(tree_predictions, upper_percentile, axis=0)
            return mean_prediction[0], lower_bound[0], upper_bound[0]

        # Make the prediction using the trained model
        result = model.predict(features_array)

        # Calculate confidence intervals
        mean_pred, lower_ci, upper_ci = predict_confidence(model, features_array)

        # Return the prediction result as a JSON response
        return jsonify({
            "Predicted insurance price": round(result[0], 2),
            "95% Prediction Interval": (round(lower_ci, 2), round(upper_ci, 2))
        })
    
    except Exception as e:
        # Handle any error and return it as a JSON response
        return jsonify({"error": str(e)}), 400

# Run the Flask app on a specific port
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
