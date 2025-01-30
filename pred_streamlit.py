import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the pre-trained Random Forest model
with open("best_random_forest_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Load the scaler used for scaling the input features
with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

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

# Function to calculate prediction confidence interval
def predict_confidence(model, X, confidence=0.95):
    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
    mean_prediction = tree_predictions.mean(axis=0)
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    lower_bound = np.percentile(tree_predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(tree_predictions, upper_percentile, axis=0)
    return mean_prediction[0], lower_bound[0], upper_bound[0]

# Set the title of the page (appears in the browser tab)
st.set_page_config(page_title="Insurance Premium Prediction", page_icon=":moneybag:", layout="wide")

# Header Image
st.image("insurancepredictimage.png", width=700)  # Replace with your own image path or URL

# Main title and description
st.title("Health Insurance Premium Price Prediction")
st.info("Welcome to the **Health Insurance Premium Prediction App**! Provide your details below to predict your premium.")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:

    Age = st.number_input("Age", min_value=1, max_value=100, value=30)
    Height = st.number_input("Height (in cm)", min_value=1, max_value=250, value=140)
    Weight = st.number_input("Weight (in kg)", min_value=1, max_value=200, value=50)
    NumberOfMajorSurgeries = st.number_input("Number of major surgeries undergone", min_value=0, max_value=10, value=0)

with col2:

    Diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])
    BloodPressureProblems = st.selectbox("Do you have blood pressure problems?", ["No", "Yes"])
    AnyTransplants = st.selectbox("Do you have any transplants?", ["No", "Yes"])
    AnyChronicDiseases = st.selectbox("Do you have any chronic diseases?", ["No", "Yes"])
    HistoryOfCancerInFamily = st.selectbox("History of cancer in family?", ["No", "Yes"])

# Convert categorical inputs to numerical values
Diabetes = 1 if Diabetes == "Yes" else 0
BloodPressureProblems = 1 if BloodPressureProblems == "Yes" else 0
AnyTransplants = 1 if AnyTransplants == "Yes" else 0
AnyChronicDiseases = 1 if AnyChronicDiseases == "Yes" else 0
HistoryOfCancerInFamily = 1 if HistoryOfCancerInFamily == "Yes" else 0

# Create the query array from user input
query = [
    Age, 
    Diabetes, 
    BloodPressureProblems, 
    AnyTransplants, 
    AnyChronicDiseases, 
    Height, 
    Weight, 
    HistoryOfCancerInFamily, 
    NumberOfMajorSurgeries
]

# Add a container for the predict button and use st.button for it to appear in the center.
with st.container():
    st.write("Click to Predict")
    if st.button("Predict Insurance Premium"):
        # Convert query list to a DataFrame with proper column names
        query_df = pd.DataFrame([query], columns=columns)

        # Scale the input data using the fitted scaler
        scaled_input = scaler.transform(query_df)

        # Prepare the input for prediction
        features_array = np.array(scaled_input).reshape(1, -1)

        # Make the prediction and get the confidence interval
        result = model.predict(features_array)
        mean_pred, lower_ci, upper_ci = predict_confidence(model, features_array)

        # Display the prediction results when the button is clicked
        st.subheader("Prediction Result:")
        st.markdown(f"**Predicted Insurance Premium Price**: **Rs. {round(result[0], 2)}**")
        st.markdown(f"**95% Prediction Confidence Interval**: **(Rs. {round(lower_ci, 2)}, Rs. {round(upper_ci, 2)})**")

        # Option to download the prediction as a report
        report_content = f"""
        Predicted Insurance Price: Rs. {round(result[0], 2)}
        95% Prediction Interval: (Rs. {round(lower_ci, 2)}, Rs. {round(upper_ci, 2)})

        Inputs:
        - Age: {Age}
        - Height (in cm): {Height}
        - Weight (in kg): {Weight}
        - Number of major surgeries undergone: {NumberOfMajorSurgeries}
        - Do you have diabetes?: {"Yes" if Diabetes == 1 else "No"}
        - Do you have blood pressure problems?: {"Yes" if BloodPressureProblems == 1 else "No"}
        - Do you have any transplants?: {"Yes" if AnyTransplants == 1 else "No"}
        - Do you have any chronic diseases?: {"Yes" if AnyChronicDiseases == 1 else "No"}
        - History of cancer in family?: {"Yes" if HistoryOfCancerInFamily == 1 else "No"}
        """

        # Option to download the prediction as a report
        st.download_button(
            label="Download Prediction Report",
            data=report_content,
            file_name="insurance_prediction_report.txt",
            mime="text/plain"
        )