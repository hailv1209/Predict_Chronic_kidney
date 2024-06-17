import numpy as np
import pickle
import streamlit as st

# Load the model and scaler
with open("model/model.pkl", 'rb') as model_file, open("model/scaler.pkl", 'rb') as scaler_file:
    Chronic_Kidney_model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Creating a function for prediction
def kidney_prediction(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    # Standardize the input data using the loaded scaler
    std_data = scaler.transform(input_data_reshaped)
    # Make the prediction
    prediction = Chronic_Kidney_model.predict(std_data)
    print(prediction)
    if prediction[0] == 1:
        return "The person has NOT Chronic Kidney"
    else:
        return "The person has Chronic Kidney"

def main():
    # Giving Title
    st.title("Chronic Kidney Disease Prediction")

    # Getting input data from the user
    WBC = st.number_input("White Blood Cell Count", step=1.0, format="%.2f")
    BU = st.number_input("Blood Urea", step=1.0, format="%.2f")
    BGR = st.number_input("Blood Glucose Random", step=1.0, format="%.2f")
    SC = st.number_input("Serum Creatinine", step=1.0, format="%.2f")
    PCV = st.number_input("Packed Cell Volume", step=1.0, format="%.2f")
    AN = st.number_input("Albumin", step=1.0, format="%.2f")
    HN = st.number_input("Hemoglobin", step=1.0, format="%.2f")
    Age = st.number_input("Age", step=1.0, format="%.2f")
    Sugar = st.number_input("Sugar", step=1.0, format="%.2f")
    Hypertension = st.selectbox("Hypertension", options=["Yes", "No"])
    
    Hypertension_value = 1 if Hypertension == "Yes" else 0

    # Creating button for Prediction
    if st.button("Kidney Test Result"):
        # Code for prediction
        diagnosis = kidney_prediction([WBC, BU, BGR, SC, PCV, AN, HN, Age, Sugar, Hypertension_value])

        # Displaying the result
        if diagnosis == "The person has NOT Chronic Kidney":
            st.success(diagnosis)
        else:
            st.error(diagnosis)

if __name__ == "__main__":
    main()
