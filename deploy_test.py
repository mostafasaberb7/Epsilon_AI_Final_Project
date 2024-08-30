
import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("model.pkl")

# Home page
def home_page():
    st.title("Fatality Prediction")
    st.write("Welcome to the Fatality Prediction App. This app predicts the likelihood of a fatal accident based on various features.")
    st.write("Use the navigation on the left to go to the prediction page.")

# Prediction page
def predict_page():
    st.title("Make a Prediction")
    
    # User inputs for the prediction
    district = st.selectbox("District", options=[0, 1, 2, 3, 4], index=0)
    visibility = st.selectbox("Visibility", options=[0, 1, 2, 3], index=0)
    light = st.selectbox("Light", options=[0, 1], index=0)
    initdir = st.selectbox("Initial Direction", options=[0, 1, 2, 3], index=0)
    pedestrian = st.selectbox("Pedestrian Involved", options=[0, 1], index=0)
    cyclist = st.selectbox("Cyclist Involved", options=[0, 1], index=0)
    automobile = st.selectbox("Automobile Involved", options=[0, 1], index=0)
    motorcycle = st.selectbox("Motorcycle Involved", options=[0, 1], index=0)
    truck = st.selectbox("Truck Involved", options=[0, 1], index=0)
    trsn_city_veh = st.selectbox("City Vehicle Involved", options=[0, 1], index=0)
    emerg_veh = st.selectbox("Emergency Vehicle Involved", options=[0, 1], index=0)
    passenger = st.selectbox("Passenger Involved", options=[0, 1], index=0)
    speeding = st.selectbox("Speeding Involved", options=[0, 1], index=0)
    ag_driv = st.selectbox("Aggressive Driving", options=[0, 1], index=0)
    redlight = st.selectbox("Redlight Violation", options=[0, 1], index=0)
    alcohol = st.selectbox("Alcohol Involved", options=[0, 1], index=0)
    disability = st.selectbox("Disability Involved", options=[0, 1], index=0)
    day_status = st.selectbox("Day Status", options=[0, 1, 2, 3], index=0)
    st.write("### Input Values:")
    st.write(f"District: {district}")
    st.write(f"Visibility: {visibility}")
    st.write(f"Light: {light}")
    st.write(f"Initial Direction: {initdir}")
    st.write(f"Pedestrian: {pedestrian}")
    st.write(f"Cyclist: {cyclist}")
    st.write(f"Automobile: {automobile}")
    st.write(f"Motorcycle: {motorcycle}")
    st.write(f"Truck: {truck}")
    st.write(f"Transit City Vehicle: {trsn_city_veh}")
    st.write(f"Emergency Vehicle: {emerg_veh}")
    st.write(f"Passenger: {passenger}")
    st.write(f"Speeding: {speeding}")
    st.write(f"Aggressive Driving: {ag_driv}")
    st.write(f"Redlight Violation: {redlight}")
    st.write(f"Alcohol: {alcohol}")
    st.write(f"Disability: {disability}")
    st.write(f"Day Status: {day_status}")

    # Button to make the prediction
    if st.button("Predict"):
        # Create the feature array
        features = np.array([[district, visibility, light, initdir, pedestrian, cyclist, automobile,
                              motorcycle, truck, trsn_city_veh, emerg_veh, passenger, speeding,
                              ag_driv, redlight, alcohol, disability, day_status]])
        
        # Make the prediction
        prediction = model.predict(features)
        
        # Store the prediction result in the session state
        st.session_state.prediction = prediction[0]
        
        # Set session state to move to result page
        st.session_state.page = "Result"

# Result page
def result_page():
    st.title("Prediction Result")
    
    # Check if a prediction exists in the session state
    if 'prediction' in st.session_state:
        prediction = st.session_state.prediction
        st.write(f"The predicted fatality status is: {'Fatal' if prediction == 1 else 'Non-Fatal'}")
    else:
        st.write("No prediction made yet. Please go to the prediction page.")

# Main function to control the app
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict", "Result"], index=["Home", "Predict", "Result"].index(st.session_state.page))

    if page == "Home":
        st.session_state.page = "Home"
        home_page()
    elif page == "Predict":
        st.session_state.page = "Predict"
        predict_page()
    elif page == "Result":
        st.session_state.page = "Result"
        result_page()

if __name__ == "__main__":
    main()











