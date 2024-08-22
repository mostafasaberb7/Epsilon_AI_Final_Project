# import streamlit as st


# import joblib

# import pandas as pd
# import streamlit as st
# import seaborn as sns
# import plotly.express as px
# import matplotlib.pyplot as plt

# df = pd.read_csv('KSI_CLEAN.csv')
# model = joblib.load("model.pkl")

# st.set_page_config("Best Website Ever!", layout='wide')

# def predict(District,VISIBILITY,LIGHT,INITDIR,PEDESTRIAN,CYCLIST,AUTOMOBILE,MOTORCYCLE,TRUCK,TRSN_CITY_VEH,EMERG_VEH,PASSENGER,SPEEDING,AG_DRIV,REDLIGHT,ALCOHOL,DISABILITY,Day_Status):
#     pred = model.predict([[District,VISIBILITY,LIGHT,INITDIR,PEDESTRIAN,CYCLIST,AUTOMOBILE,MOTORCYCLE,TRUCK,TRSN_CITY_VEH,EMERG_VEH,PASSENGER,SPEEDING,AG_DRIV,REDLIGHT,ALCOHOL,DISABILITY,Day_Status]])
#     plots(pred)

# def home_page():
#     # Display text

#     st.title("Epsilon Deployment!")
#     st.header("This is a header")
#     st.subheader("This is a subheader")

#     st.markdown("**This is written using markdown**")
#     st.markdown("This is normal markdown")

#     st.caption("This is a caption")
#     st.code("import requests\nrequests.post(<URL>, json={<DATA>})")


#     st.markdown("<a href=\"https://www.google.com\">Click me!</a>", unsafe_allow_html=True)

#     # Display media
#     st.image("https://d3544la1u8djza.cloudfront.net/APHI/Blog/2021/07-06/small+white+fluffy+dog+smiling+at+the+camera+in+close-up-min.jpg", width=200)

#     # st.audio()
#     # st.video()

#     # Dataframes
# def inputs():
#     # Inputs
#     st.title("Feature Input App")

#     # District input
#     district = st.selectbox(
#         'District', 
#         options=[4, 0, 3, 2, 1],
#         index=0
#     )

#     # VISIBILITY input
#     visibility = st.selectbox(
#         'Visibility', 
#         options=[0, 1, 2, 3],
#         index=0
#     )

#     # LIGHT input
#     light = st.selectbox(
#         'Light', 
#         options=[1, 0],
#         index=0
#     )

#     # INITDIR input
#     initdir = st.selectbox(
#         'Initial Direction', 
#         options=[0, 3, 2, 1],
#         index=0
#     )

#     # PEDESTRIAN input
#     pedestrian = st.selectbox(
#         'Pedestrian Involved', 
#         options=[0, 1],
#         index=0
#     )

#     # CYCLIST input
#     cyclist = st.selectbox(
#         'Cyclist Involved', 
#         options=[0, 1],
#         index=0
#     )

#     # AUTOMOBILE input
#     automobile = st.selectbox(
#         'Automobile Involved', 
#         options=[1, 0],
#         index=0
#     )

#     # MOTORCYCLE input
#     motorcycle = st.selectbox(
#         'Motorcycle Involved', 
#         options=[0, 1],
#         index=0
#     )

#     # TRUCK input
#     truck = st.selectbox(
#         'Truck Involved', 
#         options=[0, 1],
#         index=0
#     )

#     # TRSN_CITY_VEH input
#     trsn_city_veh = st.selectbox(
#         'Transit City Vehicle Involved', 
#         options=[0, 1],
#         index=0
#     )

#     # EMERG_VEH input
#     emerg_veh = st.selectbox(
#         'Emergency Vehicle Involved', 
#         options=[0, 1],
#         index=0
#     )

#     # PASSENGER input
#     passenger = st.selectbox(
#         'Passenger Involved', 
#         options=[0, 1],
#         index=0
#     )

#     # SPEEDING input
#     speeding = st.selectbox(
#         'Speeding Involved', 
#         options=[0, 1],
#         index=0
#     )

#     # AG_DRIV input
#     ag_driv = st.selectbox(
#         'Aggressive Driving Involved', 
#         options=[1, 0],
#         index=0
#     )

#     # REDLIGHT input
#     redlight = st.selectbox(
#         'Redlight Violation', 
#         options=[0, 1],
#         index=0
#     )

#     # ALCOHOL input
#     alcohol = st.selectbox(
#         'Alcohol Involved', 
#         options=[0, 1],
#         index=0
#     )

#     # DISABILITY input
#     disability = st.selectbox(
#         'Disability Involved', 
#         options=[0, 1],
#         index=0
#     )

#     # Day_Status input
#     day_status = st.selectbox(
#         'Day Status', 
#         options=[0, 2, 1, 3],
#         index=0
#     )

#     # Show the input values
#     st.write("### Input Values:")
#     st.write(f"District: {district}")
#     st.write(f"Visibility: {visibility}")
#     st.write(f"Light: {light}")
#     st.write(f"Initial Direction: {initdir}")
#     st.write(f"Pedestrian: {pedestrian}")
#     st.write(f"Cyclist: {cyclist}")
#     st.write(f"Automobile: {automobile}")
#     st.write(f"Motorcycle: {motorcycle}")
#     st.write(f"Truck: {truck}")
#     st.write(f"Transit City Vehicle: {trsn_city_veh}")
#     st.write(f"Emergency Vehicle: {emerg_veh}")
#     st.write(f"Passenger: {passenger}")
#     st.write(f"Speeding: {speeding}")
#     st.write(f"Aggressive Driving: {ag_driv}")
#     st.write(f"Redlight Violation: {redlight}")
#     st.write(f"Alcohol: {alcohol}")
#     st.write(f"Disability: {disability}")
#     st.write(f"Day Status: {day_status}")

    
#     st.button("Predict", on_click=predict, args=(district,visibility,light,initdir,pedestrian,cyclist,automobile,motorcycle,truck,trsn_city_veh,emerg_veh,passenger,speeding,ag_driv,redlight,alcohol,disability,day_status))
    
#     #     st.date_input("Built Year")

#     #     st.checkbox("Furnished")
#     #     st.checkbox("Amenities")

#     #     st.radio("Finishing", ['Finishing 1', 'Finishing 2', 'Finishing 3'])
#     #     st.select_slider("Another Finishing", ['Finishing 1', 'Finishing 2', 'Finishing 3'])
#     #     st.selectbox("Yet Another Finishing", ['Finishing 1', 'Finishing 2', 'Finishing 3'])

#     #     st.slider("Price", 1000,10000)

# def plots(result=0):
#     # Plotting
#     st.title("Result")
    
#     st.markdown(f"Estimated Fare: **{result}**")
    
#     # fig = plt.figure()
#     # sns.histplot(df, x='fare_amount')
    
#     # st.pyplot(fig)

#     # st.button("Make Another Prediction", on_click=inputs)
    
#     #     st.header("Matplotlib (Seaborn)")
#     #     df = sns.load_dataset("penguins")

#     #     fig = plt.figure(figsize=(4,4))
#     #     sns.scatterplot(data=df, x='bill_length_mm', y='bill_depth_mm', hue='species')

#     #     st.pyplot(fig)

#     #     st.header("Plotly")

#     #     st.caption("Using Plotly")
#     #     fig = px.scatter(df, x='bill_length_mm', y='bill_depth_mm', color='species')
#     #     st.plotly_chart(fig)

#     #     st.caption("Using Streamlit")
#     #     st.scatter_chart(df, x='bill_length_mm', y='bill_depth_mm', color='species')


# page = st.sidebar.selectbox("Select page", ["Home", "Predict", "Plots"])

# if page == 'Home':
#     home_page()
# elif page == 'Predict':
#     inputs()
# else:
#     plots()



# # recipes = ['Spaghetti Carbonara', 'Chicken Tikka Masala', 'Tacos', 'Pancakes']

# # # Selectbox to choose a recipe
# # selected_recipe = st.selectbox('Select a recipe', recipes)

# # # Display the selected recipe
# # st.write(f'You selected: {selected_recipe}')









# import streamlit as st
# import pickle
# import numpy as np
# import joblib


# # Debugging: Print statements to check model loading
# model = joblib.load("model.pkl")


# # Home page
# def home_page():
#     st.title("Fatality Prediction")
#     st.write("Welcome to the Fatality Prediction App. This app predicts the likelihood of a fatal accident based on various features.")
#     st.write("Use the navigation on the left to go to the prediction page.")

# # Prediction page
# def predict_page():
#     st.title("Make a Prediction")
    
#     # User inputs for the prediction
#     district = st.selectbox("District", options=[0, 1, 2, 3, 4], index=0)
#     visibility = st.selectbox("Visibility", options=[0, 1, 2, 3], index=0)
#     light = st.selectbox("Light", options=[0, 1], index=0)
#     initdir = st.selectbox("Initial Direction", options=[0, 1, 2, 3], index=0)
#     pedestrian = st.selectbox("Pedestrian Involved", options=[0, 1], index=0)
#     cyclist = st.selectbox("Cyclist Involved", options=[0, 1], index=0)
#     automobile = st.selectbox("Automobile Involved", options=[0, 1], index=0)
#     motorcycle = st.selectbox("Motorcycle Involved", options=[0, 1], index=0)
#     truck = st.selectbox("Truck Involved", options=[0, 1], index=0)
#     trsn_city_veh = st.selectbox("City Vehicle Involved", options=[0, 1], index=0)
#     emerg_veh = st.selectbox("Emergency Vehicle Involved", options=[0, 1], index=0)
#     passenger = st.selectbox("Passenger Involved", options=[0, 1], index=0)
#     speeding = st.selectbox("Speeding Involved", options=[0, 1], index=0)
#     ag_driv = st.selectbox("Aggressive Driving", options=[0, 1], index=0)
#     redlight = st.selectbox("Redlight Violation", options=[0, 1], index=0)
#     alcohol = st.selectbox("Alcohol Involved", options=[0, 1], index=0)
#     disability = st.selectbox("Disability Involved", options=[0, 1], index=0)
#     day_status = st.selectbox("Day Status", options=[0, 1, 2, 3], index=0)

#     # Button to make the prediction
#     if st.button("Predict"):
#         # Create the feature array
#         features = np.array([[district, visibility, light, initdir, pedestrian, cyclist, automobile,
#                               motorcycle, truck, trsn_city_veh, emerg_veh, passenger, speeding,
#                               ag_driv, redlight, alcohol, disability, day_status]])
        
#         # Make the prediction
#         prediction = model.predict(features)
#         result_page(prediction[0])

# # Result page
# def result_page(prediction):
#     st.title("Prediction Result")
#     st.write(f"The predicted fatality status is: {'Fatal' if prediction == 1 else 'Non-Fatal'}")
#     st.write("Use the navigation on the left to make another prediction.")

# # Main function to control the app
# def main():
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to", ["Home", "Predict", "Result"])

#     if page == "Home":
#         home_page()
#     elif page == "Predict":
#         predict_page()
#     else:
#         st.write("Please make a prediction first.")

# if __name__ == "__main__":
#     main()



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











