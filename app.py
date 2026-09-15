import streamlit as st
import joblib
import pandas as pd

# Load the trained model
logi = joblib.load('logi.sav')

st.title('Delivery Delay Prediction')
st.write('Enter the values for the features to predict delivery delay:')

# Get feature names from the original DataFrame (assuming x was a DataFrame)
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                 'Warehouse_Processing_Time']

# Create input widgets for each feature
input_data = {}
for feature in feature_names:
    if feature in ['Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score']:
        input_data[feature] = st.number_input(f'{feature}:', min_value=0, max_value=20, value=1)
    elif feature in ['Driver_Experience']:
        input_data[feature] = st.number_input(f'{feature}:', min_value=0, max_value=30, value=5)
    elif feature in ['Warehouse_Processing_Time']:
        input_data[feature] = st.number_input(f'{feature}:', min_value=0, max_value=100, value=50)
    elif feature in ['Package_Weight']:
        input_data[feature] = st.number_input(f'{feature}:', min_value=0.0, max_value=200.0, value=10.0, format='%.2f')
    elif feature in ['Fuel_Efficiency']:
        input_data[feature] = st.number_input(f'{feature}:', min_value=0.0, max_value=50.0, value=15.0, format='%.2f')
    else: # Delivery_Distance
        input_data[feature] = st.number_input(f'{feature}:', min_value=0.0, max_value=100.0, value=20.0, format='%.2f')

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

if st.button('Predict Delivery Delay'):
    prediction = logi.predict(input_df)
    prediction_proba = logi.predict_proba(input_df)

    st.subheader('Prediction Results:')
    if prediction[0] == 1:
        st.error('Delivery is predicted to be Delayed!')
    else:
        st.success('Delivery is predicted to be On Time.')

    st.write(f'Probability of No Delay: {prediction_proba[0][0]:.2f}')
    st.write(f'Probability of Delay: {prediction_proba[0][1]:.2f}')
