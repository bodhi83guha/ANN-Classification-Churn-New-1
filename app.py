import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd
import numpy as np

## load trained model, scalar, onehot
model = tf.keras.models.load_model('model.h5')

## load scalar and onehot
with open('Label_Encoder_gender.pkl', 'rb') as file:
    Label_Encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file) 

with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)


## streamlit app
st.title('Customer Churn Prediction')

#User Input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', Label_Encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_product = st.slider('Number of Product', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Prepare the Input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_product],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]

}
)

# One Hot Encode Geo Data
geo_encoded = onehot_encoder_geo.transform([[input_data[geography]]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#concat Geo encoded data into inputdata
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis = 1)

#scale input data
input_data_scaled = scalar.transform(input_data)

#predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba >0.5:
    st.write ('Customer is likely to churn')
else:
    st.write ('Customer is not likely not churn')

