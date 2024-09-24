import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Load trained model (assuming a pre-trained model is available)
def load_model(encoded_data_path):
    # Load encoded data from CSV
    df = pd.read_csv(encoded_data_path)

    X = df.drop(columns=['bank_account'])  # Feature matrix
    y = df['bank_account']  # Target variable

    # Train RandomForestClassifier for the sake of this demo
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)

    return clf


# Load model
encoded_data_path = '/content/drive/MyDrive/Colab Notebooks/finance_data_encoded_data.csv'  # Replace with your path
clf = load_model(encoded_data_path)

# Streamlit app
st.title("Financial Inclusion Prediction App")

st.write("""
### Predict whether an individual has a bank account based on their characteristics.
""")

# Assuming you know the column names from your encoded DataFrame
feature_columns = ['location_type', 'cellphone_access', 'household_size', 'age_of_respondent',
                   'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']

# Create input fields for the features
location_type = st.selectbox("Location Type", feature_columns[0])
cellphone_access = st.selectbox("Cellphone Access", feature_columns[1])
household_size = st.slider("Household Size", min_value=1, max_value=20, value=3)
age_of_respondent = st.slider("Age of Respondent", min_value=16, max_value=100, value=30)
gender_of_respondent = st.selectbox("Gender", feature_columns[4])
relationship_with_head = st.selectbox("Relationship with Head", feature_columns[5])
marital_status = st.selectbox("Marital Status", feature_columns[6])
education_level = st.selectbox("Education Level", feature_columns[7])
job_type = st.selectbox("Job Type", feature_columns[8])

# Create a button for prediction
if st.button("Predict"):
    # Prepare the input data based on user selections
    input_data = pd.DataFrame({
        feature_columns[0]: [location_type],
        feature_columns[1]: [cellphone_access],
        feature_columns[2]: [household_size],
        feature_columns[3]: [age_of_respondent],
        feature_columns[4]: [gender_of_respondent],
        feature_columns[5]: [relationship_with_head],
        feature_columns[6]: [marital_status],
        feature_columns[7]: [education_level],
        feature_columns[8]: [job_type]
    })

    # Prediction using the trained model
    prediction = clf.predict(input_data)[0]

    # Define a constant for clarity
    HAS_BANK_ACCOUNT = 1

    # Check the prediction and provide feedback
    if prediction == HAS_BANK_ACCOUNT:
        st.success("The individual is predicted to have a bank account.")
    else:
        st.error("The individual is predicted to not have a bank account.")

print("Code corrected and ready for execution.")