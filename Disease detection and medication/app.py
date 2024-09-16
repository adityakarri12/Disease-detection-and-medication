import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sqlite3

# Database connection
@st.cache_resource
def init_connection():
    return sqlite3.connect('healthcare_db.db')

conn = init_connection()

# Function to load data
@st.cache_data 
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Load datasets
disease_data = load_data('Training.csv')
drug_data = load_data('Drug.csv')

# Prepare datasets
# Disease Prediction Dataset
df_x_disease = disease_data.drop(columns=['prognosis'])
df_y_disease = disease_data[['prognosis']]

# Drug Prediction Dataset
# Perform replacements without downcasting warnings
drug_data['Gender'] = drug_data['Gender'].replace({'Female': 0, 'Male': 1})
drug_data['Disease'] = drug_data['Disease'].replace({
    'Acne': 0, 'Allergy': 1, 'Diabetes': 2, 'Fungal infection': 3,
    'Urinary tract infection': 4, 'Malaria': 5, 'Migraine': 6, 'Hepatitis B': 7,
    'AIDS': 8
})

df_x_drug = drug_data[['Disease', 'Gender', 'Age']]
df_y_drug = drug_data[['Drug']]

# Train/Test split for both datasets
x_train_disease, x_test_disease, y_train_disease, y_test_disease = train_test_split(df_x_disease, df_y_disease, test_size=0.2, random_state=0)
x_train_drug, x_test_drug, y_train_drug, y_test_drug = train_test_split(df_x_drug, df_y_drug, test_size=0.2, random_state=0)

# Feature Scaling
scaler_disease = StandardScaler()
scaler_drug = StandardScaler()

# Fit scalers and transform data
x_train_disease_scaled = scaler_disease.fit_transform(x_train_disease)
x_test_disease_scaled = scaler_disease.transform(x_test_disease)

x_train_drug_scaled = scaler_drug.fit_transform(x_train_drug)
x_test_drug_scaled = scaler_drug.transform(x_test_drug)

# Model Training with scaled data
def train_all_models(x_train, y_train, x_test, y_test):
    models = {
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Decision Tree': tree.DecisionTreeClassifier()
    }
    best_model = None
    best_accuracy = 0
    model_accuracies = {}

    for model_name, model in models.items():
        model.fit(x_train, np.ravel(y_train))
        accuracy = model.score(x_test, y_test)
        model_accuracies[model_name] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # Save all models
    for model_name, model in models.items():
        joblib.dump(model, f'model/{model_name.lower().replace(" ", "_")}.pkl')

    return best_model, model_accuracies

# Model Prediction
def predict(model, features, scaler=None):
    features = np.array(features).reshape(1, -1)
    if scaler:
        features = scaler.transform(features)  # Scale the feature vector
    try:
        prediction = model.predict(features)
    except ValueError as e:
        st.error(f"Error in prediction: {str(e)}")
        prediction = None
    return prediction

# Streamlit UI
st.title('Healthcare Prediction System prototype')

# Select prediction task
task_choice = st.selectbox('Choose the prediction task', ['Disease Prediction', 'Drug Prediction'])

if task_choice == 'Disease Prediction':
    # Sidebar for model training for disease prediction
    st.sidebar.subheader('Disease Prediction')

    if st.sidebar.button('Train Disease Models'):
        best_model, model_accuracies = train_all_models(x_train_disease_scaled, y_train_disease, x_test_disease_scaled, y_test_disease)
        st.sidebar.write(f'Best Model: {best_model}')
        st.sidebar.write(f'Accuracies: {model_accuracies}')

    # Dropdown for symptom selection
    available_symptoms = df_x_disease.columns.tolist()  # Use columns of the dataset as available symptoms
    selected_symptoms = st.multiselect('Select your symptoms:', available_symptoms)

    # Model Selection for Prediction
    model_to_load = st.selectbox('Choose a model for prediction', ['Naive Bayes', 'Random Forest', 'Logistic Regression', 'Decision Tree'])

    if st.button('Predict Disease'):
        if selected_symptoms:
            # Convert selected symptoms to feature vector
            feature_vector = [1 if symptom in selected_symptoms else 0 for symptom in df_x_disease.columns]
            
            model = joblib.load(f'model/{model_to_load.lower().replace(" ", "_")}.pkl')
            prediction = predict(model, feature_vector, scaler_disease)
            if prediction is not None:
                st.write(f'The predicted disease is: {prediction[0]}')
        else:
            st.error("Please select symptoms from the dropdown.")

elif task_choice == 'Drug Prediction':
    # Sidebar for model training for drug prediction
    st.sidebar.subheader('Drug Prediction')

    if st.sidebar.button('Train Drug Models'):
        best_model, model_accuracies = train_all_models(x_train_drug_scaled, y_train_drug, x_test_drug_scaled, y_test_drug)
        st.sidebar.write(f'Best Model: {best_model}')
        st.sidebar.write(f'Accuracies: {model_accuracies}')

    # Input for Drug Prediction
    disease = st.selectbox('Choose Disease', ['Acne', 'Allergy', 'Diabetes', 'Fungal infection',
                                              'Urinary tract infection', 'Malaria', 'Migraine', 'Hepatitis B', 'AIDS'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 0, 100, 25)

    # Convert inputs to the correct format
    disease_mapping = {'Acne': 0, 'Allergy': 1, 'Diabetes': 2, 'Fungal infection': 3,
                       'Urinary tract infection': 4, 'Malaria': 5, 'Migraine': 6, 'Hepatitis B': 7,
                       'AIDS': 8}
    gender_mapping = {'Male': 1, 'Female': 0}

    # Ensure features match the expected input shape
    features = [disease_mapping[disease], gender_mapping[gender], age]

    # Model Selection for Prediction
    model_to_load = st.selectbox('Choose a model for drug prediction', ['Naive Bayes', 'Random Forest'])

    if st.button('Predict Drug'):
        model = joblib.load(f'model/{model_to_load.lower().replace(" ", "_")}.pkl')
        try:
            # Ensure features are in the right format
            if len(features) == len(df_x_drug.columns):
                prediction = predict(model, features, scaler_drug)
                if prediction is not None:
                    st.write(f'The predicted drug is: {prediction[0]}')
            else:
                st.error("Feature vector length mismatch. Ensure all input features are provided.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Option to view raw data
if st.checkbox('Show raw data'):
    if task_choice == 'Disease Prediction':
        st.write(disease_data)
    elif task_choice == 'Drug Prediction':
        st.write(drug_data)
