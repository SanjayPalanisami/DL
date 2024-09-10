
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import streamlit as st

# Load the dataset
file_path = 'SeoulBikeData.csv'  # Adjust the path accordingly
df = pd.read_csv(file_path)

# One-hot encode the Seasons column
seasons_encoded = pd.get_dummies(df['Seasons'], prefix='Season')

# Binary encode the Holiday column
df['Holiday'] = df['Holiday'].apply(lambda x: 1 if x == 'Holiday' else 0)

# Binary encode the Functioning Day (target variable)
df['Functioning Day'] = df['Functioning Day'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop the original Seasons column and concatenate the encoded columns
df = df.drop(columns=['Seasons'])
df = pd.concat([df, seasons_encoded], axis=1)

# Preprocess the data (dropping Date column)
features = df.drop(columns=['Date', 'Rented Bike Count', 'Functioning Day'])
target = df['Functioning Day']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),  # Updated input layer
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile the model with binary crossentropy loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Streamlit Interface for Prediction
st.title("Seoul Bike Rental: Functioning Day Prediction")

# Streamlit input fields
Hour = st.number_input("Hour", min_value=0, max_value=23)
Temperature = st.number_input("Temperature (°C)")
Humidity = st.number_input("Humidity (%)")
Wind_speed = st.number_input("Wind Speed (m/s)")
Visibility = st.number_input("Visibility (10m)")
Dew_point_temperature = st.number_input("Dew Point Temperature (°C)")
Solar_Radiation = st.number_input("Solar Radiation (MJ/m²)")
Rainfall = st.number_input("Rainfall (mm)")
Snowfall = st.number_input("Snowfall (cm)")
Seasons = st.selectbox("Seasons", ["Autumn", "Spring", "Summer", "Winter"])
Holiday = st.selectbox("Holiday", ["Holiday", "No Holiday"])

# Convert Seasons into one-hot encoded format for input
season_encoding = [0, 0, 0, 0]  # [Season_Autumn, Season_Spring, Season_Summer, Season_Winter]
if Seasons == 'Autumn':
    season_encoding[0] = 1
elif Seasons == 'Spring':
    season_encoding[1] = 1
elif Seasons == 'Summer':
    season_encoding[2] = 1
elif Seasons == 'Winter':
    season_encoding[3] = 1

# Convert Holiday to binary format
Holiday = 1 if Holiday == 'Holiday' else 0

# Combine all features into a single input array
input_features = np.array([[Hour, Temperature, Humidity, Wind_speed, Visibility, Dew_point_temperature,
                            Solar_Radiation, Rainfall, Snowfall, Holiday] + season_encoding])

# Scale the input
input_features_scaled = scaler.transform(input_features)

# Prediction button
if st.button("Predict Functioning Day"):
    prediction = model.predict(input_features_scaled)
    rounded_prediction = np.round(prediction).astype(int)
    functioning_day = 'Yes' if rounded_prediction[0][0] == 1 else 'No'
    st.success(f"The predicted functioning day is: {functioning_day}")
