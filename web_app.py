import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Define the model
model = keras.Sequential([
    keras.layers.Dense(128, input_dim=102, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the dataset
ds = pd.read_csv('/home/user/Desktop/Lending Club/non_null_output.csv')

# Prepare the data
X = ds.drop(columns='loan_status', axis=1)
Y = ds['loan_status']

# Encode the target labels
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
Y_one_hot = tf.keras.utils.to_categorical(Y_encoded)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=0.2, stratify=Y, random_state=1)

# Standardize the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Train the model
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_std, Y_test)

# Predict on the train and test sets
Y_pred_train = model.predict(X_train_std)
Y_pred_test = model.predict(X_test_std)
Y_pred = np.concatenate((Y_pred_test, Y_pred_train), axis=0)

# Function to make a prediction
def make_prediction(input_data):
    # Get user input for the message content
    user_input = input("Enter parameter values: ")

    # Convert the user input into a NumPy array of floats
    float_array = np.array(user_input.split(","), dtype=float)

    # Reshape and standardize the input data
    input_data_reshaped = float_array.reshape(1, -1)
    input_data_std = scaler.transform(input_data_reshaped)

    # Make a prediction
    prediction = model.predict(input_data_std)
    print(prediction)

    # Determine the loan status based on the prediction
    if prediction[0, 1] >= 0.75:
        print("very good loan")
        nn_output = "very good loan"
    elif prediction[0, 1] >= 0.5:
        print("good loan")
        nn_output = "good loan"
    elif prediction[0, 0] >= 0.75:
        print("very bad loan")
        nn_output = "very bad loan"
    elif prediction[0, 0] >= 0.5:
        print("bad loan")
        nn_output = "bad loan"
    else:
        print("good loan")
        nn_output = "good loan"

# Implement the web application using Streamlit
import streamlit as st

st.title("Loan Prediction Web Application")

# Sidebar for user input
st.sidebar.header("User Input Parameters")
input_data = st.sidebar.text_input("Enter parameter values (comma-separated):")

# Extract text from the PDF document with a character limit
def extract_text_from_pdf(file_path, char_limit=2000):
    import fitz  # PyMuPDF
    document_text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            document_text += page.get_text()
            if len(document_text) >= char_limit:
                document_text = document_text[:char_limit]
                break
    return document_text

# Function to send the document content, user input, and nn_output in chunks
def send_in_chunks(document_text, user_input, nn_output, additional_text, chunk_size=1000):
    import os
    from groq import Groq

    # Initialize the client with the API key
    client = Groq(api_key='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')

    start = 0
    responses = []
    while start < len(document_text):
        chunk = document_text[start:start + chunk_size]
        combined_input = (
            f"Document content:\n{chunk}\n\nUser input: {user_input}\n\n"
            f"NN output: {nn_output}\n\nAdditional text: {additional_text}"
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": combined_input,
                }
            ],
            model="llama3-8b-8192",
        )
        responses.append(chat_completion.choices[0].message.content)
        start += chunk_size
    return responses

# Button to make a prediction
if st.sidebar.button("Predict"):
    if input_data:
        # Convert the user input into a NumPy array of floats
        float_array = np.array(input_data.split(","), dtype=float)

        # Reshape and standardize the input data
        input_data_reshaped = float_array.reshape(1, -1)
        input_data_std = scaler.transform(input_data_reshaped)

        # Make a prediction
        prediction = model.predict(input_data_std)
        st.write("Prediction:", prediction)

        # Determine the loan status based on the prediction
        if prediction[0, 1] >= 0.75:
            nn_output = "very good loan"
        elif prediction[0, 1] >= 0.5:
            nn_output = "good loan"
        elif prediction[0, 0] >= 0.75:
            nn_output = "very bad loan"
        elif prediction[0, 0] >= 0.5:
            nn_output = "bad loan"
        else:
            nn_output = "good loan"

        st.write("NN Output:", nn_output)

        # Extract text from the PDF document
        document_path = '/home/user/Desktop/Lending Club/loan-policy20 .pdf'
        document_text = extract_text_from_pdf(document_path)

        # Send the document content, user input, and nn_output in chunks
        additional_text = "Additional information"
        responses = send_in_chunks(document_text, input_data, nn_output, additional_text)
        for response in responses:
            st.write(response)
    else:
        st.write("Please enter the parameter values.")

