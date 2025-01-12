import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the original model and the retrained model
original_model_inference_layer = tf.keras.layers.TFSMLayer('C:/Users/Admin/Downloads/content/agengender',
                                                           call_endpoint='serving_default')
retrained_model_inference_layer = tf.keras.layers.TFSMLayer('C:/Users/Admin/Downloads/retrained_model',
                                                            call_endpoint='serving_default')

# Define input shape (replace with appropriate value)
input_shape = (128, 128)

# Define a new Keras model with the loaded layers as its outputs
input_tensor = tf.keras.Input(shape=input_shape + (1,))

# Define the model structure for both models
original_model_output = original_model_inference_layer(input_tensor)
retrained_model_output = retrained_model_inference_layer(input_tensor)

# Create models for both the original and retrained model
original_model = tf.keras.Model(inputs=input_tensor, outputs=original_model_output)
retrained_model = tf.keras.Model(inputs=input_tensor, outputs=retrained_model_output)


# Function to preprocess the image
def preprocess_image(image, input_shape):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize(input_shape)  # Resize the image to match the input shape
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image


# Add custom CSS for background, file uploader, and prediction boxes
st.markdown(
    """
    <style>
    /* Apply background to the main content */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
        color: white;
    }
    /* Customize the title styling */
    .title {
        font-size: 28px; 
        font-weight: 800; 
        color: #FFD700; 
        text-align: center; 
        margin-bottom: 20px;
    }
    /* Customize file uploader box */
    div[data-testid="stFileUploader"] {
        background-color: #1e3c50; 
        border: 2px dashed #FFD700; 
        border-radius: 10px; 
        padding: 20px;
    }
    /* Customize file uploader label text */
    div[data-testid="stFileUploader"] p {
        color: white;
        font-weight: bold;
        font-size: 16px; 
    }
    /* Style for the prediction box (Age) */
    .prediction-box1 {
        background-color: #020a16; 
        color: #4f0c4c; 
        border-radius: 15px;
        padding: 20px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); 
        text-align: left;
        padding-left: 30px;
    }
    /* Style for the prediction box (Gender) */
    .prediction-box2 {
        background-color: #1f0620; 
        color: #096f25; 
        border-radius: 15px;
        padding: 20px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); 
        margin-top: 10px;
        margin-bottom: 20px;
        text-align: left;
        padding-left: 30px;
    }
    /* Style for the image caption text (Uploaded Image) */
    div[data-testid="stImageCaption"] {
        color: white;
        font-weight: bold;
    }
    button[data-testid="baseButton-minimal"]{
        color: white;
        font-weight: bold;
    }
    div[data-testid="stPagination"] small{
        color: white;
        font-weight: bold;
    }
    div[data-testid="stImage"] img{
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Main function for Streamlit web app
def main():
    st.markdown('<div class="title">✨ Age and Gender Prediction ✨</div>', unsafe_allow_html=True)

    # File uploader with styling, allowing multiple files
    uploaded_files = st.file_uploader("Upload images, each featuring a single person, for predictions:", type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True)

    if uploaded_files:
        # Reverse the order of files so the last uploaded image appears first
        uploaded_files.reverse()

        # Loop through each uploaded file and process it
        for uploaded_file in uploaded_files:
            # Display the selected image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

            # Preprocess the input image
            input_data = preprocess_image(image, input_shape)

            # Add batch dimension to the input data
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.expand_dims(input_data, axis=-1)  # Add channel dimension

            # Make predictions with both models
            original_predictions = original_model.predict(input_data)
            retrained_predictions = retrained_model.predict(input_data)

            # Extract predictions from both models
            # Assuming the model outputs have 'age_out' and 'gender_out'
            original_age_prediction = original_predictions['age_out'][0][0]
            original_gender_prediction = "Female" if original_predictions['gender_out'][0][0] >= 0.5 else "Male"

            retrained_age_prediction = retrained_predictions['age_out'][0][0]
            retrained_gender_prediction = "Female" if retrained_predictions['gender_out'][0][0] >= 0.5 else "Male"

            # Display predictions with custom styling
            st.markdown(
                f'<div class="prediction-box1">Predicted = ( Age (Original Model):  {original_age_prediction:.1f} years | Age (Retrained Model):  {retrained_age_prediction:.1f} years )</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="prediction-box2">Predicted = ( Gender (Original Model):  {original_gender_prediction} | Gender (Retrained Model):  {retrained_gender_prediction} )</div>',
                unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()
