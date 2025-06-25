import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('output/eye_diseases_model.h5')

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Preprocess image for the model
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    image = image.resize((512, 512))  # Assuming model input size is 224x224
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI setup
def main():
    st.title('Eye Disease Detection using Deep Learning')
    st.write('Upload a retinal image to predict the presence of eye diseases')

    uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write('Classifying...')

        # Preprocess and predict
        image_data = preprocess_image(image)
        prediction = model.predict(image_data)
        predicted_class = np.argmax(prediction)

        labels = ['Age related Macular Degeneration', 'Cataract', 'Diabetes', 'Glaucoma', 
 'Hypertension', 'Pathological Myopia','Normal', 'Other diseases/abnormalities' ]
        
        result = labels[predicted_class]
        st.success(f'Prediction: {result}')

if __name__ == '__main__':
    main()