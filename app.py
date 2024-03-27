import streamlit as st
#from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle


pickle_in = open("autoencoder.pkl","rb")
classifier=pickle.load(pickle_in)






# Function to load and preprocess the input image
'''def load_and_prep_image(image, img_shape=420):
    image = Image.open(image).convert('L')  # Convert to grayscale
    image = image.resize((img_shape, img_shape))
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=[0, -1])  # Add batch and channel dimension
    return image
'''
'''
def load_and_prep_image(image, img_shape=(420, 540)):
    image = Image.open(image).convert('L')  # Convert to grayscale
    image = image.resize(img_shape)  # Resize image to match model's expected input shape
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
'''

def load_and_prep_image(image, img_shape=(540, 420)):  # Note the swapped dimensions to match model expectation
    image = Image.open(image).convert('L')  # Convert to grayscale
    image = image.resize(img_shape)  # Resize image (width x height)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image



# Function to display images
def display_images(noisy_img, clean_img):
    st.image(noisy_img, caption='Noisy Image', use_column_width=True)
    st.image(clean_img, caption='Denoised Image', use_column_width=True)

# Load your model
#model = load_model('my_model.h5')

# Streamlit UI
st.title('Image Denoising with Autoencoder')
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

#import streamlit as st
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)


if uploaded_file is not None:
    noisy_image = load_and_prep_image(uploaded_file)
    denoised_image = classifier.predict(noisy_image)[0]  # Predict and remove batch dimension
    denoised_image = np.clip(denoised_image * 255, 0, 255).astype(np.uint8)  # Rescale to [0, 255]
    
    # Display images
    display_images(uploaded_file, denoised_image.squeeze(-1))  # Squeeze to remove channel dim for display
#C:\Users\VIJAY\anaconda3\lib\site-packages\ipykernel_launcher.py 
