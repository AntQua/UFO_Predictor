import requests
import json
from PIL import Image
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
# Load environment variables from the .env file
load_dotenv()
# Retrieve the API key from the environment variable
API_KEY = st.secrets['OPENAI_API_KEY']

def generate_ufo_image(description):
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": description,
        "n": 1,
        "size": "1024x1024"
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        image_url = response_data['data'][0]['url']
        return image_url
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def get_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def display_ufo_image(predicted_shape):
    description = f"A {predicted_shape} shaped UFO over the mountains"
    image_url = generate_ufo_image(description)
    if image_url:
        img = get_image_from_url(image_url)
        st.image(img, caption=f"Predicted UFO Shape: {predicted_shape}", use_column_width=True)
