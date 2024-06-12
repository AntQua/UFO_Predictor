import requests
import json
from PIL import Image
from io import BytesIO
import streamlit as st
import random
from utils import add_image_styles

# comment this if in production
# from dotenv import load_dotenv
# import os

# comment this if in production
# load_dotenv()

# comment this if in production
# API_KEY = os.getenv('OPENAI_API_KEY')

# uncomment this if in production
API_KEY = st.secrets['OPENAI_API_KEY']


def generate_alien_description(texture, color, eyes, limbs, features):
    if features:
        descriptions = [
            f"An alien with {color} {texture} skin, {eyes} eyes, and {limbs}. Additional features: {features}.",
            f"An alien that is {color} in color, with {texture} skin, {eyes} eyes, and {limbs}. Additional features: {features}.",
            f"An alien with {texture} skin, {color} in color, {eyes} eyes, and {limbs}. Additional features: {features}.",
        ]
    else:
        descriptions = [
            f"An alien with {color} {texture} skin, {eyes} eyes, and {limbs}.",
            f"An alien that is {color} in color, with {texture} skin, {eyes} eyes, and {limbs}.",
            f"An alien with {texture} skin, {color} in color, {eyes} eyes, and {limbs}.",
        ]

    return random.choice(descriptions)


def generate_alien_image(description):
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

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raises an HTTPError for bad responses
        response_data = response.json()
        image_url = response_data['data'][0]['url']
        return image_url
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to generate image: {e}")
        return None

def get_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def display_alien_image(race, color, size, shape, features):
    # Apply the custom styles
    add_image_styles()

    description = generate_alien_description(race, color, size, shape, features)
    st.write(f"Image Description: {description}")  # Print the description above the image
    image_url = generate_alien_image(description)
    if image_url:
        st.markdown(f"""
            <img src="{image_url}" alt="Alien" class="custom-image"/>
            """, unsafe_allow_html=True)
