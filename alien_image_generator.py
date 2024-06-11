# alien_image_generator.py

import requests
import json
from PIL import Image
from io import BytesIO
import streamlit as st
import random

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment variable
API_KEY = os.getenv('OPENAI_API_KEY')

def generate_alien_description(race, color, size, shape, features):
    if features:
        descriptions = [
            f"A {race} alien with {color} skin, {size} size, {shape} shape. Additional features: {features}.",
            f"A {race} alien that is {color} in color, {size} in size, and has a {shape} shape. Additional features: {features}.",
            f"An alien of the {race} race, {color} in color, {size} size, {shape} shaped. Additional features: {features}.",
        ]
    else:
        descriptions = [
            f"A {race} alien with {color} skin, {size} size, {shape} shape.",
            f"A {race} alien that is {color} in color, {size} in size, and has a {shape} shape.",
            f"An alien of the {race} race, {color} in color, {size} size, {shape} shaped.",
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

def display_alien_image(race, color, size, shape, features):
    description = generate_alien_description(race, color, size, shape, features)
    st.write(f"Image Description: {description}")  # Print the description above the image
    image_url = generate_alien_image(description)
    if image_url:
        img = get_image_from_url(image_url)
        st.image(img, caption=f"Alien Description: {description}", use_column_width=True)
