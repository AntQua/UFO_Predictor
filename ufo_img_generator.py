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

# Define a dictionary with placeholders and corresponding descriptions
prompt_options = {
    "shape": [
        "{predicted_shape} shaped UFO",
        "UFO resembling a {predicted_shape}",
        "UFO in the form of a {predicted_shape}",
        "a {predicted_shape} structure UFO",
        "UFO appearing as a {predicted_shape}"
    ],
    "environment": [
        "hovering over a calm lake",
        "flying above a dense forest",
        "gliding over a bustling city",
        "floating above a serene desert",
        "maneuvering in a stormy sky"
    ],
    "time_of_day": [
        "during a starry night",
        "in the early morning twilight",
        "under the bright midday sun",
        "as the sun sets on the horizon",
        "in the light of dawn"
    ],
    "color": [
        "emitting a bright blue glow",
        "shining with a metallic silver hue",
        "glowing with a vibrant green color",
        "radiating a soft golden light",
        "illuminated with a mysterious red aura"
    ]
}

def generate_dynamic_prompt(predicted_shape):
    shape_description = random.choice(prompt_options["shape"]).format(predicted_shape=predicted_shape)
    environment_description = random.choice(prompt_options["environment"])
    time_of_day_description = random.choice(prompt_options["time_of_day"])
    color_description = random.choice(prompt_options["color"])

    description = f"{shape_description}, {environment_description} {time_of_day_description}, {color_description}."
    return description

def generate_ufo_image(predicted_shape):
    description = generate_dynamic_prompt(predicted_shape)
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

def display_ufo_image(predicted_shape):
    # Apply the custom styles
    add_image_styles()

    description = generate_dynamic_prompt(predicted_shape)
    st.markdown("<h4 style='text-align: center;'>👇 Probably you will see this type of UFO 👇</h4>", unsafe_allow_html=True)
    image_url = generate_ufo_image(description)
    if image_url:
        st.markdown(f"""
            <img src="{image_url}" alt="UFO" class="custom-image"/>
            """, unsafe_allow_html=True)
