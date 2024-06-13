import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import pydeck as pdk
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import ufo_img_generator
import alien_image_generator
from utils import add_image_styles
import json

# Load the models
all_models = joblib.load('ufo_model.pkl')
location_pipeline = all_models['location_pipeline']
kmeans_pipeline = all_models['kmeans_pipeline']
shape_duration_models = all_models['shape_duration_models']
nearest_sightings = all_models['nearest_sightings']
label_encoder = all_models['label_encoder']

# Function to calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).km
    except Exception as e:
        st.error(f"Error calculating distance: {e}")
        return None

def get_location_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="ufo_app")
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location:
            return location.address
        else:
            return "Location not found"
    except Exception as e:
        return f"Error: {e}"

# Function to add a background image
def add_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to center content
def center_content():
    st.markdown(
        """
        <style>
        .st-emotion-cache-ocqkz7,
        .st-emotion-cache-keje6w {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .st-emotion-cache-keje6w > div {
            margin: 0 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to handle prediction and display results
def process_prediction(date_input, time_input):
    future_date = datetime.combine(date_input, time_input)
    future_features = pd.DataFrame({
        'year': [future_date.year],
        'month': [future_date.month],
        'day': [future_date.day],
        'hour': [future_date.hour],
        'minute': [future_date.minute]
    })

    predicted_location = location_pipeline.predict(future_features)
    predicted_lat = predicted_location[0][0]
    predicted_long = predicted_location[0][1]

    # Get location name
    location_name = get_location_name(predicted_lat, predicted_long)

    st.markdown("<h3 style='text-align: center;'>🎯 Predicted Location:</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>{location_name} </h4>", unsafe_allow_html=True)
    st.markdown(f"<h6 style='text-align: center;'>Latitude: {predicted_lat:.2f}, Longitude: {predicted_long:.2f} </h6>", unsafe_allow_html=True)

    display_map(predicted_lat, predicted_long)
    predict_cluster_and_display_sightings(predicted_lat, predicted_long)

# Function to display map
def display_map(pred_lat, pred_long):
    try:
        view_state = pdk.ViewState(
            latitude=37.0902,
            longitude=-95.7129,
            zoom=3,
            pitch=0
        )
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=pd.DataFrame({'lat': [pred_lat], 'lon': [pred_long]}),
            get_position='[lon, lat]',
            get_fill_color='[255, 165, 0, 160]',  # Orange color
            get_radius=50000
        )
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Predicted Location: {lat}, {lon}"})
        st.pydeck_chart(r)
    except Exception as e:
        st.error(f"Error displaying map: {e}")

# Function to get the nearest sightings in the cluster
def get_nearest_sightings(pred_lat, pred_lon, sightings_df, top_n=5):
    try:
        # Calculate distance for each sighting
        sightings_df['distance'] = sightings_df.apply(
            lambda row: calculate_distance(pred_lat, pred_lon, row['latitude'], row['longitude']), axis=1
        )

        # Select the top N nearest sightings
        nearest = sightings_df.nsmallest(top_n, 'distance')

        # Rename columns for clarity
        nearest = nearest.rename(columns={'distance': 'distance (km)'})

        # Apply formatting to the duration and distance columns
        nearest['duration (seconds)'] = nearest['duration (seconds)'].apply(lambda x: f'{x:g}')
        nearest['distance (km)'] = nearest['distance (km)'].apply(lambda x: f'{x:.2f}')

        # Select the relevant columns and reset the index
        nearest = nearest[['shape', 'duration (seconds)', 'distance (km)']].reset_index(drop=True)

        # Add a new sequential index column starting from 1
        nearest.index += 1
        nearest.index.name = 'Index'

        return nearest
    except Exception as e:
        st.error(f"Error fetching nearest sightings: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Function to predict cluster and display sightings
def predict_cluster_and_display_sightings(pred_lat, pred_long):
    kmeans_input = pd.DataFrame({
        'predicted_latitude': [pred_lat],
        'predicted_longitude': [pred_long]
    })
    preprocessed_input = kmeans_pipeline.named_steps['preprocessor'].transform(kmeans_input)
    predicted_cluster = kmeans_pipeline.named_steps['kmeans'].predict(preprocessed_input)[0]

    shape_model = shape_duration_models[predicted_cluster]['shape_model']
    duration_model = shape_duration_models[predicted_cluster]['duration_model']

    if isinstance(shape_model, str):
        predicted_shape_encoded = shape_model
        predicted_shape = label_encoder.inverse_transform([predicted_shape_encoded])[0]
    else:
        predicted_shape_encoded = shape_model.predict(kmeans_input)[0]
        predicted_shape = label_encoder.inverse_transform([predicted_shape_encoded])[0]

    if isinstance(duration_model, (int, float)):
        predicted_duration = duration_model
    else:
        predicted_duration = duration_model.predict(kmeans_input)[0]

    st.markdown(
        f"""
        <div style='text-align: center; font-size: 20px;'>
             🕒 Predicted Duration of <span style='color: orange; font-weight: bold;'>UFO</span> Sighting:
             <span style='color: orange; font-weight: bold;'>{predicted_duration:.2f}</span>
             seconds
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        table {
            width: 100%;
        }
        table th {
            font-size: 20px;
            font-weight: bold;
            text-align: center !important;
            color: white !important;
        }
        table td {
            text-align: center !important;
            vertical-align: middle;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    cluster_sightings = nearest_sightings[predicted_cluster]
    nearest_sightings_df = get_nearest_sightings(pred_lat, pred_long, cluster_sightings)

    st.markdown("<h4 style='text-align: center;'>🛸 Previous sightings nearest to predicted location 🛸</h4>", unsafe_allow_html=True)
    st.table(nearest_sightings_df)

    ufo_img_generator.display_ufo_image(predicted_shape)

# Alien sighting section
def alien_sighting_section():
    if not st.session_state.get('show_alien_section', False):
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Yes"):
                st.session_state['show_alien_section'] = True
                st.rerun()

        with col2:
            if st.button("No"):
                st.session_state.clear()
                st.rerun()
    else:
        st.markdown(
            "<h2 style='text-align: center;'>👽 Describe the <span style='color: #A1DD70; font-weight: bold;'>ALIEN</span> you saw 👽</h2>",
            unsafe_allow_html=True
        )

        with st.form("alien_description_form"):
            col1, col2 = st.columns([1, 1])

            with col1:
                alien_race = st.selectbox("Select the alien race:", ["Grays", "Reptilians", "Nordics", "Anunnaki", "Other"])
                alien_color = st.selectbox("Select the alien color:", ["Green", "Gray", "Blue", "Red", "Other"])
                alien_size = st.slider("Select the alien height (in feet):", min_value=2.0, max_value=15.0, step=0.1)
                alien_shape = st.selectbox("Select the alien body shape:", ["Humanoid", "Insectoid", "Reptilian", "Other"])

            with col2:
                number_of_eyes = st.selectbox("Number of eyes:", ["1", "2", "3", "4", "More"])
                number_of_limbs = st.selectbox("Number of limbs:", ["2", "4", "6", "More"])
                additional_features = st.selectbox(
                    "Select additional features:",
                    ["None", "Wings", "Antennae", "Tentacles", "Fins", "Horns", "Glowing Eyes", "Multiple Mouths", "Scales", "Feathers"]
                )

            submit_button = st.form_submit_button("Generate Alien Image")

        if submit_button:
            with st.spinner("Generating image... Please wait."):
                alien_image_generator.display_alien_image(
                    alien_race, alien_color, alien_size, alien_shape,
                    number_of_eyes, number_of_limbs, additional_features
                )
                st.session_state['show_alien_section'] = False
                st.session_state['alien_image_generated'] = True

    if st.session_state.get('alien_image_generated'):
        st.markdown("<h3 style='text-align: center;'>🛸 Do you want to find another <span style='color: orange; font-weight: bold;'>UFO</span>? 🛸</h3>", unsafe_allow_html=True)
        if st.button("Go back to date and time selection"):
            st.session_state.clear()
            st.experimental_rerun()

# Main function to run the app
def run():
    # Call add_image_styles to apply custom styles
    add_image_styles()

    image_url = "https://images.pexels.com/photos/1169754/pexels-photo-1169754.jpeg"
    add_background_image(image_url)

    if 'show_alien_section' in st.session_state and st.session_state['show_alien_section']:
        alien_sighting_section()
    else:
        st.markdown("<h1 style='text-align: center;'>🛸 <span style='color: orange; font-weight: bold;'>UFO</span> Sighting Predictor 🛸</h1>", unsafe_allow_html=True)
        center_content()

        with st.form(key='date_time_form'):
            cols = st.columns([2, 2])
            with cols[0]:
                date_input = st.date_input("Choose a date for prediction:", value=datetime.now())
            with cols[1]:
                if 'time_input' not in st.session_state:
                    st.session_state['time_input'] = datetime.now().time()
                time_input = st.time_input("Choose a time for prediction:", value=st.session_state['time_input'])
                if st.session_state['time_input'] != time_input:
                    st.session_state['time_input'] = time_input

            # Centered and styled button for form submit
            st.markdown("<div class='button-container'>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("Predict Location")
            st.markdown("</div>", unsafe_allow_html=True)

        if submit_button:
            with st.spinner("Predicting... Please wait."):
                process_prediction(date_input, time_input)
                st.session_state['prediction_done'] = True

            # Format date and time for the output text
            formatted_date = date_input.strftime("%d/%m/%Y")
            formatted_time = time_input.strftime("%H:%M")

            st.markdown(
                f"<h4 style='text-align: center;'>👽 Have you also seen an <span style='color: #A1DD70; font-weight: bold;'>ALIEN</span> in {formatted_date} at {formatted_time}h? 👽</h4>",
                unsafe_allow_html=True
            )

        if 'prediction_done' in st.session_state and st.session_state['prediction_done']:
            alien_sighting_section()

# CSS for custom button styling
st.markdown(
    """
    <style>
    /* Centering the button container */
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin: 20px 0;  /* Adds some margin for spacing */
    }

    /* Centering and styling buttons */
    .row-widget.stButton {
        display: flex;
        justify-content: center;
    }

    .row-widget.stButton button {
        background-color: black !important;
        color: white !important;
        border: 2px solid black !important;
        padding: 10px 24px !important;
        text-align: center !important;
        text-decoration: none !important;
        font-size: 16px !important;
        cursor: pointer !important;
        transition: background-color 0.4s, color 0.4s, border 0.4s !important;
    }

    /* Hover effect for buttons */
    .row-widget.stButton button:hover {
        background-color: black !important;
        color: green !important;
        border: 2px solid green !important;
    }

    /* Focus and active states */
    .row-widget.stButton button:active,
    .row-widget.stButton button:focus {
        outline: none !important;
        box-shadow: none !important;
        background-color: black !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == '__main__':
    run()
