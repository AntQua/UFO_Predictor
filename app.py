import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import pydeck as pdk
from geopy.distance import geodesic
import ufo_img_generator
import alien_image_generator  # Import the new module

# Load the models
all_models = joblib.load('ufo_model.pkl')
location_pipeline = all_models['location_pipeline']
kmeans_pipeline = all_models['kmeans_pipeline']
shape_duration_models = all_models['shape_duration_models']
nearest_sightings = all_models['nearest_sightings']
label_encoder = all_models['label_encoder']

# Function to calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Function to get the nearest sightings in the cluster
def get_nearest_sightings(pred_lat, pred_lon, sightings_df, top_n=5):
    sightings_df['distance'] = sightings_df.apply(
        lambda row: calculate_distance(pred_lat, pred_lon, row['latitude'], row['longitude']), axis=1
    )
    nearest = sightings_df.nsmallest(top_n, 'distance')
    nearest['duration (seconds)'] = nearest['duration (seconds)'].apply(lambda x: f'{x:g}')
    nearest['distance'] = nearest['distance'].apply(lambda x: f'{x:.2f}')
    nearest = nearest.rename(columns={'distance': 'distance (km)'})
    return nearest[['shape', 'duration (seconds)', 'distance (km)']]

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

def center_content():
    st.markdown(
        """
        <style>
        /* Center the button container */
        .st-emotion-cache-ocqkz7 {
            display: flex;
            justify-content: center;
        }

        /* Center the buttons and make them inline */
        .st-emotion-cache-keje6w {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        /* Adjust margin and spacing between buttons */
        .st-emotion-cache-keje6w > div {
            margin: 0 10px; /* Adjust as needed for spacing between buttons */
        }
        </style>
        """,
        unsafe_allow_html=True
    )



# Function to run the app
def run():
    # Set the background image
    image_url = "https://images.pexels.com/photos/1169754/pexels-photo-1169754.jpeg"
    add_background_image(image_url)

    st.markdown("<h1 style='text-align: center;'>🛸 UFO Sighting Predictor 🛸</h1>", unsafe_allow_html=True)

    # Center the content
    center_content()

    # Date and time inputs
    with st.container():
        cols = st.columns([2, 2])
        with cols[0]:
            date_input = st.date_input("Choose a date for prediction:", value=datetime.now())
        with cols[1]:
            if 'time_input' not in st.session_state:
                st.session_state['time_input'] = datetime.now().time()
            time_input = st.time_input("Choose a time for prediction:", value=st.session_state['time_input'])
            if st.session_state['time_input'] != time_input:
                st.session_state['time_input'] = time_input

    # Predict location button
    with st.container():
        cols = st.columns([1, 1, 1])
        with cols[1]:
            button_clicked = st.button("Predict Location", key="predict_location_button")

    # Predict the location logic
    if button_clicked:
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
        st.write(f"Predicted Location: Latitude {predicted_lat}, Longitude {predicted_long}")

        # Display the predicted location on a map
        view_state = pdk.ViewState(
            latitude=37.0902,
            longitude=-95.7129,
            zoom=3,
            pitch=0
        )
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=pd.DataFrame({'lat': [predicted_lat], 'lon': [predicted_long]}),
            get_position='[lon, lat]',
            get_fill_color='[200, 30, 0, 160]',
            get_radius=50000
        )
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Predicted Location: {lat}, {lon}"})
        st.pydeck_chart(r)

        kmeans_input = pd.DataFrame({
            'predicted_latitude': [predicted_lat],
            'predicted_longitude': [predicted_long]
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
            """
            <style>
            table {
                width: 100%;
            }
            table th {
                font-size: 20px;
                font-weight: bold;
                text-align: center !important;
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
        nearest_sightings_df = get_nearest_sightings(predicted_lat, predicted_long, cluster_sightings)
        nearest_sightings_df = nearest_sightings_df.reset_index(drop=True)
        st.markdown("<h4 style='text-align: center;'>🛸 Previous sightings nearest to predicted location 🛸</h4>", unsafe_allow_html=True)
        st.table(nearest_sightings_df)

        st.markdown(
            f"""
            <div style='text-align: center; font-size: 20px; font-weight: bold;'>
                Predicted Duration of UFO Sighting: {predicted_duration:.2f} seconds
            </div>
            """,
            unsafe_allow_html=True
        )

        # Use the image_generator module to display the image
        #ufo_img_generator.display_ufo_image(predicted_shape)

        # Mark that prediction is done
        st.session_state['prediction_done'] = True

    # Alien sighting section - Show after prediction
    if 'prediction_done' in st.session_state and st.session_state['prediction_done']:
        st.markdown("<h2 style='text-align: center;'>👽 Have you seen an alien? 👽</h2>", unsafe_allow_html=True)

        st.markdown("<div class='st-emotion-cache-ocqkz7'>", unsafe_allow_html=True)  # Open the button container
        col1, col2 = st.columns([1, 1])

        with col1:
            yes_button = st.button("Yes", key="yes_button")
        with col2:
            no_button = st.button("No", key="no_button")

        st.markdown("</div>", unsafe_allow_html=True)  # Close the button container


        if yes_button:
            st.session_state['show_alien_form'] = True

        if 'show_alien_form' in st.session_state and st.session_state['show_alien_form']:
            with st.form("alien_description_form"):
                st.markdown("### Describe the alien you saw:")
                alien_race = st.text_input("Enter the alien race:", "")
                alien_color = st.text_input("Enter the alien color:", "")
                alien_size = st.text_input("Enter the alien size:", "")
                alien_shape = st.text_input("Enter the alien shape:", "")
                additional_features = st.text_area("Additional Features", "")
                submit_button = st.form_submit_button("Generate Alien Image")

            if submit_button:
                alien_image_generator.display_alien_image(
                    alien_race, alien_color, alien_size, alien_shape, additional_features
                )
                st.session_state['show_alien_form'] = False

        if no_button:
            st.session_state.clear()
            st.rerun()

# Run the Streamlit app
if __name__ == '__main__':
    run()
