import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import pydeck as pdk
from geopy.distance import geodesic
import ufo_img_generator as image_generator  # Import the image generator module
# Load the models
all_models = joblib.load('ufo_model.pkl')
lat_pipeline = all_models['lat_pipeline']
long_pipeline = all_models['long_pipeline']
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
    # Format duration and distance
    nearest['duration (seconds)'] = nearest['duration (seconds)'].apply(lambda x: f'{x:g}')
    nearest['distance'] = nearest['distance'].apply(lambda x: f'{x:.2f}')
    # Rename columns
    nearest = nearest.rename(columns={'distance': 'distance (km)'})
    return nearest[['shape', 'duration (seconds)', 'distance (km)']]
# Function to run the app
def run():
    st.markdown("<h1 style='text-align: center;'>:flying_saucer: UFO Sighting Predictor :alien:</h1>", unsafe_allow_html=True)
    # Input the date for prediction
    date_input = st.date_input("Choose a date for prediction:", value=datetime.now())
    # Initialize 'time_input' in session state if not already set
    if 'time_input' not in st.session_state:
        st.session_state['time_input'] = datetime.now().time()
    # Use the value from session state for the widget
    time_input = st.time_input("Choose a time for prediction:", value=st.session_state['time_input'])
    # Update session state only if the time input is changed
    if st.session_state['time_input'] != time_input:
        st.session_state['time_input'] = time_input
    # Predict the location
    if st.button("Predict Location"):
        future_date = datetime.combine(date_input, time_input)
        future_features = pd.DataFrame({
            'year': [future_date.year],
            'month': [future_date.month],
            'day': [future_date.day],
            'hour': [future_date.hour],
            'minute': [future_date.minute]
        })
        predicted_lat = lat_pipeline.predict(future_features)[0]
        predicted_long = long_pipeline.predict(future_features)[0]
        st.write(f"Predicted Location: Latitude {predicted_lat}, Longitude {predicted_long}")
        # Display the predicted location on a map centered on the USA
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
            get_fill_color='[200, 30, 0, 160]',  # Use getFillColor
            get_radius=50000
        )
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Predicted Location: {lat}, {lon}"})
        st.pydeck_chart(r)
        # Prepare the input for the KMeans model
        kmeans_input = pd.DataFrame({
            'predicted_latitude': [predicted_lat],  # Ensure consistent feature names
            'predicted_longitude': [predicted_long]  # Ensure consistent feature names
        })
        # Transform the input using the KMeans preprocessing pipeline
        preprocessed_input = kmeans_pipeline.named_steps['preprocessor'].transform(kmeans_input)
        predicted_cluster = kmeans_pipeline.named_steps['kmeans'].predict(preprocessed_input)[0]
        # Retrieve the shape and duration models for the cluster
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
        # Use the new image_generator module to display the image
        image_generator.display_ufo_image(predicted_shape)
        st.write(f"Predicted Duration of UFO Sighting: {predicted_duration:.2f} seconds")
        # Get the nearest sightings in the cluster
        cluster_sightings = nearest_sightings[predicted_cluster]
        nearest_sightings_df = get_nearest_sightings(predicted_lat, predicted_long, cluster_sightings)
        st.write("Nearest Sightings in the Cluster:")
        st.table(nearest_sightings_df)
# Run the Streamlit app
if __name__ == '__main__':
    run()
