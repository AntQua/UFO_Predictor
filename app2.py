import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import pydeck as pdk
import numpy as np

# Load the models
all_models = joblib.load('ufo_model2.pkl')
lat_pipeline = all_models['lat_pipeline']
long_pipeline = all_models['long_pipeline']
kmeans_pipeline = all_models['kmeans_pipeline']
cluster_info = all_models['cluster_info']
nearest_sightings = all_models['nearest_sightings']

# Function to run the app
def run():
    st.title("UFO Sighting Predictor")

    # Input the date for prediction
    date_input = st.date_input("Choose a date for prediction:", value=datetime.now())
    time_input = st.time_input("Choose a time for prediction:", value=datetime.now().time())

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

        # Ensure the features are of type float32
        future_features = future_features.astype(np.float32)

        predicted_lat = lat_pipeline.predict(future_features)[0]
        predicted_long = long_pipeline.predict(future_features)[0]

        # Convert predictions to float32 for visualization
        predicted_lat = np.float32(predicted_lat)
        predicted_long = np.float32(predicted_long)

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
            get_color='[200, 30, 0, 160]',
            get_radius=50000
        )

        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Predicted Location: {lat}, {lon}"})
        st.pydeck_chart(r)

        # Prepare the input for the KMeans model
        kmeans_input = pd.DataFrame({
            'latitude': [predicted_lat],
            'longitude': [predicted_long]
        })

        # Ensure the kmeans input is of type float32
        kmeans_input = kmeans_input.astype('float')
        #st.write(type(kmeans_input))

        # Transform the input using the KMeans preprocessing pipeline
        preprocessed_input = kmeans_pipeline.named_steps['preprocessor'].transform(kmeans_input)

        # Ensure the preprocessed input is of type float32
        preprocessed_input = np.asarray(preprocessed_input) #, dtype=np.float32
        #st.write(type(preprocessed_input[0][0]))

        predicted_cluster = kmeans_pipeline.named_steps['kmeans'].predict(preprocessed_input)[0]
        #st.write(predicted_cluster)

        most_common_shape = cluster_info[predicted_cluster]['most_common_shape']
        average_duration = cluster_info[predicted_cluster]['average_duration']

        st.write(f"Predicted UFO Shape: {most_common_shape}")
        st.write(f"Average Duration of UFO Sighting: {average_duration:.2f} seconds")

        # Display the five sightings nearest to the predicted cluster
        st.write("Five Sightings Nearest to the Predicted Cluster:")
        nearest_sightings_data = nearest_sightings[predicted_cluster]
        st.table(nearest_sightings_data)

# Run the Streamlit app
if __name__ == '__main__':
    run()
