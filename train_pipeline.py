import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from collections import Counter
import joblib

# Data cleaning function
def clean_data(value):
    try:
        return float(value)
    except ValueError:
        cleaned_value = ''.join(char for char in value if char.isdigit() or char == '.')
        try:
            return float(cleaned_value)
        except ValueError:
            return float('nan')

# Load and preprocess the data
file_path = 'raw_data/scrubbed.csv'
df = pd.read_csv(file_path)
df['latitude'] = df['latitude'].apply(clean_data)
df['longitude '] = df['longitude '].apply(clean_data)
df.columns = df.columns.str.strip()
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['duration (seconds)'] = pd.to_numeric(df['duration (seconds)'], errors='coerce')
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna()

# Extract features and target variables
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute

features = ['year', 'month', 'day', 'hour', 'minute']
X = df[features]
y_lat = df['latitude']
y_long = df['longitude']

# Split the data into training and test sets
X_train, X_test, y_lat_train, y_lat_test, y_long_train, y_long_test = train_test_split(X, y_lat, y_long, test_size=0.2, random_state=42)

# Define and train the pipelines for latitude and longitude prediction
lat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

long_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

lat_pipeline.fit(X_train, y_lat_train)
long_pipeline.fit(X_train, y_long_train)

# Train the KMeans model for shape and duration prediction
location_data = df[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(location_data)
df['cluster'] = kmeans.labels_

# Extract the cluster information
cluster_info = {}
nearest_sightings = {}

for cluster_label in range(kmeans.n_clusters):
    cluster_data = df[df['cluster'] == cluster_label]
    shape_counter = Counter(cluster_data['shape'])
    most_common_shape = shape_counter.most_common(1)[0][0]
    average_duration = cluster_data['duration (seconds)'].mean()

    cluster_info[cluster_label] = {
        'most_common_shape': most_common_shape,
        'average_duration': average_duration
    }

    # Save the nearest sightings for each cluster
    nearest_sightings[cluster_label] = cluster_data[['shape', 'duration (seconds)']].head(5)

# Save all models and data into a single pkl file
all_models = {
    'lat_pipeline': lat_pipeline,
    'long_pipeline': long_pipeline,
    'kmeans': kmeans,
    'cluster_info': cluster_info,
    'nearest_sightings': nearest_sightings
}

joblib.dump(all_models, 'ufo_model.pkl')
