import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

# Extract features for prediction
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute

features = ['year', 'month', 'day', 'hour', 'minute']
X = df[features]
y_lat = df['latitude']
y_long = df['longitude']

# Preprocessing pipeline for latitude and longitude prediction
numeric_features = ['year', 'month', 'day', 'hour', 'minute']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

lat_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=9, random_state=42))
])

long_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=9, random_state=42))
])

# Train latitude and longitude models
lat_pipeline.fit(X, y_lat)
long_pipeline.fit(X, y_long)


# Preprocessing for KMeans clustering
kmeans_features = ['latitude', 'longitude']

preprocessor_for_kmeans = Pipeline(steps=[
    ('scaler', StandardScaler())
])

kmeans_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_for_kmeans),
    ('kmeans', KMeans(n_clusters=10, random_state=0))
])

# Prepare data for KMeans
X_cluster = df[kmeans_features]

# Fit KMeans pipeline
kmeans_pipeline.fit(X_cluster)
df['cluster'] = kmeans_pipeline.named_steps['kmeans'].labels_

# Extract cluster information
cluster_info = {}
nearest_sightings = {}

for cluster_label in range(kmeans_pipeline.named_steps['kmeans'].n_clusters):
    cluster_data = df[df['cluster'] == cluster_label]
    shape_counter = Counter(cluster_data['shape'])
    most_common_shape = shape_counter.most_common(1)[0][0]
    average_duration = cluster_data['duration (seconds)'].mean()

    cluster_info[cluster_label] = {
        'most_common_shape': most_common_shape,
        'average_duration': average_duration
    }

    nearest_sightings[cluster_label] = cluster_data[['shape', 'duration (seconds)']].head(5)


# Save all models and data into a single pkl file
all_models = {
    'lat_pipeline': lat_pipeline,
    'long_pipeline': long_pipeline,
    'kmeans_pipeline': kmeans_pipeline,
    'cluster_info': cluster_info,
    'nearest_sightings': nearest_sightings
}

joblib.dump(all_models, 'ufo_model.pkl')
