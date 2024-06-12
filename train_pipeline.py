import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
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

# Function to ensure all classes are present
def ensure_all_classes(X, y, all_classes):
    current_classes = set(y.unique())
    missing_classes = set(all_classes) - current_classes

    if missing_classes:
        for missing_class in missing_classes:
            # Create a synthetic sample with median features and the missing class
            synthetic_sample = X.median().to_dict()
            synthetic_sample['shape_encoded'] = missing_class
            X = pd.concat([X, pd.DataFrame([synthetic_sample], columns=X.columns)], ignore_index=True)
            y = pd.concat([y, pd.Series([missing_class])], ignore_index=True)
    return X, y

# Load and preprocess the data
file_path = 'raw_data/scrubbed.csv'
df = pd.read_csv(file_path, low_memory=False)

# Strip whitespace from all column names
df.columns = df.columns.str.strip()

# Clean latitude and longitude data
df['latitude'] = df['latitude'].apply(clean_data)
df['longitude'] = df['longitude'].apply(clean_data)

# Consider only the continental United States
df = df[
    (df['longitude'] >= -124.67) &
    (df['longitude'] <= -66.95) &
    (df['latitude'] >= 25.84) &
    (df['latitude'] <= 49.38)
]

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

# Get unique shape classes dynamically
unique_shapes = df['shape'].unique()
all_classes = list(range(len(unique_shapes)))  # Create a list based on the number of unique shapes

label_encoder = LabelEncoder()
label_encoder.fit(unique_shapes)

# Fitting the label encoder and transforming the shapes
df['shape_encoded'] = label_encoder.transform(df['shape'])

# Ensure all classes are included in the dataset
current_classes = set(df['shape_encoded'].unique())
missing_classes = set(all_classes) - current_classes

if missing_classes:
    for missing_class in missing_classes:
        new_row = df.iloc[0].copy()
        new_row['shape_encoded'] = missing_class
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Verify the shape encoding
# print("Classes after encoding: ", sorted(df['shape_encoded'].unique()))

# Feature extraction for duration
def log_transform(x):
    return np.log1p(x)  # log1p is used to avoid log(0) which is undefined

df['log_duration'] = df['duration (seconds)'].apply(log_transform)

# Extract features for training
features = ['year', 'month', 'day', 'hour', 'minute']
X = df[features]
y = df[['latitude', 'longitude']]

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

location_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(XGBRegressor(n_estimators=50, random_state=42)))
])

# Train the model to predict both latitude and longitude
location_pipeline.fit(X, y)

# Predict latitude and longitude
predictions = location_pipeline.predict(X)
df['predicted_latitude'] = predictions[:, 0]
df['predicted_longitude'] = predictions[:, 1]

# Preprocessing for KMeans clustering using predicted values
kmeans_features = ['predicted_latitude', 'predicted_longitude']

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

# Train models to predict shape and duration for each cluster
shape_duration_models = {}
nearest_sightings = {}

for cluster_label in range(kmeans_pipeline.named_steps['kmeans'].n_clusters):
    cluster_data = df[df['cluster'] == cluster_label]

    if cluster_data.empty:
        continue

    # Predict shape
    X_shape = cluster_data[['predicted_latitude', 'predicted_longitude']]
    y_shape = cluster_data['shape_encoded']

    # Ensure all classes are present in the training data
    X_shape, y_shape = ensure_all_classes(X_shape, y_shape, all_classes)

    if y_shape.nunique() <= 1:
        shape_model = y_shape.mode()[0]
    else:
        X_shape_train, X_shape_test, y_shape_train, y_shape_test = train_test_split(X_shape, y_shape, test_size=0.2, random_state=42)

        # Ensure all classes are present in the training and testing data
        X_shape_train, y_shape_train = ensure_all_classes(X_shape_train, y_shape_train, all_classes)
        X_shape_test, y_shape_test = ensure_all_classes(X_shape_test, y_shape_test, all_classes)

        shape_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=5, random_state=42))
        ])
        shape_pipeline.fit(X_shape_train, y_shape_train)
        shape_model = shape_pipeline

    # Predict duration
    X_duration = cluster_data[['predicted_latitude', 'predicted_longitude']]
    y_duration = cluster_data['log_duration']

    if len(y_duration.unique()) <= 1:
        duration_model = y_duration.mean()
    else:
        X_duration_train, X_duration_test, y_duration_train, y_duration_test = train_test_split(X_duration, y_duration, test_size=0.2, random_state=42)
        duration_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('regressor', XGBRegressor(n_estimators=50, random_state=42))
        ])
        duration_pipeline.fit(X_duration_train, y_duration_train)
        duration_model = duration_pipeline

    # Get the nearest sightings in the cluster
    nearest_sightings[cluster_label] = cluster_data[['latitude', 'longitude', 'shape', 'duration (seconds)']]

    shape_duration_models[cluster_label] = {
        'shape_model': shape_model,
        'duration_model': duration_model
    }

# Save all models and data into a single pkl file
all_models = {
    'location_pipeline': location_pipeline,
    'kmeans_pipeline': kmeans_pipeline,
    'shape_duration_models': shape_duration_models,
    'nearest_sightings': nearest_sightings,
    'label_encoder': label_encoder  # Save the label encoder to decode shapes later
}

joblib.dump(all_models, 'ufo_model.pkl')

print("Model training and saving completed successfully.")
