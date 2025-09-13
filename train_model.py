import os
import time
import argparse

import googlemaps
import pandas as pd
import numpy as np
import joblib

from geopy.distance import geodesic
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load environment variables from the specified .env file
load_dotenv(dotenv_path="D:/cc/eta_engine_project/.env")

# Retrieve the API key from the environment
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
print("Loaded API KEY:", API_KEY)

# Validate and initialize the Google Maps client
if not API_KEY:
    raise ValueError("Please set GOOGLE_MAPS_API_KEY in your .env file or environment.")

gmaps = googlemaps.Client(key=API_KEY)



def get_road_distance(start_lat, start_lon, dest_lat, dest_lon):
    """Get road distance using Google Maps Directions API."""
    try:
        directions = gmaps.directions(
            origin=(start_lat, start_lon),
            destination=(dest_lat, dest_lon),
            mode="driving",
            alternatives=False
        )
        if directions:
            # distance.value is in meters
            return directions[0]["legs"][0]["distance"]["value"] / 1000.0
        return None
    except Exception as e:
        print(f"Error fetching road distance: {e}")
        return None


def calculate_geodesic_distance(start_lat, start_lon, dest_lat, dest_lon):
    """Calculate straight‐line (geodesic) distance in kilometers."""
    return geodesic((start_lat, start_lon), (dest_lat, dest_lon)).kilometers


def create_mock_dataset_with_limited_api_calls(n_samples=100):
    """Build a small dataset, fetch some real road-distances, fallback to estimates."""
    np.random.seed(42)
    lats_a = np.random.uniform(34.0, 34.3, size=n_samples)
    lons_a = np.random.uniform(-118.5, -118.2, size=n_samples)
    lats_b = np.random.uniform(34.0, 34.3, size=n_samples)
    lons_b = np.random.uniform(-118.5, -118.2, size=n_samples)

    distances = []
    geodesics = []
    api_success = 0

    print("Fetching distances (this may take a bit)…")
    for i in range(n_samples):
        geo = calculate_geodesic_distance(lats_a[i], lons_a[i], lats_b[i], lons_b[i])
        geodesics.append(geo)

        road = get_road_distance(lats_a[i], lons_a[i], lats_b[i], lons_b[i])
        if road is not None:
            distances.append(road)
            api_success += 1
        else:
            # Heuristic multiplier for no-data cases
            distances.append(geo * 1.3)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_samples}")
            time.sleep(1)

    print(f"Successful API calls: {api_success}/{n_samples}")

    traffic     = np.random.randint(0, 3, size=n_samples)
    weather     = np.random.randint(0, 3, size=n_samples)
    time_of_day = np.random.randint(0, 24, size=n_samples)

    eta = (
        np.array(distances) * 2.5 +
        traffic * 10 +
        weather * 7 +
        ((time_of_day > 16) & (time_of_day < 19)) * 15 +
        np.random.normal(0, 5, size=n_samples)
    )
    eta = np.maximum(eta, 1)

    return pd.DataFrame({
        "start_lat":            lats_a,
        "start_lon":            lons_a,
        "dest_lat":             lats_b,
        "dest_lon":             lons_b,
        "distance_km":          distances,
        "geodesic_distance_km": geodesics,
        "traffic_level":        traffic,
        "weather_condition":    weather,
        "time_of_day":          time_of_day,
        "actual_eta":           eta
    })


def create_synthetic_dataset(n_samples=1000):
    """Generate a larger dataset purely synthetically."""
    np.random.seed(42)
    lats_a = np.random.uniform(34.0, 34.3, size=n_samples)
    lons_a = np.random.uniform(-118.5, -118.2, size=n_samples)
    lats_b = np.random.uniform(34.0, 34.3, size=n_samples)
    lons_b = np.random.uniform(-118.5, -118.2, size=n_samples)

    geodesics = [
        calculate_geodesic_distance(lats_a[i], lons_a[i], lats_b[i], lons_b[i])
        for i in range(n_samples)
    ]
    multiplier = np.random.uniform(1.2, 1.5, size=n_samples)
    distances = np.array(geodesics) * multiplier

    traffic     = np.random.randint(0, 3, size=n_samples)
    weather     = np.random.randint(0, 3, size=n_samples)
    time_of_day = np.random.randint(0, 24, size=n_samples)

    eta = (
        distances * 2.5 +
        traffic * 10 +
        weather * 7 +
        ((time_of_day > 16) & (time_of_day < 19)) * 15 +
        np.random.normal(0, 5, size=n_samples)
    )
    eta = np.maximum(eta, 1)

    return pd.DataFrame({
        "start_lat":            lats_a,
        "start_lon":            lons_a,
        "dest_lat":             lats_b,
        "dest_lon":             lons_b,
        "distance_km":          distances,
        "geodesic_distance_km": geodesics,
        "traffic_level":        traffic,
        "weather_condition":    weather,
        "time_of_day":          time_of_day,
        "actual_eta":           eta
    })


def train_model(df: pd.DataFrame) -> RandomForestRegressor:
    """Train a RandomForest to predict ETA (in minutes)."""
    print("Training model…")
    features = ["distance_km", "traffic_level", "weather_condition", "time_of_day"]
    X = df[features]
    y = df["actual_eta"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"Model Training Complete")
    print(f"  MAE: {mae:.2f} minutes")
    print(f"  R² : {r2:.3f}")

    fi = pd.DataFrame({
        "feature":    features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nFeature importance:")
    for _, row in fi.iterrows():
        print(f"  {row.feature}: {row.importance:.3f}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an ETA prediction model with optional Google Maps API calls."
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use limited Google Maps Directions API calls when building the dataset."
    )
    return parser.parse_args()


def main(use_api: bool):
    os.makedirs("model", exist_ok=True)

    if use_api:
        df = create_mock_dataset_with_limited_api_calls()
    else:
        df = create_synthetic_dataset()

    model = train_model(df)

    joblib.dump(model, "model/eta_model.pkl")
    df.to_csv("model/eta_training_data.csv", index=False)

    print(f"\nModel and data saved successfully! Dataset size: {len(df)} samples")


if __name__ == "__main__":
    args = parse_args()
    main(use_api=args.use_api)