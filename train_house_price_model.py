from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from house_price_utils import clean_data


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Bengaluru_House_Data.csv"
MODEL_PATH = BASE_DIR / "house_price_model.joblib"

FEATURES = [
    "location",
    "area_type",
    "total_sqft",
    "bath",
    "bhk",
]

TARGET = "price"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place Bengaluru_House_Data.csv inside the data folder."
        )

    print("Loading dataset...")
    raw_df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", raw_df.shape)

    df = clean_data(raw_df)

    print("Cleaned dataset shape:", df.shape)

    if len(df) < 20:
        raise ValueError(
            "Too few valid rows remain after cleaning. "
            "Check the dataset and cleaning rules."
        )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                ),
                ["location", "area_type"],
            ),
            (
                "numerical",
                "passthrough",
                ["total_sqft", "bath", "bhk"],
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training the model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel evaluation")
    print("----------------")
    print(f"Mean Absolute Error: {mae:.2f} lakhs")
    print(f"R² score: {r2:.3f}")

    # Save everything needed by the Streamlit app.
    bundle = {
        "model": model,
        "locations": sorted(df["location"].unique().tolist()),
        "area_types": sorted(df["area_type"].unique().tolist()),
        "features": FEATURES,
        "target_unit": "lakhs",
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

    # Demonstrate one prediction.
    example = pd.DataFrame(
        [
            {
                "location": (
                    "Other" if "Other" in bundle["locations"]
                    else bundle["locations"][0]
                ),
                "area_type": bundle["area_types"][0],
                "total_sqft": 1200,
                "bath": 2,
                "bhk": 2,
            }
        ]
    )

    example_price = model.predict(example)[0]
    print(f"Example predicted price: ₹{example_price:.2f} lakhs")


if __name__ == "__main__":
    main()