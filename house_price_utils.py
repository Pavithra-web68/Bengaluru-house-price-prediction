import re
import numpy as np
import pandas as pd


def parse_bhk(value):
    """Extract the number of bedrooms from values such as '2 BHK'."""
    if pd.isna(value):
        return np.nan

    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan


def parse_sqft(value):
    """Convert square-foot values and ranges to numeric values."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(",", "")

    # Handle ranges such as '1000 - 1200'
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if not numbers:
        return np.nan

    values = [float(number) for number in numbers]

    # Use the average when the area is given as a range.
    return sum(values) / len(values)


def clean_data(df):
    """Clean the Bengaluru housing dataset for model training."""
    df = df.copy()

    required = {"location", "size", "total_sqft", "bath", "price"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    # Remove duplicate rows.
    df = df.drop_duplicates()

    # Convert the target and numerical fields.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["bath"] = pd.to_numeric(df["bath"], errors="coerce")
    df["total_sqft"] = df["total_sqft"].apply(parse_sqft)
    df["bhk"] = df["size"].apply(parse_bhk)

    # Clean location names.
    df["location"] = df["location"].astype("string").str.strip()
    df["location"] = df["location"].replace("", pd.NA)

    # Keep rows with valid model inputs and target.
    df = df.dropna(
        subset=["location", "total_sqft", "bath", "bhk", "price"]
    )

    # Remove impossible or clearly invalid values.
    df = df[
        (df["total_sqft"] > 0)
        & (df["bath"] > 0)
        & (df["bhk"] > 0)
        & (df["price"] > 0)
    ]

    # Avoid using implausible area-per-bedroom records.
    df = df[df["total_sqft"] / df["bhk"] >= 200]

    # Group rare locations to reduce the number of categories.
    location_counts = df["location"].value_counts()
    common_locations = set(location_counts[location_counts >= 10].index)

    df["location"] = df["location"].apply(
        lambda location: (
            location if location in common_locations else "Other"
        )
    )

    # Use area type when available.
    if "area_type" not in df.columns:
        df["area_type"] = "Unknown"

    df["area_type"] = df["area_type"].fillna("Unknown").astype(str)

    return df