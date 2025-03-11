import os
import pickle

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Load the Spotify tracks dataset from Hugging Face
    """
    print("Loading Spotify tracks dataset from Hugging Face...")
    try:
        # Load the dataset using Hugging Face datasets
        ds = load_dataset("maharshipandya/spotify-tracks-dataset")

        # Convert to pandas DataFrame
        df = pd.DataFrame(ds["train"])
        print(
            f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns"
        )
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def explore_data(df):
    """
    Basic exploration of the dataset
    """
    print("\n=== Dataset Overview ===")
    print(f"Dataset shape: {df.shape}")

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\n=== Genre Distribution ===")
    genre_counts = df["track_genre"].value_counts()
    print(f"Number of unique genres: {len(genre_counts)}")
    print(genre_counts.head(10))

    print("\n=== Popularity Statistics ===")
    print(df["popularity"].describe())

    return {"missing_values": missing_values, "genre_counts": genre_counts}


def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers
    """
    print("\n=== Cleaning Data ===")

    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()

    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().sum() > 0:
            if cleaned_df[column].dtype == "object":
                # For string columns, fill with 'Unknown'
                cleaned_df[column].fillna("Unknown", inplace=True)
            else:
                # For numeric columns, fill with median
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)

    # Remove duplicates if any
    original_shape = cleaned_df.shape
    cleaned_df.drop_duplicates(subset=["track_id"], keep="first", inplace=True)
    if original_shape[0] > cleaned_df.shape[0]:
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} duplicate tracks")

    return cleaned_df


def create_feature_matrix(df):
    """
    Create a feature matrix for content-based recommendations
    """
    print("\n=== Creating Feature Matrix ===")

    # Select audio features
    audio_features = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        # Adding previously categorical features as numeric
        "key",
        "mode",
        "time_signature",
    ]

    # Scale numeric features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[audio_features])

    # Save preprocessing objects for future use
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "preprocessing.pkl"), "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "audio_features": audio_features,
            },
            f,
        )
    print(f"Preprocessing objects saved to {models_dir}/preprocessing.pkl")

    # Feature matrix is just the scaled features
    feature_matrix = scaled_features

    # Print feature matrix shape
    print(f"Feature matrix shape: {feature_matrix.shape}")

    return feature_matrix, {"scaler": scaler}


def save_processed_data(df, feature_matrix, output_dir=None):
    """
    Save processed data for later use
    """
    if output_dir is None:
        # Use absolute path
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )

    print(f"\n=== Saving Processed Data to {output_dir} ===")

    os.makedirs(output_dir, exist_ok=True)

    # Save cleaned dataframe
    df.to_csv(os.path.join(output_dir, "cleaned_tracks.csv"), index=False)
    print(f"Saved cleaned dataset to {os.path.join(output_dir, 'cleaned_tracks.csv')}")

    # Save feature matrix as numpy array with allow_pickle=True
    np.save(
        os.path.join(output_dir, "feature_matrix.npy"),
        feature_matrix,
        allow_pickle=True,
    )
    print(f"Saved feature matrix to {os.path.join(output_dir, 'feature_matrix.npy')}")

    # Save track IDs for mapping
    df[["track_id"]].to_csv(os.path.join(output_dir, "track_ids.csv"), index=False)
    print(f"Saved track IDs to {os.path.join(output_dir, 'track_ids.csv')}")


def main():
    # Load data from Hugging Face
    tracks_df = load_data()
    if tracks_df is None:
        return

    # Explore data
    explore_data(tracks_df)

    # Clean data
    cleaned_df = clean_data(tracks_df)

    # Split data into train and test sets for evaluation
    train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=42)
    print(
        f"Split data into {train_df.shape[0]} training samples and {test_df.shape[0]} test samples"
    )

    # Create feature matrix for recommendations
    feature_matrix, preprocessors = create_feature_matrix(train_df)

    # Get the absolute data directory path
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )

    # Save processed data
    save_processed_data(cleaned_df, feature_matrix, data_dir)

    # Also save the train/test split for evaluation
    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(os.path.join(data_dir, "train_set.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test_set.csv"), index=False)
    print("Saved train/test splits for evaluation")

    print("\nData processing completed successfully!")


if __name__ == "__main__":
    main()
