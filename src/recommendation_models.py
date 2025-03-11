import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class ContentBasedRecommender:
    """
    Content-based recommendation system that uses track audio features
    to recommend similar tracks
    """

    def __init__(self, feature_matrix, track_df):
        """
        Initialize the recommender with feature matrix and track dataframe

        Parameters:
        -----------
        feature_matrix : numpy.ndarray
            The feature matrix of audio features
        track_df : pandas.DataFrame
            DataFrame containing track information
        """
        # Ensure feature_matrix is a dense numpy array
        if (
            not isinstance(feature_matrix, np.ndarray)
            or not feature_matrix.flags["C_CONTIGUOUS"]
        ):
            print("Converting feature matrix to dense numpy array...")
            self.feature_matrix = np.array(feature_matrix, order="C")
        else:
            self.feature_matrix = feature_matrix

        self.track_df = track_df
        self.model = None

    def fit(self, n_neighbors=10, algorithm="auto", metric="cosine"):
        """
        Fit the recommendation model

        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to use for kNN
        algorithm : str
            Algorithm to use for nearest neighbors
        metric : str
            Distance metric to use
        """
        print("Training content-based recommendation model...")

        # Extract only audio features for the model if feature_matrix is too large
        if self.feature_matrix.shape[1] > 50:
            print("Feature matrix has many columns, extracting only audio features...")
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
                "key",
                "mode",
                "time_signature",
            ]
            # Check if all features exist in the dataframe
            if all(feature in self.track_df.columns for feature in audio_features):
                # Use just the audio features
                feature_matrix = self.track_df[audio_features].values
                self.feature_matrix = feature_matrix

        self.model = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm=algorithm, metric=metric
        )
        self.model.fit(self.feature_matrix)
        print("Content-based model trained successfully!")

        # Save the model using absolute path
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "content_based_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_path}")

    def recommend_by_track_id(self, track_id, n_recommendations=5):
        """
        Recommend tracks similar to the specified track

        Parameters:
        -----------
        track_id : str
            The Spotify track ID
        n_recommendations : int
            Number of recommendations to return

        Returns:
        --------
        pandas.DataFrame
            DataFrame of recommended tracks
        """
        # Find the index of the track
        track_idx = self.track_df[self.track_df["track_id"] == track_id].index

        if len(track_idx) == 0:
            print(f"Track ID {track_id} not found in the dataset")
            return None

        track_idx = track_idx[0]

        try:
            # Define the standard audio features we use
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
                "key",
                "mode",
                "time_signature",
            ]

            # Check if our model was trained on a subset of features
            if self.feature_matrix.shape[1] < 50:
                # The model was likely trained on just the audio features
                # Extract only audio features from the track
                track_features = self.track_df.iloc[track_idx][
                    audio_features
                ].values.reshape(1, -1)
            else:
                # Use the full feature matrix as during training
                track_features = self.feature_matrix[track_idx].reshape(1, -1)

            # Find nearest neighbors
            distances, indices = self.model.kneighbors(
                track_features, n_neighbors=n_recommendations + 1
            )

            # Skip the first result (the track itself)
            indices = indices.flatten()[1:]
            distances = distances.flatten()[1:]

            # Get recommended tracks
            recommended_tracks = self.track_df.iloc[indices][
                [
                    "track_id",
                    "track_name",
                    "artists",
                    "album_name",
                    "popularity",
                    "track_genre",
                ]
            ].copy()

            # Add similarity score
            recommended_tracks["similarity_score"] = 1 - distances

            return recommended_tracks
        except Exception as e:
            print(f"Error recommending similar tracks: {e}")

            # Fallback: try using audio features directly
            try:
                print("Attempting fallback recommendation using audio features...")
                track_row = self.track_df.iloc[track_idx]
                features_dict = {
                    feature: track_row[feature] for feature in audio_features
                }
                return self.recommend_by_features(
                    features_dict, n_recommendations=n_recommendations
                )
            except Exception as e2:
                print(f"Fallback recommendation also failed: {e2}")
                return None

    def recommend_by_features(self, features_dict, n_recommendations=5):
        """
        Recommend tracks based on specified audio features

        Parameters:
        -----------
        features_dict : dict
            Dictionary of audio features and their values
        n_recommendations : int
            Number of recommendations to return

        Returns:
        --------
        pandas.DataFrame
            DataFrame of recommended tracks
        """
        # Extract the audio features we can use
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
            "key",
            "mode",
            "time_signature",
        ]

        # Filter features that are in our model
        valid_features = {k: v for k, v in features_dict.items() if k in audio_features}

        if not valid_features:
            print("No valid audio features provided")
            return None

        # Create feature vector using the same features as in training
        feature_vector = np.zeros(len(audio_features))

        # Fill in the feature vector with provided values
        for i, feature in enumerate(audio_features):
            if feature in valid_features:
                feature_vector[i] = valid_features[feature]

        # Scale the feature vector (similar to how training data was scaled)
        # In a production system, we would load the scaler from the model
        # For simplicity, we'll use the feature vector as is

        # Reshape to 2D for sklearn
        feature_vector = feature_vector.reshape(1, -1)

        try:
            # Find nearest neighbors
            distances, indices = self.model.kneighbors(
                feature_vector, n_neighbors=n_recommendations
            )

            # Get recommended tracks
            indices = indices.flatten()
            distances = distances.flatten()

            if len(indices) == 0:
                return None

            recommended_tracks = self.track_df.iloc[indices][
                [
                    "track_id",
                    "track_name",
                    "artists",
                    "album_name",
                    "popularity",
                    "track_genre",
                ]
            ].copy()

            # Add similarity score
            recommended_tracks["similarity_score"] = 1 - distances

            return recommended_tracks
        except Exception as e:
            print(f"Error recommending by features: {e}")
            return None


class PopularityRecommender:
    """
    Recommender that suggests the most popular tracks by genre or overall
    """

    def __init__(self, track_df):
        """
        Initialize with track dataframe

        Parameters:
        -----------
        track_df : pandas.DataFrame
            DataFrame containing track information
        """
        self.track_df = track_df

    def recommend_popular(self, n_recommendations=10, genre=None):
        """
        Recommend popular tracks, optionally filtered by genre

        Parameters:
        -----------
        n_recommendations : int
            Number of recommendations to return
        genre : str, optional
            Genre to filter by

        Returns:
        --------
        pandas.DataFrame
            DataFrame of recommended tracks
        """
        if genre:
            # Filter by genre
            genre_tracks = self.track_df[self.track_df["track_genre"] == genre]
            if len(genre_tracks) == 0:
                print(f"No tracks found for genre: {genre}")
                return None

            recommendations = genre_tracks.sort_values(
                by="popularity", ascending=False
            ).head(n_recommendations)
        else:
            # Overall popularity
            recommendations = self.track_df.sort_values(
                by="popularity", ascending=False
            ).head(n_recommendations)

        return recommendations[
            [
                "track_id",
                "track_name",
                "artists",
                "album_name",
                "popularity",
                "track_genre",
            ]
        ]


class GenreRecommender:
    """
    Recommender that suggests tracks from similar genres
    """

    def __init__(self, track_df):
        """
        Initialize with track dataframe

        Parameters:
        -----------
        track_df : pandas.DataFrame
            DataFrame containing track information
        """
        self.track_df = track_df
        self.genre_similarity = None
        self.genre_to_idx = None
        self.idx_to_genre = None

    def fit(self):
        """
        Compute genre similarity based on audio features
        """
        # Get unique genres
        genres = self.track_df["track_genre"].unique()

        # Compute average audio features for each genre
        genre_features = {}

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
        ]

        for genre in genres:
            genre_tracks = self.track_df[self.track_df["track_genre"] == genre]
            if not genre_tracks.empty:  # Ensure we have tracks for this genre
                genre_features[genre] = genre_tracks[audio_features].mean().values

        # Create genre feature matrix
        self.genre_to_idx = {genre: i for i, genre in enumerate(genre_features.keys())}
        self.idx_to_genre = {i: genre for genre, i in self.genre_to_idx.items()}

        genre_feature_matrix = np.array(list(genre_features.values()))

        # Compute similarity between genres
        self.genre_similarity = cosine_similarity(genre_feature_matrix)

        print(f"Genre similarity matrix computed for {len(genre_features)} genres")

        # Save the model using absolute path
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, "genre_recommender.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "genre_similarity": self.genre_similarity,
                    "genre_to_idx": self.genre_to_idx,
                    "idx_to_genre": self.idx_to_genre,
                },
                f,
            )
        print(f"Genre model saved to {model_path}")

    def get_similar_genres(self, genre, n=5):
        """
        Get genres similar to the specified genre

        Parameters:
        -----------
        genre : str
            The genre to find similar genres for
        n : int
            Number of similar genres to return

        Returns:
        --------
        list
            List of similar genres with similarity scores
        """
        if self.genre_similarity is None or self.genre_to_idx is None:
            print("Model not trained. Call fit() first.")
            return None

        if genre not in self.genre_to_idx:
            print(f"Genre {genre} not found in the dataset")
            return None

        # Get genre index
        genre_idx = self.genre_to_idx[genre]

        # Get similarity scores for this genre
        similarity_scores = self.genre_similarity[genre_idx]

        # Get indices of most similar genres (excluding itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1 : n + 1]

        # Get similar genres with scores
        similar_genres = [
            (self.idx_to_genre[idx], similarity_scores[idx]) for idx in similar_indices
        ]

        return similar_genres

    def recommend_by_genre(self, genre, n_recommendations=10):
        """
        Recommend tracks from genres similar to the specified genre

        Parameters:
        -----------
        genre : str
            The genre to base recommendations on
        n_recommendations : int
            Number of recommendations to return

        Returns:
        --------
        pandas.DataFrame
            DataFrame of recommended tracks
        """
        # Get similar genres
        similar_genres = self.get_similar_genres(genre, n=3)

        if similar_genres is None:
            return None

        # Get top tracks from each similar genre
        recommendations = pd.DataFrame()

        # Include some tracks from the original genre
        original_genre_tracks = (
            self.track_df[self.track_df["track_genre"] == genre]
            .sort_values(by="popularity", ascending=False)
            .head(n_recommendations // 2)
        )

        recommendations = pd.concat([recommendations, original_genre_tracks])

        # Add tracks from similar genres
        tracks_per_genre = n_recommendations // (len(similar_genres) + 1)

        for similar_genre, similarity in similar_genres:
            genre_tracks = (
                self.track_df[self.track_df["track_genre"] == similar_genre]
                .sort_values(by="popularity", ascending=False)
                .head(tracks_per_genre)
            )

            recommendations = pd.concat([recommendations, genre_tracks])

        # Ensure we don't have more than requested
        recommendations = recommendations.head(n_recommendations)

        return recommendations[
            [
                "track_id",
                "track_name",
                "artists",
                "album_name",
                "popularity",
                "track_genre",
            ]
        ]


def load_models(model_dir=None, data_dir=None):
    """
    Load saved models and data

    Returns:
    --------
    tuple
        (ContentBasedRecommender, PopularityRecommender, GenreRecommender)
    """
    # Use absolute paths if not provided
    if model_dir is None:
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )

    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )

    try:
        # Load data
        track_df = pd.read_csv(os.path.join(data_dir, "cleaned_tracks.csv"))

        # Define standard audio features
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
            "key",
            "mode",
            "time_signature",
        ]

        # Extract just audio features for consistency
        content_features = track_df[audio_features].values

        content_recommender = ContentBasedRecommender(content_features, track_df)
        popularity_recommender = PopularityRecommender(track_df)
        genre_recommender = GenreRecommender(track_df)

        # Load content-based model if available
        content_model_path = os.path.join(model_dir, "content_based_model.pkl")
        try:
            with open(content_model_path, "rb") as f:
                content_model = pickle.load(f)
                content_recommender.model = content_model
                print(f"Loaded content-based model from {content_model_path}")
        except (FileNotFoundError, EOFError) as e:
            print(f"Content-based model not found or corrupt, will need to train: {e}")
            content_recommender.fit()

        # Load genre recommender model if available
        genre_model_path = os.path.join(model_dir, "genre_recommender.pkl")
        try:
            with open(genre_model_path, "rb") as f:
                genre_model = pickle.load(f)
                genre_recommender.genre_similarity = genre_model["genre_similarity"]
                genre_recommender.genre_to_idx = genre_model["genre_to_idx"]
                genre_recommender.idx_to_genre = genre_model["idx_to_genre"]
                print(f"Loaded genre model from {genre_model_path}")
        except (FileNotFoundError, EOFError) as e:
            print(f"Genre model not found or corrupt, will need to train: {e}")
            genre_recommender.fit()

        return content_recommender, popularity_recommender, genre_recommender

    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None
