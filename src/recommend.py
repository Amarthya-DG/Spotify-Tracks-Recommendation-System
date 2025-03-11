import os
import sys

import numpy as np
import pandas as pd

from recommendation_models import ContentBasedRecommender, GenreRecommender, load_models


class SpotifyRecommendationSystem:
    """
    Main recommendation system interface that combines multiple recommendation approaches
    """

    def __init__(self, data_dir="../data", model_dir="../models"):
        """
        Initialize the recommendation system

        Parameters:
        -----------
        data_dir : str
            Directory containing processed data
        model_dir : str
            Directory containing trained models
        """
        self.data_dir = data_dir
        self.model_dir = model_dir

        # Try to load existing models
        content_recommender, popularity_recommender, genre_recommender = load_models(
            model_dir=model_dir, data_dir=data_dir
        )

        # If models failed to load, train new ones
        if content_recommender is None:
            print("Models not found. Training new models...")
            self._train_models()
            content_recommender, popularity_recommender, genre_recommender = (
                load_models(model_dir=model_dir, data_dir=data_dir)
            )

        self.content_recommender = content_recommender
        self.popularity_recommender = popularity_recommender
        self.genre_recommender = genre_recommender

        # Load the track dataframe
        self.track_df = pd.read_csv(f"{data_dir}/cleaned_tracks.csv")

        print("Recommendation system initialized successfully!")

    def _train_models(self):
        """
        Train recommendation models from processed data
        """
        try:
            # Load data
            track_df = pd.read_csv(f"{self.data_dir}/cleaned_tracks.csv")
            feature_matrix = np.load(
                f"{self.data_dir}/feature_matrix.npy", allow_pickle=True
            )

            # Extract just the audio features for content-based filtering for consistency
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

            # Use just audio features for the content-based recommender
            content_features = track_df[audio_features].values

            # Create and train recommenders
            content_recommender = ContentBasedRecommender(content_features, track_df)
            content_recommender.fit(n_neighbors=10, algorithm="auto", metric="cosine")

            genre_recommender = GenreRecommender(track_df)
            genre_recommender.fit()

            print("Models trained successfully!")

        except FileNotFoundError:
            print(
                f"Processed data not found in {self.data_dir}. Please run data_processing.py first."
            )
            sys.exit(1)
        except Exception as e:
            print(f"Error training models: {e}")
            sys.exit(1)

    def get_track_info(self, track_id=None, track_name=None, artist=None):
        """
        Get information about a specific track

        Parameters:
        -----------
        track_id : str, optional
            Spotify track ID
        track_name : str, optional
            Name of the track
        artist : str, optional
            Artist name

        Returns:
        --------
        pandas.Series
            Track information
        """
        if track_id:
            track = self.track_df[self.track_df["track_id"] == track_id]
        elif track_name and artist:
            track = self.track_df[
                (self.track_df["track_name"].str.contains(track_name, case=False))
                & (self.track_df["artists"].str.contains(artist, case=False))
            ]
        elif track_name:
            track = self.track_df[
                self.track_df["track_name"].str.contains(track_name, case=False)
            ]
        else:
            print(
                "Please provide either track_id, track_name, or track_name and artist"
            )
            return None

        if len(track) == 0:
            print("No matching track found")
            return None
        elif len(track) > 1:
            print(f"Found {len(track)} matching tracks. Showing the first one:")
            return track.iloc[0]
        else:
            return track.iloc[0]

    def search_tracks(self, query, limit=5):
        """
        Search for tracks by name or artist

        Parameters:
        -----------
        query : str
            Search query
        limit : int
            Maximum number of results to return

        Returns:
        --------
        pandas.DataFrame
            Matching tracks
        """
        # Search in track name and artist
        matches = self.track_df[
            (self.track_df["track_name"].str.contains(query, case=False))
            | (self.track_df["artists"].str.contains(query, case=False))
        ]

        if len(matches) == 0:
            print(f"No tracks found matching '{query}'")
            return None

        # Sort by popularity and return limited results
        return matches.sort_values(by="popularity", ascending=False).head(limit)[
            [
                "track_id",
                "track_name",
                "artists",
                "album_name",
                "popularity",
                "track_genre",
            ]
        ]

    def recommend_by_track(self, track_id, n_recommendations=5):
        """
        Get recommendations based on a specific track

        Parameters:
        -----------
        track_id : str
            Spotify track ID
        n_recommendations : int
            Number of recommendations to return

        Returns:
        --------
        pandas.DataFrame
            Recommended tracks
        """
        recommendations = self.content_recommender.recommend_by_track_id(
            track_id, n_recommendations=n_recommendations
        )

        # If recommendations failed, use popularity-based recommendations as fallback
        if recommendations is None or len(recommendations) == 0:
            print(
                f"Content-based recommendations failed for track {track_id}, using popularity fallback"
            )
            # Get the track genre
            track_genre = None
            track_info = self.track_df[self.track_df["track_id"] == track_id]
            if not track_info.empty:
                track_genre = track_info.iloc[0]["track_genre"]

            # Get popular recommendations in the same genre
            recommendations = self.popularity_recommender.recommend_popular(
                genre=track_genre, n_recommendations=n_recommendations
            )

            # Add similarity score column for consistency
            if recommendations is not None and not recommendations.empty:
                recommendations["similarity_score"] = (
                    0.5  # Default score for popularity-based
                )

        # Ensure we have a DataFrame with a similarity_score column
        if recommendations is None:
            # Create an empty DataFrame with the required columns
            recommendations = pd.DataFrame(
                columns=[
                    "track_id",
                    "track_name",
                    "artists",
                    "album_name",
                    "popularity",
                    "track_genre",
                    "similarity_score",
                ]
            )
        elif "similarity_score" not in recommendations.columns:
            recommendations["similarity_score"] = 0.5  # Default score

        return recommendations

    def recommend_popular(self, genre=None, n_recommendations=10):
        """
        Get popular track recommendations, optionally by genre

        Parameters:
        -----------
        genre : str, optional
            Genre to filter by
        n_recommendations : int
            Number of recommendations to return

        Returns:
        --------
        pandas.DataFrame
            Popular tracks
        """
        return self.popularity_recommender.recommend_popular(
            n_recommendations=n_recommendations, genre=genre
        )

    def recommend_by_genre(self, genre, n_recommendations=10):
        """
        Get recommendations from similar genres

        Parameters:
        -----------
        genre : str
            Genre to base recommendations on
        n_recommendations : int
            Number of recommendations to return

        Returns:
        --------
        pandas.DataFrame
            Recommended tracks from similar genres
        """
        return self.genre_recommender.recommend_by_genre(
            genre, n_recommendations=n_recommendations
        )

    def get_available_genres(self):
        """
        Get a list of all available genres in the dataset

        Returns:
        --------
        list
            Available genres
        """
        return sorted(self.track_df["track_genre"].unique().tolist())

    def recommend_for_user_profile(self, liked_tracks, n_recommendations=10):
        """
        Generate recommendations based on a user's liked tracks

        Parameters:
        -----------
        liked_tracks : list
            List of track IDs that the user likes
        n_recommendations : int
            Number of recommendations to return

        Returns:
        --------
        pandas.DataFrame
            Personalized track recommendations
        """
        # Check if we have valid track IDs
        valid_tracks = [
            track_id
            for track_id in liked_tracks
            if track_id in self.track_df["track_id"].values
        ]

        if not valid_tracks:
            print("No valid track IDs provided")
            return None

        # Get recommendations for each track
        all_recommendations = pd.DataFrame()

        for track_id in valid_tracks:
            track_recs = self.recommend_by_track(track_id, n_recommendations=3)
            if track_recs is not None:
                all_recommendations = pd.concat([all_recommendations, track_recs])

        # Remove duplicates and sort by similarity score
        all_recommendations = all_recommendations.drop_duplicates(subset=["track_id"])
        all_recommendations = all_recommendations.sort_values(
            by="similarity_score", ascending=False
        )

        # Remove any tracks that were in the liked_tracks list
        all_recommendations = all_recommendations[
            ~all_recommendations["track_id"].isin(liked_tracks)
        ]

        return all_recommendations.head(n_recommendations)


def print_separator():
    """Print a separator line"""
    print("\n" + "=" * 80 + "\n")


def display_track_info(track):
    """
    Display detailed information about a track

    Parameters:
    -----------
    track : pandas.Series
        Track information
    """
    if track is None:
        return

    print_separator()
    print(f"Track: {track['track_name']}")
    print(f"Artist: {track['artists']}")
    print(f"Album: {track['album_name']}")
    print(f"Genre: {track['track_genre']}")
    print(f"Popularity: {track['popularity']}/100")
    print(f"Track ID: {track['track_id']}")

    # Display audio features
    print("\nAudio Features:")
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

    for feature in audio_features:
        if feature in track:
            print(f"  {feature}: {track[feature]}")

    print_separator()


def display_recommendations(recommendations, title):
    """
    Display a table of track recommendations

    Parameters:
    -----------
    recommendations : pandas.DataFrame
        DataFrame of recommended tracks
    title : str
        Title for the recommendations
    """
    if recommendations is None or len(recommendations) == 0:
        print("No recommendations available")
        return

    print_separator()
    print(f"{title} ({len(recommendations)} tracks)")
    print_separator()

    # Format for display
    for i, (_, track) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {track['track_name']} by {track['artists']}")
        print(f"   Album: {track['album_name']}")
        print(f"   Genre: {track['track_genre']}")
        print(f"   Popularity: {track['popularity']}/100")
        if "similarity_score" in track:
            print(f"   Similarity: {track['similarity_score']:.2f}")
        print(f"   Track ID: {track['track_id']}")
        print()

    print_separator()


def interactive_cli(recommendation_system):
    """
    Run an interactive command-line interface for the recommendation system

    Parameters:
    -----------
    recommendation_system : SpotifyRecommendationSystem
        The recommendation system instance
    """
    print("\nWelcome to the Spotify Track Recommendation System!")
    print("Type 'help' to see available commands.")

    # Keep track of liked tracks for user profile
    liked_tracks = []

    while True:
        print_separator()
        command = input("Enter command: ").strip().lower()

        if command == "exit" or command == "quit":
            print("Thank you for using the recommendation system. Goodbye!")
            break

        elif command == "help":
            print("\nAvailable commands:")
            print("  search <query>         - Search for tracks by name or artist")
            print("  info <track_id>        - Get detailed information about a track")
            print("  recommend <track_id>   - Get recommendations based on a track")
            print("  popular                - Get popular tracks")
            print("  popular <genre>        - Get popular tracks in a specific genre")
            print("  genres                 - List all available genres")
            print("  genre <genre>          - Get recommendations for a genre")
            print("  like <track_id>        - Add a track to your liked tracks")
            print("  liked                  - Show your liked tracks")
            print(
                "  for_me                 - Get personalized recommendations based on liked tracks"
            )
            print("  exit                   - Exit the program")

        elif command.startswith("search "):
            query = command[7:].strip()
            results = recommendation_system.search_tracks(query, limit=5)
            if results is not None:
                display_recommendations(results, f"Search Results for '{query}'")

        elif command.startswith("info "):
            track_id = command[5:].strip()
            track = recommendation_system.get_track_info(track_id=track_id)
            display_track_info(track)

        elif command.startswith("recommend "):
            track_id = command[10:].strip()
            recommendations = recommendation_system.recommend_by_track(
                track_id, n_recommendations=5
            )

            # Get the track name for the title
            track = recommendation_system.get_track_info(track_id=track_id)
            track_name = track["track_name"] if track is not None else track_id

            display_recommendations(
                recommendations, f"Recommendations based on '{track_name}'"
            )

        elif command == "popular":
            recommendations = recommendation_system.recommend_popular(
                n_recommendations=10
            )
            display_recommendations(recommendations, "Most Popular Tracks")

        elif command.startswith("popular "):
            genre = command[8:].strip()
            recommendations = recommendation_system.recommend_popular(
                genre=genre, n_recommendations=10
            )
            display_recommendations(
                recommendations, f"Most Popular Tracks in '{genre}'"
            )

        elif command == "genres":
            genres = recommendation_system.get_available_genres()
            print("\nAvailable Genres:")
            for i, genre in enumerate(genres, 1):
                print(f"{i}. {genre}")

        elif command.startswith("genre "):
            genre = command[6:].strip()
            recommendations = recommendation_system.recommend_by_genre(
                genre, n_recommendations=10
            )
            display_recommendations(
                recommendations, f"Recommendations for Genre '{genre}'"
            )

        elif command.startswith("like "):
            track_id = command[5:].strip()
            track = recommendation_system.get_track_info(track_id=track_id)

            if track is not None:
                if track_id not in liked_tracks:
                    liked_tracks.append(track_id)
                    print(f"Added '{track['track_name']}' to your liked tracks")
                else:
                    print(f"'{track['track_name']}' is already in your liked tracks")

        elif command == "liked":
            if not liked_tracks:
                print("You haven't liked any tracks yet")
            else:
                print("\nYour Liked Tracks:")
                for i, track_id in enumerate(liked_tracks, 1):
                    track = recommendation_system.get_track_info(track_id=track_id)
                    if track is not None:
                        print(f"{i}. {track['track_name']} by {track['artists']}")

        elif command == "for_me":
            if not liked_tracks:
                print("You need to like some tracks first using the 'like' command")
            else:
                recommendations = recommendation_system.recommend_for_user_profile(
                    liked_tracks, n_recommendations=10
                )
                display_recommendations(recommendations, "Personalized Recommendations")

        else:
            print("Unknown command. Type 'help' to see available commands.")


def main():
    # Check if the required directories and files exist
    data_dir = "../data"

    if not os.path.exists(f"{data_dir}/cleaned_tracks.csv"):
        print(f"Processed data not found in {data_dir}.")
        print("Please run data_processing.py first to prepare the dataset.")

        choice = (
            input("Would you like to process the data now? (y/n): ").strip().lower()
        )
        if choice == "y":
            # Import data_processing module
            sys.path.append("../src")
            import data_processing

            data_processing.main()
        else:
            print("Exiting. Please run data_processing.py before using this system.")
            sys.exit(1)

    # Initialize the recommendation system
    recommendation_system = SpotifyRecommendationSystem()

    # Launch the interactive CLI
    interactive_cli(recommendation_system)


if __name__ == "__main__":
    main()
