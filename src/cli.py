#!/usr/bin/env python
import argparse
import os

from recommend import SpotifyRecommendationSystem


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Spotify Track Recommendation System")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for tracks")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Get track information")
    info_parser.add_argument("track_id", type=str, help="Spotify track ID")

    # Recommend by track command
    recommend_track_parser = subparsers.add_parser(
        "track", help="Get recommendations based on a track"
    )
    recommend_track_parser.add_argument("track_id", type=str, help="Spotify track ID")
    recommend_track_parser.add_argument(
        "--count", type=int, default=5, help="Number of recommendations"
    )

    # Recommend popular tracks command
    popular_parser = subparsers.add_parser("popular", help="Get popular tracks")
    popular_parser.add_argument("--genre", type=str, help="Filter by genre (optional)")
    popular_parser.add_argument(
        "--count", type=int, default=10, help="Number of recommendations"
    )

    # Recommend by genre command
    genre_parser = subparsers.add_parser(
        "genre", help="Get recommendations for a genre"
    )
    genre_parser.add_argument("genre", type=str, help="Genre name")
    genre_parser.add_argument(
        "--count", type=int, default=10, help="Number of recommendations"
    )

    # List genres command
    subparsers.add_parser("genres", help="List all available genres")

    # Recommend for user profile command
    profile_parser = subparsers.add_parser(
        "profile", help="Get personalized recommendations"
    )
    profile_parser.add_argument(
        "track_ids", type=str, nargs="+", help="List of track IDs you like"
    )
    profile_parser.add_argument(
        "--count", type=int, default=10, help="Number of recommendations"
    )

    # Interactive mode
    subparsers.add_parser(
        "interactive", help="Start interactive recommendation session"
    )

    # Process argument
    subparsers.add_parser("process", help="Process the dataset")

    return parser.parse_args()


def display_track_info(track):
    """Display detailed information about a track"""
    if track is None:
        return

    print("\n" + "=" * 80 + "\n")
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

    print("\n" + "=" * 80 + "\n")


def display_recommendations(recommendations, title):
    """Display a table of track recommendations"""
    if recommendations is None or len(recommendations) == 0:
        print("No recommendations available")
        return

    print("\n" + "=" * 80 + "\n")
    print(f"{title} ({len(recommendations)} tracks)")
    print("\n" + "=" * 80 + "\n")

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

    print("=" * 80 + "\n")


def main():
    """
    Main function to handle command line interface
    """
    args = parse_arguments()

    # If no command is provided, show help
    if args.command is None:
        print("Please specify a command. Use --help for more information.")
        return

    # Process dataset command
    if args.command == "process":
        print("Processing dataset...")
        import data_processing

        data_processing.main()
        return

    # For all other commands, we need the recommendation system
    # Check if the data has been processed
    data_dir = "../data"
    if not os.path.exists(f"{data_dir}/cleaned_tracks.csv"):
        print(f"Processed data not found in {data_dir}.")
        print("Please run the 'process' command first to prepare the dataset.")
        return

    # Initialize the recommendation system
    try:
        recommendation_system = SpotifyRecommendationSystem()
    except Exception as e:
        print(f"Error initializing recommendation system: {e}")
        return

    # Handle different commands
    if args.command == "search":
        results = recommendation_system.search_tracks(args.query, limit=args.limit)
        display_recommendations(results, f"Search Results for '{args.query}'")

    elif args.command == "info":
        track = recommendation_system.get_track_info(track_id=args.track_id)
        display_track_info(track)

    elif args.command == "track":
        recommendations = recommendation_system.recommend_by_track(
            args.track_id, n_recommendations=args.count
        )

        # Get the track name for the title
        track = recommendation_system.get_track_info(track_id=args.track_id)
        track_name = track["track_name"] if track is not None else args.track_id

        display_recommendations(
            recommendations, f"Recommendations based on '{track_name}'"
        )

    elif args.command == "popular":
        recommendations = recommendation_system.recommend_popular(
            genre=args.genre, n_recommendations=args.count
        )

        title = "Most Popular Tracks"
        if args.genre:
            title += f" in '{args.genre}'"

        display_recommendations(recommendations, title)

    elif args.command == "genre":
        recommendations = recommendation_system.recommend_by_genre(
            args.genre, n_recommendations=args.count
        )

        display_recommendations(
            recommendations, f"Recommendations for Genre '{args.genre}'"
        )

    elif args.command == "genres":
        genres = recommendation_system.get_available_genres()
        print("\nAvailable Genres:")
        for i, genre in enumerate(genres, 1):
            print(f"{i}. {genre}")

    elif args.command == "profile":
        recommendations = recommendation_system.recommend_for_user_profile(
            args.track_ids, n_recommendations=args.count
        )

        display_recommendations(recommendations, "Personalized Recommendations")

    elif args.command == "interactive":
        from recommend import interactive_cli

        interactive_cli(recommendation_system)


if __name__ == "__main__":
    main()
