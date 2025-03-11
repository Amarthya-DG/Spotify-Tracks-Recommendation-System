import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from recommendation_models import (
    ContentBasedRecommender,
    GenreRecommender,
    PopularityRecommender,
)


def precision_at_k(recommended_items, relevant_items, k):
    """
    Calculate precision@k for a single user

    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    relevant_items : list
        List of item IDs that are relevant to the user
    k : int
        Number of recommendations to consider

    Returns:
    --------
    float
        Precision@k score
    """
    if len(recommended_items) == 0:
        return 0

    # Consider only top k recommendations
    recommended_items = recommended_items[:k]

    # Calculate precision
    num_relevant = len(set(recommended_items) & set(relevant_items))
    precision = num_relevant / min(k, len(recommended_items))

    return precision


def recall_at_k(recommended_items, relevant_items, k):
    """
    Calculate recall@k for a single user

    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    relevant_items : list
        List of item IDs that are relevant to the user
    k : int
        Number of recommendations to consider

    Returns:
    --------
    float
        Recall@k score
    """
    if len(relevant_items) == 0:
        return 0

    # Consider only top k recommendations
    recommended_items = recommended_items[:k]

    # Calculate recall
    num_relevant = len(set(recommended_items) & set(relevant_items))
    recall = num_relevant / len(relevant_items)

    return recall


def ndcg_at_k(recommended_items, relevant_items, k):
    """
    Calculate normalized discounted cumulative gain (NDCG) at k for a single user

    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    relevant_items : list
        List of item IDs that are relevant to the user
    k : int
        Number of recommendations to consider

    Returns:
    --------
    float
        NDCG@k score
    """
    if len(relevant_items) == 0:
        return 0

    # Consider only top k recommendations
    recommended_items = recommended_items[:k]

    # Calculate DCG
    dcg = 0
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            # Use binary relevance (1 if relevant, 0 if not)
            rel = 1
            dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed

    # Calculate ideal DCG (IDCG)
    idcg = 0
    for i in range(min(k, len(relevant_items))):
        idcg += 1 / np.log2(i + 2)

    # Calculate NDCG
    ndcg = dcg / idcg if idcg > 0 else 0

    return ndcg


def diversity(recommended_items, item_features):
    """
    Calculate diversity of recommendations based on pairwise distance

    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    item_features : dict
        Dictionary mapping item IDs to feature vectors

    Returns:
    --------
    float
        Diversity score (higher means more diverse)
    """
    if len(recommended_items) <= 1:
        return 0

    # Get feature vectors for recommended items
    feature_vectors = []
    for item in recommended_items:
        if item in item_features:
            feature_vectors.append(item_features[item])

    if len(feature_vectors) <= 1:
        return 0

    # Calculate pairwise cosine similarity
    feature_matrix = np.array(feature_vectors)
    sim_matrix = cosine_similarity(feature_matrix)

    # Calculate average distance (1 - similarity)
    n = sim_matrix.shape[0]
    total_distance = 0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_distance += 1 - sim_matrix[i, j]
            count += 1

    avg_distance = total_distance / count if count > 0 else 0

    return avg_distance


def novelty(recommended_items, item_popularity):
    """
    Calculate novelty of recommendations based on inverse popularity

    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    item_popularity : dict
        Dictionary mapping item IDs to popularity scores

    Returns:
    --------
    float
        Novelty score (higher means more novel)
    """
    if len(recommended_items) == 0:
        return 0

    # Calculate average inverse popularity
    total_novelty = 0
    count = 0

    for item in recommended_items:
        if item in item_popularity:
            # Use inverse popularity as a measure of novelty
            pop = item_popularity[item]
            if pop > 0:  # Avoid division by zero
                total_novelty += 1 / pop
                count += 1

    avg_novelty = total_novelty / count if count > 0 else 0

    return avg_novelty


def evaluate_content_based(train_df, test_df, feature_matrix, k=10):
    """
    Evaluate content-based recommendations

    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training set DataFrame
    test_df : pandas.DataFrame
        Test set DataFrame
    feature_matrix : numpy.ndarray
        Feature matrix for content-based filtering
    k : int
        Number of recommendations to consider

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    print("Evaluating content-based recommendations...")

    # Scale features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    recommender = ContentBasedRecommender(feature_matrix, train_df)
    recommender.fit(n_neighbors=k + 1)  # Match evaluation context

    # Sample tracks from test set to evaluate
    sample_size = min(100, len(test_df))
    test_samples = test_df.sample(sample_size)

    # Create item features dictionary for diversity calculation
    item_features = {}
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
        "time_signature",  # Add these three features
    ]
    for _, row in train_df.iterrows():
        item_features[row["track_id"]] = row[audio_features].values

    # Prepare data structures
    recommended_items = {}
    relevant_items = {}
    item_popularity = {}

    # Get popularity values for novelty calculation
    max_popularity = train_df["popularity"].max()
    for _, row in train_df.iterrows():
        item_popularity[row["track_id"]] = row["popularity"] / max_popularity

    # Evaluate recommendations for each test track
    for _, test_row in test_samples.iterrows():
        test_track_id = test_row["track_id"]
        test_track_features = test_row[audio_features].values
        test_track_features = scaler.transform([test_track_features])[
            0
        ]  # Scale test features

        # Get content-based recommendations
        recommendations = recommender.recommend_by_features(
            dict(zip(audio_features, test_track_features)), n_recommendations=k
        )
        if recommendations is not None and not recommendations.empty:
            recommended_items[test_track_id] = recommendations["track_id"].tolist()
        else:
            recommended_items[test_track_id] = []

        # Define relevant items as top k nearest neighbors from the model
        distances, indices = recommender.model.kneighbors(
            [test_track_features], n_neighbors=k + 1
        )
        relevant_ids = train_df.iloc[indices[0][1:]][
            "track_id"
        ].tolist()  # Exclude self
        relevant_items[test_track_id] = relevant_ids

    # Calculate evaluation metrics
    precisions = []
    recalls = []
    ndcgs = []
    diversities = []
    novelties = []

    for track_id in recommended_items:
        if recommended_items[track_id] and track_id in relevant_items:
            prec = precision_at_k(
                recommended_items[track_id], relevant_items[track_id], k
            )
            rec = recall_at_k(recommended_items[track_id], relevant_items[track_id], k)
            ndcg_val = ndcg_at_k(
                recommended_items[track_id], relevant_items[track_id], k
            )
            div = diversity(recommended_items[track_id], item_features)
            nov = novelty(recommended_items[track_id], item_popularity)

            precisions.append(prec)
            recalls.append(rec)
            ndcgs.append(ndcg_val)
            diversities.append(div)
            novelties.append(nov)

    # Average metrics
    results = {
        "precision@k": np.mean(precisions) if precisions else 0,
        "recall@k": np.mean(recalls) if recalls else 0,
        "ndcg@k": np.mean(ndcgs) if ndcgs else 0,
        "diversity": np.mean(diversities) if diversities else 0,
        "novelty": np.mean(novelties) if novelties else 0,
    }

    print(f"Content-Based Evaluation Results: {results}")
    return results


def evaluate_popularity_based(train_df, test_df, k=10):
    """
    Evaluate popularity-based recommendations

    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training set DataFrame
    test_df : pandas.DataFrame
        Test set DataFrame
    k : int
        Number of recommendations to consider

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    print("Evaluating popularity-based recommendations...")

    # Initialize recommender
    recommender = PopularityRecommender(train_df)

    # Sample tracks from test set to evaluate
    sample_size = min(100, len(test_df))
    test_samples = test_df.sample(sample_size)

    # Create item features dictionary for diversity calculation
    item_features = {}
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
    for _, row in train_df.iterrows():
        item_features[row["track_id"]] = row[audio_features].values

    # Create item popularity dictionary for novelty calculation
    item_popularity = {}
    for _, row in train_df.iterrows():
        item_popularity[row["track_id"]] = row["popularity"]

    # Define relevant items for each test track (same genre, high popularity)
    relevant_items = {}
    for idx, test_row in test_samples.iterrows():
        # Find popular tracks with same genre
        same_genre = train_df[train_df["track_genre"] == test_row["track_genre"]]
        popular_tracks = same_genre.sort_values(by="popularity", ascending=False).head(
            k
        )

        # Get track IDs of popular tracks
        popular_track_ids = popular_tracks["track_id"].tolist()

        relevant_items[test_row["track_id"]] = popular_track_ids

    # Evaluate recommendations
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    diversity_scores = []
    novelty_scores = []

    for _, test_row in tqdm(
        test_samples.iterrows(), total=len(test_samples), desc="Evaluating tracks"
    ):
        # Get recommendations (for the genre of the test track)
        recommendations = recommender.recommend_popular(
            genre=test_row["track_genre"], n_recommendations=k
        )

        if recommendations is not None:
            # Get recommended track IDs
            recommended_ids = recommendations["track_id"].tolist()

            # Calculate metrics
            precision = precision_at_k(
                recommended_ids, relevant_items[test_row["track_id"]], k
            )
            recall = recall_at_k(
                recommended_ids, relevant_items[test_row["track_id"]], k
            )
            ndcg = ndcg_at_k(recommended_ids, relevant_items[test_row["track_id"]], k)
            div = diversity(recommended_ids, item_features)
            nov = novelty(recommended_ids, item_popularity)

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            diversity_scores.append(div)
            novelty_scores.append(nov)

    # Calculate average metrics
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
    avg_novelty = np.mean(novelty_scores) if novelty_scores else 0

    results = {
        "precision@k": avg_precision,
        "recall@k": avg_recall,
        "ndcg@k": avg_ndcg,
        "diversity": avg_diversity,
        "novelty": avg_novelty,
    }

    print(f"Popularity-Based Evaluation Results: {results}")

    return results


def evaluate_genre_based(train_df, test_df, k=10):
    """
    Evaluate genre-based recommendations

    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training set DataFrame
    test_df : pandas.DataFrame
        Test set DataFrame
    k : int
        Number of recommendations to consider

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    print("Evaluating genre-based recommendations...")

    # Initialize recommender
    recommender = GenreRecommender(train_df)
    recommender.fit()

    # Sample tracks from test set to evaluate
    sample_size = min(100, len(test_df))
    test_samples = test_df.sample(sample_size)

    # Create item features dictionary for diversity calculation
    item_features = {}
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
    for _, row in train_df.iterrows():
        item_features[row["track_id"]] = row[audio_features].values

    # Create item popularity dictionary for novelty calculation
    item_popularity = {}
    for _, row in train_df.iterrows():
        item_popularity[row["track_id"]] = row["popularity"]

    # Define relevant items for each test track (tracks from similar genres)
    relevant_items = {}
    for idx, test_row in test_samples.iterrows():
        # Get similar genres
        try:
            similar_genres = recommender.get_similar_genres(
                test_row["track_genre"], n=3
            )

            if similar_genres:
                # Include tracks from similar genres
                similar_genre_ids = [genre for genre, _ in similar_genres]
                similar_tracks = train_df[
                    train_df["track_genre"].isin(similar_genre_ids)
                ]

                # Get popular tracks from similar genres
                popular_tracks = similar_tracks.sort_values(
                    by="popularity", ascending=False
                ).head(k)

                # Get track IDs
                relevant_track_ids = popular_tracks["track_id"].tolist()

                relevant_items[test_row["track_id"]] = relevant_track_ids
            else:
                relevant_items[test_row["track_id"]] = []
        except:
            relevant_items[test_row["track_id"]] = []

    # Evaluate recommendations
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    diversity_scores = []
    novelty_scores = []

    for _, test_row in tqdm(
        test_samples.iterrows(), total=len(test_samples), desc="Evaluating tracks"
    ):
        try:
            # Get recommendations
            recommendations = recommender.recommend_by_genre(
                test_row["track_genre"], n_recommendations=k
            )

            if recommendations is not None:
                # Get recommended track IDs
                recommended_ids = recommendations["track_id"].tolist()

                # Calculate metrics
                relevant = relevant_items.get(test_row["track_id"], [])
                precision = precision_at_k(recommended_ids, relevant, k)
                recall = recall_at_k(recommended_ids, relevant, k)
                ndcg = ndcg_at_k(recommended_ids, relevant, k)
                div = diversity(recommended_ids, item_features)
                nov = novelty(recommended_ids, item_popularity)

                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
                diversity_scores.append(div)
                novelty_scores.append(nov)
        except:
            # Skip tracks with errors (e.g., genre not in training set)
            continue

    # Calculate average metrics
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
    avg_novelty = np.mean(novelty_scores) if novelty_scores else 0

    results = {
        "precision@k": avg_precision,
        "recall@k": avg_recall,
        "ndcg@k": avg_ndcg,
        "diversity": avg_diversity,
        "novelty": avg_novelty,
    }

    print(f"Genre-Based Evaluation Results: {results}")

    return results


def plot_evaluation_results(results):
    """
    Plot evaluation results for different recommendation approaches

    Parameters:
    -----------
    results : dict
        Dictionary of evaluation results for different approaches
    """
    # Create results directory
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Prepare data for plotting
    approaches = list(results.keys())
    metrics = list(results[approaches[0]].keys())

    # Create a figure with subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))

    # Plot each metric
    for i, metric in enumerate(metrics):
        # Extract values for this metric
        values = [results[approach][metric] for approach in approaches]

        # Create bar plot
        sns.barplot(x=approaches, y=values, ax=axes[i])

        # Customize plot
        axes[i].set_title(f"{metric} by Recommendation Approach")
        axes[i].set_xlabel("Approach")
        axes[i].set_ylabel(metric)

        # Add value labels
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f"{v:.3f}", ha="center")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    results_file = os.path.join(results_dir, "evaluation_results.png")
    plt.savefig(results_file)
    print(f"Saved evaluation results to {results_file}")

    # Show figure
    plt.show()


def evaluate_all_approaches(k=10):
    """
    Evaluate all recommendation approaches and compare results

    Parameters:
    -----------
    k : int
        Number of recommendations to consider
    """
    print("Loading data...")

    # Load train and test sets
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    train_df = pd.read_csv(os.path.join(data_dir, "train_set.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_set.csv"))

    # Load feature matrix with allow_pickle=True
    feature_matrix = np.load(
        os.path.join(data_dir, "feature_matrix.npy"), allow_pickle=True
    )

    # Evaluate each approach
    content_results = evaluate_content_based(train_df, test_df, feature_matrix, k=k)
    popularity_results = evaluate_popularity_based(train_df, test_df, k=k)
    genre_results = evaluate_genre_based(train_df, test_df, k=k)

    # Combine results
    all_results = {
        "Content-Based": content_results,
        "Popularity-Based": popularity_results,
        "Genre-Based": genre_results,
    }

    # Plot results
    plot_evaluation_results(all_results)

    # Determine best approach based on different metrics
    best_approach = {}
    for metric in content_results.keys():
        values = {
            "Content-Based": content_results[metric],
            "Popularity-Based": popularity_results[metric],
            "Genre-Based": genre_results[metric],
        }
        best_approach[metric] = max(values, key=values.get)

    print("\nBest Recommendation Approaches:")
    for metric, approach in best_approach.items():
        print(f"  {metric}: {approach}")

    # Save results to CSV
    results_df = pd.DataFrame(
        {
            "Metric": list(content_results.keys()),
            "Content-Based": list(content_results.values()),
            "Popularity-Based": list(popularity_results.values()),
            "Genre-Based": list(genre_results.values()),
            "Best Approach": [
                best_approach[metric] for metric in content_results.keys()
            ],
        }
    )

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "evaluation_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Saved evaluation results to {results_file}")

    return all_results, best_approach


if __name__ == "__main__":
    # Create results directory
    os.makedirs("../results", exist_ok=True)

    # Evaluate all approaches
    all_results, best_approach = evaluate_all_approaches(k=10)

    # Print best approach overall (based on NDCG, which balances relevance and ranking)
    print(
        f"\nOverall Best Recommendation Approach: {best_approach.get('ndcg@k', 'Unknown')}"
    )
