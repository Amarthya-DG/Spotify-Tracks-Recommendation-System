#!/usr/bin/env python
# coding: utf-8

# # Spotify Tracks Exploratory Data Analysis
#
# This script explores the Spotify tracks dataset from Hugging Face to gain insights for building a recommendation system.

# Import necessary libraries
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create directories for saving results
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

print(f"Figures will be saved to: {figures_dir}")

# Set visual style
plt.style.use("ggplot")
sns.set(style="whitegrid")

print("Loading dataset from Hugging Face...")
ds = load_dataset("maharshipandya/spotify-tracks-dataset")
print("Dataset loaded successfully!")

# Convert to pandas DataFrame
df = pd.DataFrame(ds["train"])

# Display basic information
print(f"Dataset shape: {df.shape}")
print(df.head())

# Data types and missing values
print("\nData Types:")
print(df.dtypes)

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(
    missing_values[missing_values > 0]
    if sum(missing_values) > 0
    else "No missing values found"
)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# ## Analyzing Genre Distribution

# Count tracks by genre
genre_counts = df["track_genre"].value_counts()

# Display top genres
print(f"\nNumber of unique genres: {len(genre_counts)}")
print("\nTop 20 genres by number of tracks:")
print(genre_counts.head(20))

# Plot top 20 genres
plt.figure(figsize=(12, 8))
sns.barplot(x=genre_counts.head(20).values, y=genre_counts.head(20).index)
plt.title("Top 20 Genres by Number of Tracks")
plt.xlabel("Number of Tracks")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "top_20_genres.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'top_20_genres.png')}")

# ## Analyzing Popularity Distribution

# Plot popularity distribution
plt.figure(figsize=(12, 6))
sns.histplot(df["popularity"], bins=50, kde=True)
plt.title("Distribution of Track Popularity")
plt.xlabel("Popularity")
plt.ylabel("Number of Tracks")
plt.axvline(
    df["popularity"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {df['popularity'].mean():.2f}",
)
plt.axvline(
    df["popularity"].median(),
    color="green",
    linestyle="--",
    label=f"Median: {df['popularity'].median():.2f}",
)
plt.legend()
plt.savefig(os.path.join(figures_dir, "popularity_distribution.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'popularity_distribution.png')}")

# Average popularity by genre
genre_popularity = (
    df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False)
)

# Display most and least popular genres
print("\nTop 10 Most Popular Genres:")
print(genre_popularity.head(10))

print("\nLeast 10 Popular Genres:")
print(genre_popularity.tail(10))

# Plot average popularity for top 20 genres
plt.figure(figsize=(12, 8))
sns.barplot(x=genre_popularity.head(20).values, y=genre_popularity.head(20).index)
plt.title("Top 20 Genres by Average Popularity")
plt.xlabel("Average Popularity")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "genre_popularity.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'genre_popularity.png')}")

# ## Analyzing Audio Features

# Select audio features for analysis
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

# Display distribution of each audio feature
plt.figure(figsize=(15, 12))
for i, feature in enumerate(audio_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "audio_features_distribution.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'audio_features_distribution.png')}")

# Correlation matrix between audio features
correlation = df[audio_features].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Audio Features")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'correlation_heatmap.png')}")

# Correlation of features with popularity
popularity_correlation = (
    df[audio_features + ["popularity"]]
    .corr()["popularity"]
    .drop("popularity")
    .sort_values(ascending=False)
)
print("\nCorrelation of features with popularity:")
print(popularity_correlation)

# Plot correlation of features with popularity
plt.figure(figsize=(12, 6))
sns.barplot(x=popularity_correlation.values, y=popularity_correlation.index)
plt.title("Correlation of Audio Features with Popularity")
plt.xlabel("Correlation Coefficient")
plt.axvline(0, color="black", linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "popularity_correlation.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'popularity_correlation.png')}")

# ## Genre Characteristics

# Select top 10 genres by number of tracks
top_genres = genre_counts.head(10).index.tolist()

# Create a dataframe with only these genres
top_genres_df = df[df["track_genre"].isin(top_genres)]

# Compare danceability across top genres
plt.figure(figsize=(12, 6))
sns.boxplot(x="track_genre", y="danceability", data=top_genres_df)
plt.title("Danceability Distribution Across Top Genres")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "genre_danceability.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'genre_danceability.png')}")

# Compare energy across top genres
plt.figure(figsize=(12, 6))
sns.boxplot(x="track_genre", y="energy", data=top_genres_df)
plt.title("Energy Distribution Across Top Genres")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "genre_energy.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'genre_energy.png')}")

# ## Dimensionality Reduction for Visualization

# Perform PCA on audio features
X = df[audio_features].values
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
pca_df["genre"] = df["track_genre"].values
pca_df["popularity"] = df["popularity"].values

# Display explained variance
print(f"\nPCA Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")

# Plot PCA for top genres (take a sample for better visibility)
sample_size = min(200 * len(top_genres), len(pca_df[pca_df["genre"].isin(top_genres)]))
pca_top_genres = pca_df[pca_df["genre"].isin(top_genres)].sample(sample_size)

plt.figure(figsize=(12, 10))
sns.scatterplot(x="PC1", y="PC2", hue="genre", data=pca_top_genres, alpha=0.7)
plt.title("PCA of Audio Features Colored by Genre")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2f})")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2f})")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_visualization.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'pca_visualization.png')}")

# ## Top Tracks Analysis

# Top 20 most popular tracks
top_tracks = df.sort_values(by="popularity", ascending=False).head(20)[
    ["track_name", "artists", "album_name", "track_genre", "popularity"]
]

print("\nTop 20 most popular tracks:")
print(top_tracks)

# Audio features of top tracks
top_tracks_features = df.sort_values(by="popularity", ascending=False).head(20)

# Compare with average features
avg_features = df[audio_features].mean()
top_avg_features = top_tracks_features[audio_features].mean()

comparison = pd.DataFrame(
    {
        "Top Tracks Average": top_avg_features,
        "Overall Average": avg_features,
        "Difference": top_avg_features - avg_features,
    }
).sort_values(by="Difference", ascending=False)

print("\nFeature comparison between top tracks and average tracks:")
print(comparison)

# Plot the differences
plt.figure(figsize=(12, 6))
sns.barplot(x=comparison["Difference"].values, y=comparison.index)
plt.title("How Top Tracks Differ From Average Tracks")
plt.xlabel("Difference from Average")
plt.axvline(0, color="black", linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "top_tracks_comparison.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'top_tracks_comparison.png')}")

# ## Duration Analysis

# Convert duration_ms to minutes for better interpretability
df["duration_min"] = df["duration_ms"] / 60000

# Plot duration distribution
plt.figure(figsize=(12, 6))
sns.histplot(df["duration_min"], bins=100, kde=True)
plt.title("Distribution of Track Duration")
plt.xlabel("Duration (minutes)")
plt.ylabel("Number of Tracks")
plt.axvline(
    df["duration_min"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {df['duration_min'].mean():.2f} min",
)
plt.axvline(
    df["duration_min"].median(),
    color="green",
    linestyle="--",
    label=f"Median: {df['duration_min'].median():.2f} min",
)
plt.legend()
plt.xlim(0, 10)  # Limit to 10 minutes for better visualization
plt.savefig(os.path.join(figures_dir, "duration_distribution.png"))
print(f"Saved figure: {os.path.join(figures_dir, 'duration_distribution.png')}")

# Average duration by genre
genre_duration = (
    df.groupby("track_genre")["duration_min"].mean().sort_values(ascending=False)
)

# Display genres with longest and shortest tracks
print("\nTop 10 Genres with Longest Tracks:")
print(genre_duration.head(10))

print("\nTop 10 Genres with Shortest Tracks:")
print(genre_duration.tail(10))

# ## Recommendation Approach Analysis

print("\n## Recommendation Approach Analysis")
print("\nBased on EDA, the following recommendation approaches are suggested:")

print("\n1. Content-Based Filtering:")
print("   - Strengths: Rich audio features that distinguish tracks well")
print("   - PCA shows audio features can separate genres")
print("   - Good for finding tracks with similar audio characteristics")
print(
    "   - Limitations: May recommend tracks from different genres, doesn't account for popularity"
)

print("\n2. Popularity-Based Recommendations:")
print(
    "   - Strengths: Clear popularity distribution with significant differences between tracks"
)
print("   - Different genres have different popularity levels")
print("   - Simple and effective for recommending generally liked tracks")
print("   - Limitations: Doesn't consider personal preferences or track similarity")

print("\n3. Genre-Based Recommendations:")
print("   - Strengths: Dataset has 125 distinct genres, providing good categorization")
print("   - Genres have distinct audio feature profiles")
print("   - Good for introducing users to new genres similar to ones they already like")
print(
    "   - Limitations: Some genres might have too few tracks, doesn't account for cross-genre similarities"
)

print("\n4. Hybrid Approach (Recommended):")
print("   - Combine content-based and popularity-based approaches for the best results")
print("   - Use content-based filtering to find tracks with similar audio features")
print("   - Use popularity as a tie-breaker or weighting factor")
print("   - Use genre information to ensure diversity in recommendations")

print("\nEDA completed and results saved to '../figures/' directory.")
