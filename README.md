# Spotify Tracks Recommendation System

This project implements a comprehensive recommendation system for Spotify tracks based on a dataset of 114k tracks with audio features. The system provides multiple recommendation approaches and interfaces.
<img width="895" alt="image" src="https://github.com/user-attachments/assets/9b206192-01b8-44f5-8301-c496fefe0466" />
<img width="866" alt="image" src="https://github.com/user-attachments/assets/55755b68-a388-4883-b625-921a9a500cd1" />



## Dataset Description

The dataset contains Spotify tracks across 125 different genres with various audio features:

- Basic information: track_id, artists, album_name, track_name, popularity, duration_ms, explicit
- Audio features: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature
- Genre information: track_genre

## Project Structure

- `data/`: Contains the processed dataset
- `src/`: Source code for data processing and recommendation algorithms
- `models/`: Saved recommendation models
- `notebooks/`: Scripts for exploration and visualization
- `figures/`: Visualizations generated from the EDA
- `results/`: Evaluation results for different recommendation approaches

## Recommendation Approaches

The system implements multiple recommendation strategies:
1. **Content-based filtering**: Recommends tracks with similar audio features
2. **Popularity-based recommendations**: Recommends popular tracks within specific genres
3. **Genre-based recommendations**: Recommends tracks from similar genres

## Evaluation Metrics

The recommendation approaches are evaluated using several metrics:

- **Precision@k**: Measures the proportion of recommended items that are relevant
- **Recall@k**: Measures the proportion of relevant items that are recommended
- **NDCG@k**: Normalized Discounted Cumulative Gain, measures the ranking quality
- **Diversity**: Measures how different the recommended items are from each other
- **Novelty**: Measures how "non-obvious" the recommendations are (based on popularity)

## System Architecture

The recommendation system is built with a modular design consisting of:

1. **Data Processing Layer**: Handles loading, cleaning, and feature extraction from the raw dataset
2. **Recommendation Models**: Implements multiple recommendation strategies
3. **Interface Layer**: Provides CLI, interactive terminal, and web interfaces

## Recommendation Approaches

The system implements three main recommendation strategies:

### 1. Content-Based Recommendations

Recommendations are made based on track audio features such as danceability, energy, loudness, etc. The system uses:
- Feature engineering to extract meaningful audio features
- Nearest neighbors algorithm to find similar tracks
- Cosine similarity to measure track similarity

This approach is ideal for finding tracks that sound similar to a given track.

### 2. Popularity-Based Recommendations

Recommendations are made based on track popularity, optionally filtered by genre. This approach:
- Recommends most popular tracks overall or within a genre
- Can be combined with other approaches for hybrid recommendations

This approach is best for discovering generally popular tracks.

### 3. Genre-Based Recommendations

Recommendations are based on genre similarity, calculated from average audio features of tracks in each genre. This approach:
- Computes similarity between genres based on their audio profiles
- Recommends tracks from similar genres
- Introduces diversity into recommendations

### 4. Personalized Recommendations

For users who "like" multiple tracks, the system:
- Aggregates recommendations based on each liked track
- Ranks them by similarity score
- Removes duplicates and already liked tracks

## User Interfaces

### 1. Command-Line Interface (CLI)

The CLI (`src/cli.py`) provides a structured command-based interface with subcommands:
- `search`: Search for tracks
- `info`: Get track information
- `track`: Get recommendations based on a track
- `popular`: Get popular tracks
- `genre`: Get recommendations for a genre
- `genres`: List all available genres
- `profile`: Get personalized recommendations
- `interactive`: Start interactive recommendation session
- `process`: Process the dataset

### 2. Interactive Terminal Interface

The interactive mode (`src/recommend.py`) provides a text-based conversational interface:
- Search for tracks
- View track details
- Get recommendations
- Build a profile of liked tracks
- Get personalized recommendations

### 3. Web Interface

The web app (`src/web_app.py`) provides a modern, responsive user interface:
- Search functionality with visual results
- Detailed track information display
- Multiple recommendation options
- Like/unlike track functionality
- Personalized recommendations based on liked tracks

## Setup and Usage

1. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Process Data**:
   ```
   python src/data_processing.py
   ```
   or
   ```
   python src/cli.py process
   ```

3. **Using the CLI**:
   ```
   python src/cli.py <command> [options]
   ```

4. **Using the Interactive Terminal**:
   ```
   python src/recommend.py
   ```
   or
   ```
   python src/cli.py interactive
   ```

5. **Using the Web Interface**:
   ```
   python src/web_app.py
   ```

## Exploratory Data Analysis

The project includes a comprehensive notebook (`notebooks/Spotify_Tracks_EDA.py`) for exploring:
- Genre distribution
- Popularity patterns
- Audio feature distributions and correlations
- Key and mode analysis
- Explicit content analysis

## Future Enhancements

1. **Collaborative Filtering**: Implement user-user or item-item collaborative filtering based on listening history
2. **Deep Learning Models**: Incorporate deep learning for feature extraction from audio
3. **API Integration**: Add Spotify API integration to fetch current popularity data and audio samples
4. **Advanced Hybrid Models**: Combine multiple recommendation approaches with weighted algorithms
5. **Artist/Album Recommendations**: Expand to recommend artists and albums in addition to tracks 
