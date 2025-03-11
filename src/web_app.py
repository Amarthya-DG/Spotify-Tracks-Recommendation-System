import os

from flask import Flask, jsonify, render_template, request

from recommend import SpotifyRecommendationSystem

app = Flask(__name__)

# Initialize the recommendation system
try:
    data_dir = "../data"
    if not os.path.exists(f"{data_dir}/cleaned_tracks.csv"):
        print(f"Warning: Processed data not found in {data_dir}.")
        print("Please run data_processing.py first to prepare the dataset.")
        recommendation_system = None
    else:
        recommendation_system = SpotifyRecommendationSystem()
except Exception as e:
    print(f"Error initializing recommendation system: {e}")
    recommendation_system = None


@app.route("/")
def index():
    """Render the home page"""
    return render_template("index.html")


@app.route("/search")
def search():
    """Search for tracks by name or artist"""
    query = request.args.get("query", "")
    limit = int(request.args.get("limit", 5))

    if not query or not recommendation_system:
        return jsonify([])

    results = recommendation_system.search_tracks(query, limit=limit)
    if results is None:
        return jsonify([])

    # Convert to list of dictionaries for JSON
    return jsonify(results.to_dict(orient="records"))


@app.route("/track/<track_id>")
def track_info(track_id):
    """Get information about a specific track"""
    if not recommendation_system:
        return jsonify({"error": "Recommendation system not initialized"})

    track = recommendation_system.get_track_info(track_id=track_id)
    if track is None:
        return jsonify({"error": "Track not found"})

    # Convert to dictionary for JSON
    return jsonify(track.to_dict())


@app.route("/recommend/track/<track_id>")
def recommend_by_track(track_id):
    """Get recommendations based on a specific track"""
    if not recommendation_system:
        return jsonify({"error": "Recommendation system not initialized"})

    count = int(request.args.get("count", 5))

    recommendations = recommendation_system.recommend_by_track(
        track_id, n_recommendations=count
    )

    if recommendations is None:
        return jsonify([])

    # Convert to list of dictionaries for JSON
    return jsonify(recommendations.to_dict(orient="records"))


@app.route("/recommend/popular")
def recommend_popular():
    """Get popular track recommendations"""
    if not recommendation_system:
        return jsonify({"error": "Recommendation system not initialized"})

    genre = request.args.get("genre", None)
    count = int(request.args.get("count", 10))

    recommendations = recommendation_system.recommend_popular(
        genre=genre, n_recommendations=count
    )

    if recommendations is None:
        return jsonify([])

    # Convert to list of dictionaries for JSON
    return jsonify(recommendations.to_dict(orient="records"))


@app.route("/recommend/genre/<genre>")
def recommend_by_genre(genre):
    """Get recommendations for a specific genre"""
    if not recommendation_system:
        return jsonify({"error": "Recommendation system not initialized"})

    count = int(request.args.get("count", 10))

    recommendations = recommendation_system.recommend_by_genre(
        genre, n_recommendations=count
    )

    if recommendations is None:
        return jsonify([])

    # Convert to list of dictionaries for JSON
    return jsonify(recommendations.to_dict(orient="records"))


@app.route("/genres")
def get_genres():
    """Get a list of all available genres"""
    if not recommendation_system:
        return jsonify({"error": "Recommendation system not initialized"})

    genres = recommendation_system.get_available_genres()
    return jsonify(genres)


@app.route("/recommend/profile", methods=["POST"])
def recommend_for_profile():
    """Get personalized recommendations based on a list of track IDs"""
    if not recommendation_system:
        return jsonify({"error": "Recommendation system not initialized"})

    data = request.get_json()

    if not data or "track_ids" not in data:
        return jsonify({"error": "No track IDs provided"})

    track_ids = data["track_ids"]
    count = data.get("count", 10)

    recommendations = recommendation_system.recommend_for_user_profile(
        track_ids, n_recommendations=count
    )

    if recommendations is None:
        return jsonify([])

    # Convert to list of dictionaries for JSON
    return jsonify(recommendations.to_dict(orient="records"))


def create_templates():
    """Create the templates directory and HTML files"""
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)

    # Create index.html
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spotify Track Recommender</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            background-color: #f8f9fa;
        }
        .header {
            background-color: #1DB954;
            color: white;
            padding: 20px 0;
            margin-bottom: 30px;
            border-radius: 5px;
        }
        .card {
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .track-card {
            transition: transform 0.3s;
        }
        .track-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }
        .track-img {
            width: 100%;
            height: auto;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
        .btn-spotify {
            background-color: #1DB954;
            color: white;
        }
        .btn-spotify:hover {
            background-color: #1ED760;
            color: white;
        }
        .search-section, .recommendations-section, .track-info-section, .liked-section {
            margin-bottom: 40px;
        }
        #liked-tracks {
            min-height: 100px;
        }
        .liked-track {
            background-color: #f1f8e9;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1>Spotify Track Recommender</h1>
            <p class="lead">Discover new music based on your preferences</p>
        </div>

        <div class="search-section">
            <div class="card">
                <div class="card-body">
                    <h2 class="card-title">Search Tracks</h2>
                    <div class="input-group mb-3">
                        <input type="text" id="search-input" class="form-control" placeholder="Search for tracks or artists...">
                        <button class="btn btn-spotify" id="search-button">Search</button>
                    </div>
                </div>
            </div>
            <div id="search-results" class="row"></div>
        </div>

        <div class="track-info-section" style="display: none;" id="track-info-container">
            <div class="card">
                <div class="card-body">
                    <h2 class="card-title">Track Information</h2>
                    <div id="track-info"></div>
                    <button class="btn btn-spotify" id="recommend-button">Get Recommendations</button>
                    <button class="btn btn-outline-success" id="like-button">Add to Liked Tracks</button>
                </div>
            </div>
        </div>

        <div class="recommendations-section" style="display: none;" id="recommendations-container">
            <div class="card">
                <div class="card-body">
                    <h2 class="card-title">Recommendations</h2>
                    <div class="d-flex mb-3">
                        <button class="btn btn-outline-success me-2" id="popular-button">Popular Tracks</button>
                        <select class="form-select me-2" id="genre-select" style="max-width: 200px;">
                            <option value="">Select Genre...</option>
                        </select>
                        <button class="btn btn-outline-success" id="genre-button">By Genre</button>
                    </div>
                    <div id="recommendations-results" class="row"></div>
                </div>
            </div>
        </div>

        <div class="liked-section">
            <div class="card">
                <div class="card-body">
                    <h2 class="card-title">Your Liked Tracks</h2>
                    <div id="liked-tracks"></div>
                    <button class="btn btn-spotify mt-3" id="personal-recommendations-button" disabled>Get Personalized Recommendations</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // State management
        const state = {
            currentTrack: null,
            likedTracks: [],
            genres: []
        };

        // DOM Elements
        const searchInput = document.getElementById('search-input');
        const searchButton = document.getElementById('search-button');
        const searchResults = document.getElementById('search-results');
        const trackInfoContainer = document.getElementById('track-info-container');
        const trackInfo = document.getElementById('track-info');
        const recommendButton = document.getElementById('recommend-button');
        const likeButton = document.getElementById('like-button');
        const recommendationsContainer = document.getElementById('recommendations-container');
        const recommendationsResults = document.getElementById('recommendations-results');
        const popularButton = document.getElementById('popular-button');
        const genreSelect = document.getElementById('genre-select');
        const genreButton = document.getElementById('genre-button');
        const likedTracks = document.getElementById('liked-tracks');
        const personalRecommendationsButton = document.getElementById('personal-recommendations-button');

        // Event Listeners
        document.addEventListener('DOMContentLoaded', init);
        searchButton.addEventListener('click', searchTracks);
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchTracks();
        });
        recommendButton.addEventListener('click', getTrackRecommendations);
        likeButton.addEventListener('click', addToLikedTracks);
        popularButton.addEventListener('click', getPopularTracks);
        genreButton.addEventListener('click', getGenreRecommendations);
        personalRecommendationsButton.addEventListener('click', getPersonalRecommendations);

        // Functions
        async function init() {
            await loadGenres();
            updateLikedTracksUI();
        }

        async function loadGenres() {
            try {
                const response = await fetch('/genres');
                state.genres = await response.json();
                
                // Populate genre select
                genreSelect.innerHTML = '<option value="">Select Genre...</option>';
                state.genres.forEach(genre => {
                    const option = document.createElement('option');
                    option.value = genre;
                    option.textContent = genre;
                    genreSelect.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading genres:', error);
            }
        }

        async function searchTracks() {
            const query = searchInput.value.trim();
            if (!query) return;

            try {
                const response = await fetch(`/search?query=${encodeURIComponent(query)}&limit=9`);
                const data = await response.json();
                
                displaySearchResults(data);
            } catch (error) {
                console.error('Error searching tracks:', error);
            }
        }

        function displaySearchResults(tracks) {
            searchResults.innerHTML = '';
            
            if (tracks.length === 0) {
                searchResults.innerHTML = '<div class="col-12 text-center">No tracks found</div>';
                return;
            }

            tracks.forEach(track => {
                const trackCard = createTrackCard(track);
                searchResults.appendChild(trackCard);
            });
        }

        function createTrackCard(track) {
            const col = document.createElement('div');
            col.className = 'col-md-4 mb-4';
            
            col.innerHTML = `
                <div class="card track-card">
                    <img src="https://via.placeholder.com/300x300.png?text=${encodeURIComponent(track.track_name)}" class="track-img" alt="${track.track_name}">
                    <div class="card-body">
                        <h5 class="card-title">${track.track_name}</h5>
                        <p class="card-text">${track.artists}</p>
                        <p><small>Genre: ${track.track_genre}</small></p>
                        <p><small>Popularity: ${track.popularity}/100</small></p>
                        <button class="btn btn-spotify view-track" data-track-id="${track.track_id}">View Details</button>
                    </div>
                </div>
            `;
            
            col.querySelector('.view-track').addEventListener('click', () => {
                getTrackDetails(track.track_id);
            });
            
            return col;
        }

        async function getTrackDetails(trackId) {
            try {
                const response = await fetch(`/track/${trackId}`);
                const track = await response.json();
                
                state.currentTrack = track;
                displayTrackInfo(track);
                
                // Check if this track is already liked
                const isLiked = state.likedTracks.some(t => t.track_id === track.track_id);
                likeButton.textContent = isLiked ? 'Remove from Liked Tracks' : 'Add to Liked Tracks';
                
                // Show track info and recommendations sections
                trackInfoContainer.style.display = 'block';
                recommendationsContainer.style.display = 'block';
                window.scrollTo({
                    top: trackInfoContainer.offsetTop,
                    behavior: 'smooth'
                });
            } catch (error) {
                console.error('Error getting track details:', error);
            }
        }

        function displayTrackInfo(track) {
            const audioFeatures = [
                'danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 
                'valence', 'tempo'
            ];
            
            let featuresHTML = '';
            audioFeatures.forEach(feature => {
                if (track[feature] !== undefined) {
                    featuresHTML += `<span class="badge bg-light text-dark me-2">${feature}: ${track[feature]}</span>`;
                }
            });
            
            trackInfo.innerHTML = `
                <div class="row">
                    <div class="col-md-4">
                        <img src="https://via.placeholder.com/300x300.png?text=${encodeURIComponent(track.track_name)}" class="img-fluid rounded" alt="${track.track_name}">
                    </div>
                    <div class="col-md-8">
                        <h3>${track.track_name}</h3>
                        <p class="lead">by ${track.artists}</p>
                        <p><strong>Album:</strong> ${track.album_name}</p>
                        <p><strong>Genre:</strong> ${track.track_genre}</p>
                        <p><strong>Popularity:</strong> ${track.popularity}/100</p>
                        <p><strong>Track ID:</strong> ${track.track_id}</p>
                        <div class="mt-3">
                            <h5>Audio Features:</h5>
                            <div>${featuresHTML}</div>
                        </div>
                    </div>
                </div>
            `;
        }

        async function getTrackRecommendations() {
            if (!state.currentTrack) return;
            
            try {
                const response = await fetch(`/recommend/track/${state.currentTrack.track_id}?count=6`);
                const recommendations = await response.json();
                
                displayRecommendations(recommendations, `Based on "${state.currentTrack.track_name}"`);
            } catch (error) {
                console.error('Error getting recommendations:', error);
            }
        }

        async function getPopularTracks() {
            try {
                const response = await fetch('/recommend/popular?count=6');
                const recommendations = await response.json();
                
                displayRecommendations(recommendations, 'Popular Tracks');
            } catch (error) {
                console.error('Error getting popular tracks:', error);
            }
        }

        async function getGenreRecommendations() {
            const genre = genreSelect.value;
            if (!genre) return;
            
            try {
                const response = await fetch(`/recommend/genre/${encodeURIComponent(genre)}?count=6`);
                const recommendations = await response.json();
                
                displayRecommendations(recommendations, `${genre} Recommendations`);
            } catch (error) {
                console.error('Error getting genre recommendations:', error);
            }
        }

        function displayRecommendations(tracks, title) {
            recommendationsResults.innerHTML = '';
            
            if (tracks.length === 0) {
                recommendationsResults.innerHTML = '<div class="col-12 text-center">No recommendations found</div>';
                return;
            }
            
            // Add title
            const titleEl = document.createElement('div');
            titleEl.className = 'col-12 mb-3';
            titleEl.innerHTML = `<h4>${title}</h4>`;
            recommendationsResults.appendChild(titleEl);
            
            // Add track cards
            tracks.forEach(track => {
                const trackCard = createTrackCard(track);
                recommendationsResults.appendChild(trackCard);
            });
            
            // Scroll to recommendations
            window.scrollTo({
                top: recommendationsContainer.offsetTop,
                behavior: 'smooth'
            });
        }

        function addToLikedTracks() {
            if (!state.currentTrack) return;
            
            const trackId = state.currentTrack.track_id;
            const trackIndex = state.likedTracks.findIndex(track => track.track_id === trackId);
            
            if (trackIndex === -1) {
                // Add track to liked tracks
                state.likedTracks.push(state.currentTrack);
                likeButton.textContent = 'Remove from Liked Tracks';
            } else {
                // Remove track from liked tracks
                state.likedTracks.splice(trackIndex, 1);
                likeButton.textContent = 'Add to Liked Tracks';
            }
            
            updateLikedTracksUI();
        }

        function updateLikedTracksUI() {
            likedTracks.innerHTML = '';
            
            if (state.likedTracks.length === 0) {
                likedTracks.innerHTML = '<div class="text-center">No liked tracks yet</div>';
                personalRecommendationsButton.disabled = true;
                return;
            }
            
            state.likedTracks.forEach(track => {
                const likedTrack = document.createElement('div');
                likedTrack.className = 'liked-track';
                likedTrack.innerHTML = `
                    <div>
                        <strong>${track.track_name}</strong> by ${track.artists}
                    </div>
                    <button class="btn btn-sm btn-outline-danger remove-liked" data-track-id="${track.track_id}">Remove</button>
                `;
                
                likedTrack.querySelector('.remove-liked').addEventListener('click', () => {
                    const trackIndex = state.likedTracks.findIndex(t => t.track_id === track.track_id);
                    if (trackIndex !== -1) {
                        state.likedTracks.splice(trackIndex, 1);
                        updateLikedTracksUI();
                        
                        // Update like button if this is the current track
                        if (state.currentTrack && state.currentTrack.track_id === track.track_id) {
                            likeButton.textContent = 'Add to Liked Tracks';
                        }
                    }
                });
                
                likedTracks.appendChild(likedTrack);
            });
            
            personalRecommendationsButton.disabled = state.likedTracks.length === 0;
        }

        async function getPersonalRecommendations() {
            if (state.likedTracks.length === 0) return;
            
            try {
                const trackIds = state.likedTracks.map(track => track.track_id);
                
                const response = await fetch('/recommend/profile', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        track_ids: trackIds,
                        count: 6
                    }),
                });
                
                const recommendations = await response.json();
                displayRecommendations(recommendations, 'Personalized Recommendations');
                
                // Make sure recommendations section is visible
                recommendationsContainer.style.display = 'block';
            } catch (error) {
                console.error('Error getting personalized recommendations:', error);
            }
        }
    </script>
</body>
</html>
    """

    with open(os.path.join(templates_dir, "index.html"), "w") as f:
        f.write(index_html)


if __name__ == "__main__":
    # Create templates directory and files
    create_templates()

    # Run the Flask app
    print("Starting web server at http://127.0.0.1:5000")
    app.run(debug=True)
