"""
Music Recommender Simulation — core recommendation logic.

Scoring formula:
    total = 0.40 * genre_score + 0.30 * mood_score + 0.20 * energy_score

    genre_score  : 1.0 if genre matches, else 0.0              (weight: 0.40)
    mood_score   : 1.0 if mood matches,  else 0.0              (weight: 0.30)
    energy_score : 1.0 - abs(song.energy - user.energy)        (weight: 0.20)
"""

import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Song:
    """
    Represents a single track and its audio features.

    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a listener's taste preferences.

    Required by tests/test_recommender.py

    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


class Recommender:
    """
    OOP wrapper around the recommendation logic.

    Required by tests/test_recommender.py

    Attributes:
        songs (List[Song]): The full catalogue of songs available to recommend.
    """

    def __init__(self, songs: List[Song]) -> None:
        """
        Initializes the Recommender with a catalogue of songs.

        Args:
            songs (List[Song]): List of Song objects loaded from the dataset.
        """
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Returns the top-k songs for a given user profile.

        Args:
            user (UserProfile): The listener's taste preferences.
            k (int): Number of recommendations to return. Defaults to 5.

        Returns:
            List[Song]: Up to k songs ranked by relevance to the user's profile.
        """
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Produces a human-readable explanation of why a song was recommended.

        Args:
            user (UserProfile): The listener's taste preferences.
            song (Song): The song being explained.

        Returns:
            str: A sentence describing which features matched the user's preferences.
        """
        # TODO: Implement explanation logic
        return "Explanation placeholder"


def load_songs(csv_path: str) -> List[Dict]:
    """
    Reads a CSV file and returns each row as a dictionary.

    Numeric fields (id, energy, tempo_bpm, valence, danceability, acousticness)
    are automatically converted to float so they are ready for calculations.
    All other fields (title, artist, genre, mood) are kept as strings.

    Required by src/main.py

    Args:
        csv_path (str): Relative or absolute path to the songs CSV file.

    Returns:
        List[Dict]: A list of song dictionaries, one per row in the CSV.
    """
    numeric_fields = {"id", "energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    songs = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(f"Loading songs from {csv_path}...")
        for row in reader:
            song = {
                key: float(value) if key in numeric_fields else value
                for key, value in row.items()
            }
            songs.append(song)

    return songs


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int) -> List[Tuple[Dict, float, str]]:
    """
    Scores every song against the user's preferences and returns the top-k results.

    Each song is evaluated on three features using a weighted formula:

        total = 0.40 * genre_score + 0.30 * mood_score + 0.20 * energy_score

    genre_score and mood_score are binary (1.0 on match, 0.0 otherwise).
    energy_score is a proximity value: 1.0 - abs(song.energy - user.energy).

    Required by src/main.py

    Args:
        user_prefs (Dict): User preferences with keys 'genre' (str),
                           'mood' (str), and 'energy' (float).
        songs (List[Dict]): Full list of song dictionaries from load_songs().
        k (int): Number of top recommendations to return.

    Returns:
        List[Tuple[Dict, float, str]]: Each entry is a tuple of:
            - song (Dict): The song dictionary.
            - score (float): Weighted total score between 0.0 and 1.0.
            - explanation (str): Pipe-separated reasons describing matched features.
    """
    scored = []

    for song in songs:
        reasons = []

        # Genre score — binary match
        if song["genre"] == user_prefs["genre"]:
            genre_score = 1.0
            reasons.append("Genre match (+0.20)")
        else:
            genre_score = 0.0

        # Mood score — binary match
        if song["mood"] == user_prefs["mood"]:
            mood_score = 1.0
            reasons.append("Mood match (+0.30)")
        else:
            mood_score = 0.0

        # Energy score — proximity, always contributes
        energy_score = 1.0 - abs(song["energy"] - user_prefs["energy"])
        reasons.append(f"Energy proximity (+{0.40 * energy_score:.2f})")

        # Weighted total
        total = 0.20 * genre_score + 0.30 * mood_score + 0.40 * energy_score

        explanation = " | ".join(reasons)
        scored.append((song, total, explanation))

    # Sort by score descending, return top k songs
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
