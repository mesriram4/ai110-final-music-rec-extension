"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("songs.csv") 

    # Starter example profile
    """
    Dictionary of specific user preference terms for genre: 
    + Post-hardcore rock 
    + Post-punk rock 
    + Shoegaze Rock
    + Modal Jazz
    + Bebop Jazz
    + Chill Synthwave 
    + Lo-fi House 
    + Dream Pop
    """
    user_prefs_1 = {
        "genre": "lofi",
        "mood":  "chill",
        "energy": 0.61
    }

    # Adversarial profile: Shoegaze listener
    # "shoegaze" is not in the catalog → genre_score is always 0.0.
    # Only one song has mood "dreamy" (Neon Daydream, indie pop),
    # so the recommender must lean almost entirely on energy proximity.
    # Exposes: sub-genre specificity is lost in binary genre matching.
    user_prefs_2 = {
        "genre": "shoegaze",
        "mood":  "dreamy",
        "energy": 0.52
    }

    # Adversarial profile: Bebop jazz listener
    # "bebop jazz" is not in the catalog → genre_score is always 0.0.
    # Bebop is high-energy and fast, but the only "focused" song in the
    # catalog is a lo-fi track at low energy (0.40), creating a direct
    # contradiction between mood and energy that stresses the scoring weights.
    # Exposes: genre + mood combo that has no plausible catalog match.
    user_prefs_3 = {
        "genre": "bebop jazz",
        "mood":  "focused",
        "energy": 0.88
    }


    all_profiles = [
        ("Lo-fi listener (baseline)",   user_prefs_1),
        ("Shoegaze listener (adversarial)", user_prefs_2),
        ("Bebop jazz listener (adversarial)", user_prefs_3),
    ]

    print("\nLoading songs...:\n")
    for label, user_prefs in all_profiles:
        print(f"=== {label} ===")
        recommendations = recommend_songs(user_prefs, songs, k=5)
        for rec in recommendations:
            # You decide the structure of each returned item.
            # A common pattern is: (song, score, explanation)
            song, score, explanation = rec
            print(f"{song['title']} - Score: {score:.2f}")
            print(f"Because: {explanation}")
            print()


if __name__ == "__main__":
    response = input("Currently running music recommender. Would you like to switch to AI Mode?: ")

    if response.lower() in ["yes", "y"]:
        print("Switching to AI Mode...")
        import Music_RAG as rag
        prompt = input("Enter Prompt Here: ")
        rag.main(prompt)
    else: 
        print("Remaining in original mode...")
        main()
