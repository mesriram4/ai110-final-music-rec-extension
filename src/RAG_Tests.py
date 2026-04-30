#RAG Tests
import Music_RAG as rag
import chromadb
from unittest.mock import patch, MagicMock

"""
NOTE: ALL TEST CASES ARE WRITTEN BY CLAUDE!!!
"""
# ──────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────

MOCK_SONGS = [
    {"title": "Midnight Coding", "artist": "LoRoom",        "genre": "lofi", "mood": "chill",   "energy": 0.42},
    {"title": "Storm Runner",    "artist": "Voltline",       "genre": "rock", "mood": "intense", "energy": 0.91},
    {"title": "Sunrise City",    "artist": "Neon Echo",      "genre": "pop",  "mood": "happy",   "energy": 0.82},
    {"title": "Library Rain",    "artist": "Paper Lanterns", "genre": "lofi", "mood": "chill",   "energy": 0.35},
    {"title": "Neon Daydream",   "artist": "Soft Static",    "genre": "indie","mood": "dreamy",  "energy": 0.52},
]

# DefaultEmbeddingFunction (all-MiniLM-L6-v2 via ONNX) produces 384-dimensional vectors
MOCK_VECTOR = [0.1] * 384


def make_mock_ef():
    """Returns a mock DefaultEmbeddingFunction instance whose __call__ returns MOCK_VECTOR per input."""
    mock_ef = MagicMock()
    mock_ef.side_effect = lambda texts: [MOCK_VECTOR for _ in texts]
    return mock_ef


def make_mock_generate_response():
    message = MagicMock()
    message.content = "Top 5 recommendations: ..."
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ──────────────────────────────────────────────
# Phase 1 — Indexing
# ──────────────────────────────────────────────

def test_list_to_text_conv_returns_one_string_per_song():
    with patch("Music_RAG.load_songs", return_value=MOCK_SONGS):
        result = rag.list_to_text_conv("fake.csv")
    assert len(result) == len(MOCK_SONGS)


def test_list_to_text_conv_contains_required_fields():
    with patch("Music_RAG.load_songs", return_value=MOCK_SONGS):
        result = rag.list_to_text_conv("fake.csv")
    first = result[0]
    assert "Title: Midnight Coding" in first
    assert "Genre: lofi"            in first
    assert "Mood: chill"            in first
    assert "Energy: 0.42"           in first
    assert "Artist: LoRoom"         in first


def test_embed_songs_returns_one_vector_per_description():
    descriptions = ["desc one", "desc two", "desc three"]
    with patch("Music_RAG.DefaultEmbeddingFunction") as MockEF:
        MockEF.return_value = make_mock_ef()
        result = rag.embed_songs(descriptions)
    assert len(result) == 3


def test_embed_songs_each_vector_has_correct_length():
    descriptions = ["single description"]
    with patch("Music_RAG.DefaultEmbeddingFunction") as MockEF:
        MockEF.return_value = make_mock_ef()
        result = rag.embed_songs(descriptions)
    assert len(result[0]) == 384


def test_store_vector_stores_correct_count():
    vectors      = [[0.1] * 8, [0.2] * 8, [0.3] * 8]
    descriptions = ["Song A", "Song B", "Song C"]
    in_memory    = chromadb.EphemeralClient()

    with patch("Music_RAG.chromadb.PersistentClient", return_value=in_memory):
        collection = rag.store_vector(vectors, descriptions)

    assert collection.count() == 3


def test_store_vector_returns_chromadb_collection():
    vectors      = [[0.1] * 8]
    descriptions = ["Song A"]
    in_memory    = chromadb.EphemeralClient()

    with patch("Music_RAG.chromadb.PersistentClient", return_value=in_memory):
        collection = rag.store_vector(vectors, descriptions)

    assert isinstance(collection, chromadb.Collection)


# ──────────────────────────────────────────────
# Phase 2 — Querying
# ──────────────────────────────────────────────

def test_embed_prompt_returns_a_list():
    with patch("Music_RAG.DefaultEmbeddingFunction") as MockEF:
        MockEF.return_value = make_mock_ef()
        result = rag.embed_prompt("Give me chill lofi songs")
    assert isinstance(result, list)


def test_embed_prompt_vector_length_matches_embed_songs():
    with patch("Music_RAG.DefaultEmbeddingFunction") as MockEF:
        MockEF.return_value = make_mock_ef()
        result = rag.embed_prompt("Give me chill lofi songs")
    assert len(result) == 384


def test_nearest_songs_returns_k_results():
    client     = chromadb.EphemeralClient()
    collection = client.create_collection("test_nearest")
    vectors    = [[float(i)] * 8 for i in range(5)]
    docs       = [f"Song {i}" for i in range(5)]
    ids        = [str(i) for i in range(5)]
    collection.add(embeddings=vectors, documents=docs, ids=ids)

    results = rag.nearest_songs([0.0] * 8, collection, k=3)
    assert len(results) == 3


def test_nearest_songs_returns_strings():
    client     = chromadb.EphemeralClient()
    collection = client.create_collection("test_strings")
    collection.add(embeddings=[[0.1] * 8], documents=["Only Song"], ids=["0"])

    results = rag.nearest_songs([0.0] * 8, collection, k=1)
    assert all(isinstance(r, str) for r in results)


def test_format_retrieved_strips_to_three_fields():
    descriptions = ["Title: Midnight Coding. Artist: LoRoom. Genre: lofi. Mood: chill. Energy: 0.42."]
    result = rag.format_retrieved(descriptions)
    assert "Title: Midnight Coding" in result[0]
    assert "Genre: lofi"            in result[0]
    assert "Mood: chill"            in result[0]
    assert "Artist"                 not in result[0]
    assert "Energy"                 not in result[0]


def test_format_retrieved_correct_output_format():
    descriptions = ["Title: Song A. Artist: Artist 1. Genre: pop. Mood: happy. Energy: 0.8."]
    result = rag.format_retrieved(descriptions)
    assert result[0] == "Title: Song A | Genre: pop | Mood: happy"


def test_format_retrieved_handles_multiple_songs():
    descriptions = [
        "Title: Song A. Artist: A1. Genre: pop.  Mood: happy.   Energy: 0.8.",
        "Title: Song B. Artist: A2. Genre: rock. Mood: intense. Energy: 0.9.",
    ]
    result = rag.format_retrieved(descriptions)
    assert len(result) == 2
    assert "Song A" in result[0]
    assert "Song B" in result[1]


def test_generate_recs_returns_a_string():
    formatted_songs = ["Title: Midnight Coding | Genre: lofi | Mood: chill"]
    with patch("Music_RAG.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = make_mock_generate_response()
        result = rag.generate_recs("Give me chill songs", formatted_songs)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_recs_passes_prompt_and_context_to_llm():
    formatted_songs = ["Title: Library Rain | Genre: lofi | Mood: chill"]
    with patch("Music_RAG.Groq") as MockGroq:
        mock_client = MockGroq.return_value
        mock_client.chat.completions.create.return_value = make_mock_generate_response()
        rag.generate_recs("energetic songs", formatted_songs)

    call_kwargs = mock_client.chat.completions.create.call_args
    messages    = call_kwargs.kwargs["messages"]
    user_content = messages[-1]["content"]
    assert "energetic songs" in user_content
    assert "Library Rain"    in user_content


# ──────────────────────────────────────────────
# Integration — full pipeline via main()
# ──────────────────────────────────────────────

def test_main_pipeline_prints_output(capsys):
    in_memory = chromadb.EphemeralClient()
    try:
        in_memory.delete_collection("songs")
    except Exception:
        pass
    with patch("Music_RAG.load_songs",                return_value=MOCK_SONGS), \
         patch("Music_RAG.chromadb.PersistentClient", return_value=in_memory), \
         patch("Music_RAG.DefaultEmbeddingFunction") as MockEF, \
         patch("Music_RAG.Groq") as MockGroq:

        MockEF.return_value = make_mock_ef()
        MockGroq.return_value.chat.completions.create.return_value = make_mock_generate_response()

        rag.main("Give me 5 chill lofi songs")

    captured = capsys.readouterr()
    assert len(captured.out.strip()) > 0


def test_main_pipeline_stores_all_songs(capsys):
    in_memory = chromadb.EphemeralClient()
    try:
        in_memory.delete_collection("songs")
    except Exception:
        pass
    with patch("Music_RAG.load_songs",                return_value=MOCK_SONGS), \
         patch("Music_RAG.chromadb.PersistentClient", return_value=in_memory), \
         patch("Music_RAG.DefaultEmbeddingFunction") as MockEF, \
         patch("Music_RAG.Groq") as MockGroq:

        MockEF.return_value = make_mock_ef()
        MockGroq.return_value.chat.completions.create.return_value = make_mock_generate_response()

        rag.main("Give me 5 chill lofi songs")

    collection = in_memory.get_collection("songs")
    assert collection.count() == len(MOCK_SONGS)
