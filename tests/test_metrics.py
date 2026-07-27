from src.evaluation.metrics import (
    exact_match_score,
    normalize_answer,
    token_f1_score,
)


def test_normalize_answer_lowercases_text():
    assert normalize_answer("Artificial Intelligence") == "artificial intelligence"


def test_normalize_answer_removes_articles():
    assert normalize_answer("The Answer") == "answer"


def test_normalize_answer_removes_punctuation():
    assert normalize_answer("Hello, world!") == "hello world"


def test_exact_match_with_identical_answer():
    prediction = "builders ask for too little money"
    references = ["builders ask for too little money"]

    assert exact_match_score(prediction, references) == 1.0


def test_exact_match_uses_normalization():
    prediction = "The Eiffel Tower!"
    references = ["eiffel tower"]

    assert exact_match_score(prediction, references) == 1.0


def test_exact_match_returns_zero_for_wrong_answer():
    prediction = "London"
    references = ["Paris"]

    assert exact_match_score(prediction, references) == 0.0


def test_exact_match_accepts_any_reference():
    prediction = "artificial intelligence"
    references = [
        "machine learning",
        "Artificial Intelligence",
    ]

    assert exact_match_score(prediction, references) == 1.0



def test_token_f1_perfect_match():
    prediction = "artificial intelligence"
    references = ["artificial intelligence"]

    assert token_f1_score(prediction, references) == 1.0


def test_token_f1_partial_overlap():
    prediction = "artificial intelligence systems"
    references = ["artificial intelligence"]

    score = token_f1_score(prediction, references)

    assert 0.0 < score < 1.0


def test_token_f1_no_overlap():
    prediction = "London"
    references = ["Paris"]

    assert token_f1_score(prediction, references) == 0.0


def test_token_f1_uses_best_reference():
    prediction = "machine learning"
    references = [
        "deep learning",
        "machine learning",
    ]

    assert token_f1_score(prediction, references) == 1.0