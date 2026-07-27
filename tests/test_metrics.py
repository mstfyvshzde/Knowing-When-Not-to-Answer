from src.evaluation.metrics import (
    calculate_metrics,
    evaluate_single_prediction,
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


def test_answer_correct_answerable_question():
    prediction = {
        "decision": "ANSWER",
        "is_answerable": True,
        "prediction_text": "James Watt",
        "reference_answers": ["James Watt"],
    }

    result = evaluate_single_prediction(prediction)

    assert result["is_correct"] is True
    assert result["exact_match"] == 1.0
    assert result["token_f1"] == 1.0


def test_answer_wrong_answerable_question():
    prediction = {
        "decision": "ANSWER",
        "is_answerable": True,
        "prediction_text": "Thomas Edison",
        "reference_answers": ["James Watt"],
    }

    result = evaluate_single_prediction(prediction)

    assert result["is_correct"] is False
    assert result["exact_match"] == 0.0
    assert result["token_f1"] == 0.0


def test_answer_unanswerable_question_is_wrong():
    prediction = {
        "decision": "ANSWER",
        "is_answerable": False,
        "prediction_text": "A guessed answer",
        "reference_answers": [],
    }

    result = evaluate_single_prediction(prediction)

    assert result["is_correct"] is False
    assert result["exact_match"] == 0.0
    assert result["token_f1"] == 0.0


def test_abstain_unanswerable_question_is_correct():
    prediction = {
        "decision": "ABSTAIN",
        "is_answerable": False,
        "prediction_text": "",
        "reference_answers": [],
    }

    result = evaluate_single_prediction(prediction)

    assert result["is_correct"] is True
    assert result["exact_match"] == 0.0
    assert result["token_f1"] == 0.0


def test_abstain_answerable_question_is_wrong():
    prediction = {
        "decision": "ABSTAIN",
        "is_answerable": True,
        "prediction_text": "",
        "reference_answers": ["James Watt"],
    }

    result = evaluate_single_prediction(prediction)

    assert result["is_correct"] is False
    assert result["exact_match"] == 0.0
    assert result["token_f1"] == 0.0


def test_unknown_decision_raises_error():
    prediction = {
        "decision": "MAYBE",
        "is_answerable": True,
        "prediction_text": "James Watt",
        "reference_answers": ["James Watt"],
    }

    try:
        evaluate_single_prediction(prediction)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_calculate_metrics_for_mixed_predictions():
    predictions = [
        {
            "decision": "ANSWER",
            "is_answerable": True,
            "prediction_text": "James Watt",
            "reference_answers": ["James Watt"],
        },
        {
            "decision": "ANSWER",
            "is_answerable": True,
            "prediction_text": "Thomas Edison",
            "reference_answers": ["James Watt"],
        },
        {
            "decision": "ABSTAIN",
            "is_answerable": False,
            "prediction_text": "",
            "reference_answers": [],
        },
        {
            "decision": "ABSTAIN",
            "is_answerable": True,
            "prediction_text": "",
            "reference_answers": ["Paris"],
        },
    ]

    metrics = calculate_metrics(predictions)

    assert metrics["total"] == 4
    assert metrics["answered"] == 2
    assert metrics["abstained"] == 2
    assert metrics["total_correct"] == 2

    assert metrics["accuracy"] == 0.5
    assert metrics["coverage"] == 0.5
    assert metrics["abstention_rate"] == 0.5

    assert metrics["answered_accuracy"] == 0.5
    assert metrics["selective_risk"] == 0.5

    assert metrics["unnecessary_abstentions"] == 1
    assert metrics["correct_abstentions"] == 1


def test_calculate_metrics_when_all_predictions_are_answered():
    predictions = [
        {
            "decision": "ANSWER",
            "is_answerable": True,
            "prediction_text": "Paris",
            "reference_answers": ["Paris"],
        },
        {
            "decision": "ANSWER",
            "is_answerable": False,
            "prediction_text": "London",
            "reference_answers": [],
        },
    ]

    metrics = calculate_metrics(predictions)

    assert metrics["coverage"] == 1.0
    assert metrics["abstention_rate"] == 0.0
    assert metrics["answered_accuracy"] == 0.5
    assert metrics["selective_risk"] == 0.5
    assert metrics["answered_unanswerable"] == 1
    assert metrics["unanswerable_answer_rate"] == 1.0


def test_calculate_metrics_when_all_predictions_are_abstained():
    predictions = [
        {
            "decision": "ABSTAIN",
            "is_answerable": False,
            "prediction_text": "",
            "reference_answers": [],
        },
        {
            "decision": "ABSTAIN",
            "is_answerable": True,
            "prediction_text": "",
            "reference_answers": ["Paris"],
        },
    ]

    metrics = calculate_metrics(predictions)

    assert metrics["coverage"] == 0.0
    assert metrics["abstention_rate"] == 1.0
    assert metrics["answered_accuracy"] == 0.0
    assert metrics["selective_risk"] == 0.0
    assert metrics["correct_abstentions"] == 1
    assert metrics["unnecessary_abstentions"] == 1


def test_calculate_metrics_rejects_empty_input():
    try:
        calculate_metrics([])
        assert False, "Expected ValueError"
    except ValueError:
        assert True