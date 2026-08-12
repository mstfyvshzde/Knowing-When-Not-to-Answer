from src.baselines import raw_answer_baseline


class FakeDataset:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def select(self, indices):
        return FakeDataset([self.records[index] for index in indices])


def test_raw_baseline_skips_empty_candidate_and_uses_next_valid(monkeypatch):
    dataset = FakeDataset(
        [
            {
                "id": "example-1",
                "question": "What is the answer?",
                "context": "The answer is gravity.",
                "answers": {
                    "text": ["gravity"],
                    "answer_start": [14],
                },
                "is_answerable": True,
            }
        ]
    )

    monkeypatch.setattr(
        raw_answer_baseline,
        "load_split",
        lambda split_name: dataset,
    )

    def fake_pipeline(*args, **kwargs):
        def fake_qa_model(**model_inputs):
            assert model_inputs["top_k"] == 5

            return [
                {
                    "answer": "",
                    "score": 0.9,
                    "start": 0,
                    "end": 0,
                },
                {
                    "answer": "gravity",
                    "score": 0.2,
                    "start": 14,
                    "end": 21,
                },
            ]

        return fake_qa_model

    monkeypatch.setattr(
        raw_answer_baseline,
        "pipeline",
        fake_pipeline,
    )

    predictions = raw_answer_baseline.run_raw_baseline(
        split_name="calibration",
        limit=1,
        device_name="cpu",
    )

    assert len(predictions) == 1
    assert predictions[0]["prediction_text"] == "gravity"
    assert predictions[0]["start"] == 14
    assert predictions[0]["end"] == 21
    assert predictions[0]["pipeline_score"] == 0.2
