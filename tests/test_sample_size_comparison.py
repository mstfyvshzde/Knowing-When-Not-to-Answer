from experiments.compare_question_aware_ablation_sample_sizes import (
    deterministic_nested_order,
)


def test_deterministic_nested_order_is_reproducible_and_nested():
    records = [{"id": index} for index in range(20)]

    first_order = deterministic_nested_order(records, seed=42)
    second_order = deterministic_nested_order(records, seed=42)

    assert first_order == second_order
    assert first_order is not records
    assert records == [{"id": index} for index in range(20)]

    subset_5 = first_order[:5]
    subset_10 = first_order[:10]
    subset_15 = first_order[:15]

    ids_5 = {record["id"] for record in subset_5}
    ids_10 = {record["id"] for record in subset_10}
    ids_15 = {record["id"] for record in subset_15}

    assert ids_5 < ids_10
    assert ids_10 < ids_15
