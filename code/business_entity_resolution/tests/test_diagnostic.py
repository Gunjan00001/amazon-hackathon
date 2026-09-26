def test_oracle_macro():
    from ber.diagnostic import oracle_macro

    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-9"}}
    cand = {"S1-1": {"S2-1", "S2-7"}, "S1-2": set()}
    assert round(oracle_macro(truth, cand), 4) == round((0.8333333333 + 0.0) / 2, 4)


def test_oracle_macro_singleton_scores_one():
    from ber.diagnostic import oracle_macro

    truth = {"S1-1": set()}
    cand = {"S1-1": {"S2-1"}}
    assert oracle_macro(truth, cand) == 1.0
