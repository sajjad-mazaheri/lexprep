import pandas as pd

from lexprep.sampling.stratified import stratified_sample_quantiles


def test_stratified_sample_size():
    df = pd.DataFrame({"score": list(range(100))})
    out, report = stratified_sample_quantiles(
        df, score_col="score", n_total=30, bins=3, random_state=1,
    )
    assert len(out) == report.total_sampled
    assert report.total_sampled == 30
