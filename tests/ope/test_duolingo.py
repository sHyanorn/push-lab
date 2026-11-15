import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pytest

from push_lab import runtime
from push_lab.data import duolingo as conversion
from push_lab.data import fixtures
from push_lab.ope import duolingo as dl
from push_lab.policy.templates import ArmStats

K = dl.K


def _rounds(
    rng: np.random.Generator,
    n: int,
    days: int = 2,
    excess: float = 0.0,
    reward: np.ndarray | None = None,
) -> dl.Rounds:
    """Uniform choices among random eligible sets; with ``excess``, template A is chosen with
    probability (1 + excess) / |A_t| where eligible with others."""
    eligible = rng.random((n, K)) < 0.4
    eligible[np.arange(n), rng.integers(0, K, n)] = True
    size = eligible.sum(axis=1)
    rank = np.floor(rng.random(n) * size).astype(np.int64)
    selected = (np.cumsum(eligible, axis=1) <= rank[:, None]).sum(axis=1)
    if excess:
        others = eligible.copy()
        others[:, 0] = False
        rank = np.floor(rng.random(n) * others.sum(axis=1)).astype(np.int64)
        other = (np.cumsum(others, axis=1) <= rank[:, None]).sum(axis=1)
        a = rng.random(n) < (1 + excess) / size
        selected = np.where(eligible[:, 0] & (size > 1), np.where(a, 0, other), selected)
    recency = np.where(rng.random((n, K)) < 0.5, rng.uniform(0, 30, (n, K)), np.inf)
    return dl.Rounds(
        day=rng.integers(0, days, n),
        time=rng.random(n),
        language=rng.integers(0, 3, n),
        eligible=eligible,
        size=size,
        selected=selected,
        reward=(rng.random(n) < 0.13).astype(np.float64) if reward is None else reward,
        recency=recency,
        j=np.zeros(n, dtype=np.int64),
        since=recency.min(axis=1),
        length=rng.integers(0, 9, n),
    )


def test_logging_check_passes_uniform_choices_and_fails_two_percent_more() -> None:
    rng = np.random.default_rng(0)
    uniform = dl.logging_check(dl.logging_moments(_rounds(rng, 600_000)))
    assert uniform["uniform"] and uniform["departures"] == []
    assert uniform["ratios"]["A|1"]["value"] == 1.0
    tilted = dl.logging_check(dl.logging_moments(_rounds(rng, 600_000, excess=0.02)))
    assert not tilted["uniform"] and tilted["departures"] == ["A"]
    assert tilted["ratios"]["A|all"]["conservative"]["ci_low"] > 1.005


def _outcome_model(r: dl.Rounds) -> lgb.Booster:
    return dl.fit_outcome_model(dl.features(r, r.selected), r.reward, 1)


def test_ips_of_the_uniform_policy_equals_the_mean_reward() -> None:
    rng = np.random.default_rng(1)
    r = _rounds(rng, 20_000, days=3)
    scores = dl.Scores(rng.normal(0, 0.05, K), rng.normal(0, 0.05, 3 * K), rng.normal(0, 0.05, K))
    sample = rng.random(len(r)) < 0.2
    moments, found = dl.policy_moments(r, scores, _outcome_model(r), sample)
    result = dl.dl1(moments, found)
    assert result["baseline"]["IPS"]["value"] == result["mean_test_reward"] == r.reward.mean()
    assert set(result["rows"]) == set(dl.DL1)
    # Another policy's IPS differs from the mean reward
    assert result["rows"]["Template"]["levels"]["IPS"]["value"] != r.reward.mean()


def _planted(
    rng: np.random.Generator, n: int, gamma: float, h: float, days: int = 2, same_day: int = 0
) -> dl.Completed:
    """Completed rounds under the model of DL2, with mu_t rising with the recency of the
    round's eligible templates, so a fit that did not condition on the round would be biased;
    j_t drawn up to ``same_day``."""
    r = _rounds(rng, n, days=days, reward=np.zeros(n))
    c = np.linspace(0, 0.1, K)
    u = c[None, :] - dl.penalty(r.recency, gamma, h)
    spread = np.where(r.eligible, np.minimum(r.recency, 30), 0).sum(axis=1) / r.size / 30
    mu = 0.1 + 0.2 * spread
    done = rng.random(n) < mu * np.exp(dl.logged(r, u))
    j = rng.integers(0, same_day + 1, n)
    return dl.Completed.of(dl.Rounds(**{**r.__dict__, "reward": done.astype(np.float64), "j": j}))


def test_dl2_recovers_a_planted_penalty_and_finds_none_where_none_is_planted() -> None:
    rng = np.random.default_rng(2)
    fitted = dl.fit(dl.PaperForm(), _planted(rng, 2_000_000, 0.05, 7.0))
    gamma = fitted["parameters"]["gamma"]["conservative"]
    h = dl.read_half_life(fitted)["conservative"]
    assert gamma["ci_low"] <= 0.05 <= gamma["ci_high"] and gamma["ci_low"] > 0
    assert h["ci_low"] <= 7.0 <= h["ci_high"]
    null = dl.fit(dl.PaperForm(), _planted(rng, 2_000_000, 0.0, 7.0))
    gamma = null["parameters"]["gamma"]["conservative"]
    assert gamma["ci_low"] <= 0 <= gamma["ci_high"]
    assert dl.read_gamma(null["parameters"]["gamma"]) == "decay not detected"


def test_gamma_keeps_the_standard_error_with_h_held_where_the_full_hessian_shrinks_it() -> None:
    """With no effect and gamma near 0 at a point where the score of h is not 0, the full
    Hessian is near singular and its standard error of gamma collapses."""
    rng = np.random.default_rng(6)
    c = dl.Completed.of(_rounds(rng, 100_000, days=3))
    model = dl.PaperForm()
    theta = np.r_[np.zeros(K - 1), 0.0002, np.log(15.0)]
    gamma = dl.intervals(model, theta, c)["gamma"]
    full, held = gamma["full_hessian"]["conservative"], gamma["held_fixed"]["conservative"]
    assert full < held / 10
    assert gamma["conservative"]["se"] == held


def _moments(z: np.ndarray, day: np.ndarray, j: np.ndarray) -> dl.DayMoments:
    m = dl.DayMoments(z.shape[1])
    m.add(day, z, j)
    return m


def test_conservative_and_independent_days_standard_errors() -> None:
    rng = np.random.default_rng(3)
    n = 3_000
    z = rng.normal(size=(n, 2))
    zeros = np.zeros(n, dtype=np.int64)
    g = np.array([1.0, -0.5])
    several = _moments(z, rng.integers(0, 3, n), zeros).standard_errors(g)
    # Three days of equal size and spread: sqrt(3) apart
    assert several["conservative"] / several["independent"] == pytest.approx(np.sqrt(3), rel=0.05)
    assert several["days"] == 3
    one = _moments(z, zeros, zeros).standard_errors(g)
    assert one["conservative"] == one["independent"]
    # j = 0: the plain standard error of the mean of g . z
    assert one["conservative"] == pytest.approx(np.std(z @ g, ddof=1) / np.sqrt(n), rel=1e-12)


@pytest.mark.parametrize("k", [2, 3])
def test_within_day_factor_matches_perfectly_correlated_rows_of_one_user(k: int) -> None:
    """One user with k rows and 5 users with one row each, all in one day: if a user's rows
    are perfectly correlated with equal variance, the variance of the day's mean grows by
    sum k_u^2 / sum k_u."""
    rows = k + 5
    z = np.random.default_rng(k).normal(size=(rows, 1))
    j = np.r_[np.arange(k), np.zeros(5)].astype(np.int64)
    day = np.zeros(rows, dtype=np.int64)
    plain = _moments(z, day, np.zeros(rows, dtype=np.int64)).standard_errors(np.ones(1))
    inflated = _moments(z, day, j).standard_errors(np.ones(1))
    hand = (k**2 + 5) / (k + 5)
    assert (inflated["conservative"] / plain["conservative"]) ** 2 == pytest.approx(hand)
    assert dl.inflation(0.0) == 1.0


def _hand_rounds() -> dl.Rounds:
    """Six rounds over templates A, B, C (rewards in brackets): A and B eligible, A chosen (1),
    A chosen (0), B chosen (1), B chosen (1); A alone chosen (0); A, B and C, C chosen (0)."""
    eligible = np.zeros((6, K), dtype=bool)
    eligible[:4, :2] = True
    eligible[4, 0] = True
    eligible[5, :3] = True
    n = 6
    return dl.Rounds(
        day=np.zeros(n, dtype=np.int64),
        time=np.zeros(n),
        language=np.zeros(n, dtype=np.int64),
        eligible=eligible,
        size=eligible.sum(axis=1),
        selected=np.array([0, 0, 1, 1, 0, 2]),
        reward=np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        recency=np.full((n, K), np.inf),
        j=np.zeros(n, dtype=np.int64),
        since=np.full(n, np.inf),
        length=np.zeros(n, dtype=np.int64),
    )


def test_arm_scores_match_a_hand_computed_example() -> None:
    r = _hand_rounds()
    sums = dl.arm_sums(r, np.broadcast_to(np.arange(K), r.eligible.shape), K)
    scores = dl.raw_scores(sums)
    # A: plus (2 * 1 + 2 * 0) / 4 = 1/2, minus (2 + 2 + 1.5 * 0) / 5.5 = 8/11
    # B: plus 1, minus 2 / 5.5 = 4/11; C: no round on the minus side
    np.testing.assert_allclose(scores[:3], [-5 / 16, 7 / 4, 0.0])
    assert (scores[3:] == 0).all()


def test_arm_sums_equal_those_of_the_template_policies() -> None:
    rng = np.random.default_rng(4)
    r = _rounds(rng, 5_000)
    sums = dl.arm_sums(r, np.broadcast_to(np.arange(K), r.eligible.shape), K)
    stats = ArmStats(K)
    stats.add(r.eligible / r.size[:, None], r.selected, r.reward)
    np.testing.assert_allclose(sums[0:3], stats.plus)
    np.testing.assert_allclose(sums[3:6], stats.minus)
    np.testing.assert_allclose(sums[6:8], stats.eligible)
    rd, _ = dl.shrunk_scores(sums, stats.rounds, stats.rewards)
    np.testing.assert_allclose(rd, stats.scores())


def test_policies_choose_among_eligible_templates() -> None:
    r = _hand_rounds()
    scores = dl.Scores(np.r_[0.0, 0.0, 1.0, np.zeros(K - 3)], np.zeros(K), np.zeros(K))
    pis = dl.policies(r, scores)
    for name, pi in pis.items():
        np.testing.assert_allclose(pi.sum(axis=1), 1, err_msg=name)
        assert (pi[~r.eligible] == 0).all(), name
    # Ties go to the earliest letter; C wins where eligible
    np.testing.assert_array_equal(pis["Template"].argmax(axis=1), [0, 0, 0, 0, 0, 2])
    # No eligible template in the history: Reuse Last is uniform
    np.testing.assert_allclose(pis["Reuse Last Template"], pis["Baseline"])
    recent = dl.Rounds(
        **{**r.__dict__, "recency": np.where(np.arange(K) == 1, 2.0, np.inf)[None, :].repeat(6, 0)}
    )
    assert dl.reuse_last(recent)[0, 1] == 1.0


def test_curve_stops_on_a_recency_outside_its_bins() -> None:
    rng = np.random.default_rng(5)
    c = _planted(rng, 50_000, 0.0, 7.0, days=5, same_day=2)
    c.recency[0, np.flatnonzero(c.eligible[0])[0]] = 31.0
    out = dl.dl2(c)
    assert "stopped" in out["curve"] and "gamma_reading" in out


def test_slices_of_a_row_group_give_the_same_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    rows = fixtures.duolingo_raw(raw, 3_000, row_group=3_000)
    conversion.convert(raw, processed, rows, {s: (f"{s}-part-1",) for s in rows})
    path = conversion.files(processed, "train")[0]
    languages = ["de", "en", "es", "fr", "pt"]
    whole = dl.train_job((0, path, 0, languages))
    monkeypatch.setattr(dl, "SLICE", 700)
    sliced = dl.train_job((0, path, 0, languages))
    np.testing.assert_allclose(sliced["template"], whole["template"])
    np.testing.assert_array_equal(sliced["x"], whole["x"])
    np.testing.assert_array_equal(sliced["completed"].recency, whole["completed"].recency)
    for key, m in whole["logging"].items():
        np.testing.assert_allclose(sliced["logging"][key].mean(), m.mean(), err_msg=key)


def test_the_analysis_runs_on_a_converted_synthetic_archive(tmp_path: Path) -> None:
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    rows = fixtures.duolingo_raw(raw, 6_000, row_group=3_000)
    conversion.convert(raw, processed, rows, {s: (f"{s}-part-1",) for s in rows})
    out = dl.run(processed, 2)
    # The result file is JSON: every count is a plain int
    json.dumps(out)
    assert out["interpreted"] and out["DL1"]["answer"] in ("replicated", "differs")
    assert set(out["DL3"]) >= {"DL3a", "DL3b", "levels", "prior_strength"}
    assert len(out["DL2"]["grid"]) == 35 and "parameters" in out["DL2"]["curve"]


def test_the_analysis_counts_the_conversion_time_toward_the_stop_line(tmp_path: Path) -> None:
    """A conversion whose two splits took the whole line of D074 between them, 1.5 times
    6,900 s, leaves the analysis no time: it stops before its first pass."""
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    rows = fixtures.duolingo_raw(raw, 1_000)
    conversion.convert(raw, processed, rows, {s: (f"{s}-part-1",) for s in rows})
    path = processed / conversion.REPORT
    report = json.loads(path.read_text("utf-8"))
    halves = {split: report[split] | {"seconds": 10_350 / 2} for split in ("train", "test")}
    path.write_text(json.dumps(report | halves), "utf-8")
    with pytest.raises(Exception, match="at the stop line of 10,350 s") as stopped:
        dl.run(processed, 1)
    assert isinstance(stopped.value, runtime.RunStopped)


def test_dl2_checks_the_line_at_each_likelihood_evaluation() -> None:
    """A line that passes at the fifth check stops the first fit within its iterations, which a
    single check before fitting would not."""

    class FifthCheck(runtime.Deadline):
        checks = 0

        def check(self) -> None:
            self.checks += 1
            if self.checks == 5:
                raise runtime.RunStopped("fifth check")

    c = _planted(np.random.default_rng(7), 20_000, 0.05, 7.0)
    with pytest.raises(runtime.RunStopped, match="fifth check"):
        dl.dl2(c, FifthCheck(60))


@pytest.mark.parametrize("test_start", [2.5, 3.0])
def test_the_analysis_stops_unless_training_precedes_test(
    tmp_path: Path, test_start: float
) -> None:
    """Training ends at day 3. Test times from day 2.5 on overlap it, as a test clock that
    restarted would, and day numbers would then mix the splits; a test split that starts at the
    last training time is allowed."""
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    rng = np.random.default_rng(8)
    for split, days in (("train", (0.0, 3.0)), ("test", (test_start, 6.0))):
        table = fixtures.duolingo_table(rng, 1_000, days)
        when = table["datetime"].to_numpy().copy()
        when[0] = days[1] if split == "train" else days[0]
        table = table.set_column(0, table.schema.field(0), pa.array(when))
        fixtures.duolingo_archive(raw / f"{split}-part-1.tar.gz", [table])
    rows = {"train": 1_000, "test": 1_000}
    conversion.convert(raw, processed, rows, {s: (f"{s}-part-1",) for s in rows})
    if test_start < 3.0:
        with pytest.raises(ValueError, match="do not precede test times"):
            dl.run(processed, 1)
    else:
        ranges = dl.run(processed, 1)["conversion"]
        assert ranges["train"]["datetime_max"] == ranges["test"]["datetime_min"] == 3.0
