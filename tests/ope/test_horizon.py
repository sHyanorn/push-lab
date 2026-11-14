from typing import Any

import numpy as np
import pytest

from push_lab.ope import horizon as hz
from push_lab.ope import simulator as sim_ope
from push_lab.sim import env, world
from push_lab.sim.explore import EpsilonGreedy
from push_lab.sim.world import DAYS, Params

from ..policy.conftest import make_world

USERS = 400
TARGET_P = np.array([0.9, 0.7])  # target's probability of action 1 in states 0 and 1


def _context(control: bool, users: int = USERS) -> sim_ope.Context:
    pop, static = make_world(seed=3, users=users)
    pop = env.Population(pop.visit_rate, pop.profile, pop.p0 * 0.5)
    return sim_ope.make_context(pop, static, control)


CTX = _context(control=False)


def _chain(
    rng: np.random.Generator, n: int, horizon: int, log_p: np.ndarray, target_p: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A two-state chain whose state is the previous action and whose reward depends on it:
    states, actions, rewards and weights [round, user]. ``log_p`` and ``target_p`` give the
    probability of action 1 in each state."""
    s = np.zeros((horizon, n), dtype=np.int64)
    a = np.zeros((horizon, n), dtype=np.int64)
    r = np.zeros((horizon, n))
    for t in range(horizon):
        if t:
            s[t] = a[t - 1]
        a[t] = rng.random(n) < log_p[s[t]]
        r[t] = rng.random(n) < 0.1 + 0.5 * s[t] + 0.2 * a[t]
    pi = np.where(a == 1, target_p[s], 1 - target_p[s])
    b = np.where(a == 1, log_p[s], 1 - log_p[s])
    return s, a, r, pi / b


def _chain_value(horizon: int, target_p: np.ndarray) -> float:
    """Exact mean reward per round of the target, starting in state 0."""
    p1, total = 0.0, 0.0
    for _ in range(horizon):
        pa = (1 - p1) * target_p[0] + p1 * target_p[1]
        total += 0.1 + 0.5 * p1 + 0.2 * pa
        p1 = pa
    return total / horizon


def test_estimators_on_a_chain_with_carryover() -> None:
    horizon, log_p = 5, np.array([0.5, 0.5])
    truth = _chain_value(horizon, TARGET_P)
    rng = np.random.default_rng(0)
    # PDIS is unbiased: its mean over 200 logs lies within 3 standard errors of the value
    values = []
    for _ in range(200):
        _, _, r, w = _chain(rng, 1_000, horizon, log_p, TARGET_P)
        values.append(hz.pdis(w, r)[0].sum() / (1_000 * horizon))
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    assert abs(np.mean(values) - truth) < 3 * se
    n = 20_000
    s, a, r, w = _chain(rng, n, horizon, log_p, TARGET_P)
    once = np.ones((1, n))
    value, _ = hz.smis(s, 2, w, r, once)
    assert abs(value[0] - truth) < 0.01
    # Without the carried-over variable SMIS weights each round alone and misses the carryover
    blind, _ = hz.smis(np.zeros_like(s), 1, w, r, once)
    assert abs(blind[0] - truth) > 0.01
    x = s[..., None].astype(np.float64)
    mixture = [(TARGET_P[s], np.ones((n, 1))), (1 - TARGET_P[s], np.zeros((n, 1)))]
    v0 = hz.fqe(x, a[..., None].astype(np.float64), r, mixture)
    assert abs(v0.mean() / horizon - truth) < 0.01


def test_target_equal_to_logging_gives_unit_weights_ratios_and_mass() -> None:
    log = hz.simulate(CTX, Params(), "E1", 5_001)
    assert log.explorer is not None
    rows = sim_ope.user_days(log.out, CTX.static)
    b = sim_ope.logging_probability(log.explorer, rows.bits, rows.user)
    w = (b / b).reshape(DAYS, USERS)
    sums, diagnostics = hz.pdis(w, log.out.visited.astype(np.float64))
    np.testing.assert_array_equal(sums, log.out.visited.sum(axis=0))
    assert diagnostics["max_rho"] == 1.0 and diagnostics["effective_size"] == USERS
    s = hz.state("S0", log.out, Params(), hz.round_starts("E1"))
    trace: list[np.ndarray] = []
    _, mass = hz.smis(
        s.index, s.bins, w, log.out.visited.astype(np.float64), np.ones((1, USERS)), trace
    )
    assert len(trace) == DAYS - 1
    assert all(np.all(t[~np.isnan(t)] == 1.0) for t in trace)
    assert mass[0] == 1.0


def test_mass_falls_when_the_target_leaves_the_logged_actions() -> None:
    rng = np.random.default_rng(1)
    n, horizon = 20_000, 4
    # The logs never take action 1 in state 1
    log_p = np.array([0.5, 0.0])
    for target_p, inside in ((np.array([0.5, 0.0]), True), (np.array([0.5, 0.5]), False)):
        s, _, r, w = _chain(rng, n, horizon, log_p, target_p)
        _, mass = hz.smis(s, 2, w, r, np.ones((1, n)))
        assert (abs(mass[0] - 1) < 0.02) if inside else (mass[0] < 0.9)


def test_e2_draws_an_arm_each_week_and_e3_one_per_user() -> None:
    n = 5_000
    e2, e3 = hz.draw_arms("E2", 6_001, n), hz.draw_arms("E3", 7_001, n)
    sd = np.sqrt(n * 0.25 * 0.75)
    for arms in (e2, e3):
        assert arms.shape == (hz.WEEKS, n)
        for week in arms:
            assert (np.abs(np.bincount(week, minlength=4) - n / 4) < 4 * sd).all()
    # Independent weekly draws: a user keeps its arm from one week to the next with p = 1/4
    for w in range(1, hz.WEEKS):
        assert abs((e2[w] == e2[w - 1]).mean() - 0.25) < 4 * np.sqrt(0.25 * 0.75 / n)
    assert (e3 == e3[0]).all()


def test_users_whose_arms_are_all_x_receive_what_x_gives_them() -> None:
    ctx = _context(control=False, users=5_000)
    params, seed = Params(), 6_001
    log = hz.simulate(ctx, params, "E2", seed)
    assert log.arms is not None
    for i, target in enumerate(hz.TARGETS):
        policy = sim_ope.target_policy(target, ctx, seed)
        alone = env.simulate(
            ctx.pop, params, policy, env.draws(ctx.users, seed), sim_ope.WORLD_SEED
        )
        users = np.flatnonzero((log.arms == i).all(axis=0))
        assert len(users) >= 5
        others = np.flatnonzero((log.arms != i).any(axis=0))
        np.testing.assert_array_equal(log.out.send[:, :, users], alone.send[:, :, users])
        np.testing.assert_array_equal(log.out.template[:, :, users], alone.template[:, :, users])
        np.testing.assert_array_equal(log.out.visited[:, users], alone.visited[:, users])
        # A user with another arm in some week is sent something else by every target but N
        if target != "N":
            assert (log.out.send[:, :, others] != alone.send[:, :, others]).any()


def test_oracle_state_reproduces_the_simulators_organic_visit_probability() -> None:
    params = Params(return_boost=0.05, fatigue_half_life=3.0)
    explorer = EpsilonGreedy(CTX.best, seed=2)
    out = env.simulate(
        CTX.pop, params, explorer, env.draws(USERS, 2), sim_ope.WORLD_SEED, truth=True
    )
    starts = range(DAYS)
    volume = hz.decayed_volume(out.delivered, params.fatigue_half_life, starts)
    stock = hz.return_stock(out.opened, params.return_half_life, starts)
    organic = np.minimum(
        1.0,
        CTX.pop.visit_rate * np.exp(-world.annoyance_rate(params) * volume)
        + params.return_boost * stock,
    )
    assert out.organic is not None
    np.testing.assert_array_equal(organic.astype(np.float32), out.organic)


def test_seven_day_counts_and_bins() -> None:
    log = hz.simulate(CTX, Params(), "E1", 5_002)
    starts = hz.round_starts("E2")
    s = hz.state("S0", log.out, Params(), starts)
    delivered = log.out.delivered.sum(axis=1)
    for w, start in enumerate(starts):
        days = slice(max(0, start - 7), start)
        np.testing.assert_array_equal(s.features[w, :, 0], delivered[days].sum(axis=0))
        np.testing.assert_array_equal(s.features[w, :, 2], log.out.visited[days].sum(axis=0))
    assert s.bins == 60 and s.index.max() < 60
    assert (s.index[0] == 0).all()  # nothing before the first round
    assert hz.state("S_none", log.out, Params(), starts).bins == 4
    # Each bin includes its lower edge
    np.testing.assert_array_equal(
        hz._bin(np.array([0, 1, 3, 4, 6, 7, 9, 10, 25]), hz.DELIVERY_EDGES),
        [0, 1, 1, 2, 2, 3, 3, 4, 4],
    )


def test_both_targets_of_a_difference_share_their_resamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hz, "DIFFS", ("T",))
    out = hz.estimate_log(CTX, Params(), "E2", 6_003)
    for entry in out["estimates"].values():
        d = entry["D"]["T"]
        assert d["value"] == d["ci_low"] == d["ci_high"] == 0.0


def test_truth_runs_share_their_seeds_across_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, int]] = []

    def fake(ctx: Any, params: Params, target: str, seed: int) -> dict[str, float]:
        seen.append((target, seed))
        return {"visit": 0.0, "opens": 0.0}

    monkeypatch.setattr(sim_ope, "truth", fake)
    for k in range(3):
        hz.truth_run(CTX, Params(), k)
    for target in hz.TARGETS:
        assert [s for t, s in seen if t == target] == [300_000, 300_001, 300_002]


def test_truth_differences_are_paired() -> None:
    rng = np.random.default_rng(2)
    common = rng.normal(0.5, 0.05, 200)
    runs = [{"N": c + 0.01, "T_half": c, "T": c, "T2": c - 0.002} for c in common]
    summary = hz.truth_summary(runs, None)
    assert summary["D"]["N"]["value"] == pytest.approx(0.01)
    assert summary["D"]["N"]["se"] < 1e-12 < summary["levels"]["N"]["se"]
    assert summary["D"]["T2"]["sign"] == -1 and summary["D"]["T_half"]["sign"] == 0


def test_control_truth_is_the_exact_value_of_14_3(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(control=True)
    monkeypatch.setattr(sim_ope, "truth", lambda *a: pytest.fail("the control is not simulated"))
    summary = hz.setting_truth(ctx, "control", None)
    exact = {t: sim_ope.exact_truth(ctx, sim_ope.CONTROL, t)["visit"] for t in hz.TARGETS}
    for x in hz.DIFFS:
        assert summary["D"][x]["value"] == exact[x] - exact["T"]
        assert summary["D"][x]["se"] == 0.0


def _check_settings(bias: float) -> dict[str, Any]:
    """Summaries of PDIS-week and arm means whose estimates sit at the truth plus ``bias`` (arm
    means only), with noise of standard deviation 0.01 over logs."""
    rng = np.random.default_rng(3)
    truth = {"D": {x: {"value": 0.02, "se": 0.0005, "sign": 1} for x in hz.DIFFS}}
    out: dict[str, Any] = {}
    for setting, m in (("control", 10), ("reference", 50)):
        designs: dict[str, Any] = {}
        for design, name in hz.CHECK_ESTIMATORS:
            shift = bias if name == "arm_means" else 0.0
            logs = []
            for _ in range(m):
                d = {}
                for x in hz.DIFFS:
                    v = 0.02 + shift + 0.01 * rng.standard_normal()
                    d[x] = {"value": v, "ci_low": v - 0.02, "ci_high": v + 0.02}
                logs.append({"estimates": {name: {"D": d}}})
            designs[design] = {name: hz.summarize_estimator(logs, name, truth)}
        out[setting] = {"designs": designs}
    return out


def test_implementation_check_fails_on_a_planted_bias() -> None:
    assert hz.implementation_check(_check_settings(0.0))["passed"]
    # Four standard errors of the 10-log bias at the control
    check = hz.implementation_check(_check_settings(4 * 0.01 / np.sqrt(10)))
    assert not check["passed"]
    failed = {(r["setting"], r["estimator"]) for r in check["intervals"] if not r["contains_zero"]}
    assert ("reference", "PDIS_week") not in failed and ("reference", "arm_means") in failed


def _estimates(values: np.ndarray, half: float = 0.01) -> list[dict[str, float]]:
    return [{"value": v, "ci_low": v - half, "ci_high": v + half} for v in values]


def test_labels() -> None:
    rng = np.random.default_rng(4)
    truth = {"value": 0.02, "se": 0.0005, "sign": 1}
    good = hz.measures(_estimates(0.02 + 0.005 * rng.standard_normal(50)), truth)
    assert good["verdict"] == "validated" and good["coverage_read"]
    assert good["label"] == "recovers the ordering"
    wide = hz.measures(_estimates(0.02 + 0.015 * rng.standard_normal(50), 0.03), truth)
    assert wide["label"] == "validated, does not separate" and wide["separation"] < 0.8
    flipped = hz.measures(_estimates(-0.02 + 0.005 * rng.standard_normal(10)), truth)
    assert flipped["label"] == "reverses" and "bias" in flipped["reasons"]
    # Ten logs: coverage is not read, however far it is from 0.95
    narrow = hz.measures(_estimates(0.02 + 0.005 * rng.standard_normal(10), 1e-6), truth)
    assert narrow["coverage"]["value"] == 0 and narrow["verdict"] == "validated"
    no_sign = {"value": 0.0, "se": 0.001, "sign": 0}
    flat = hz.measures(_estimates(0.0002 * rng.standard_normal(10)), no_sign)
    assert flat["label"] == "validated" and flat["direction"] is None


def test_jobs_follow_the_design() -> None:
    seeds = hz.design_seeds()
    work = hz.jobs(seeds)
    logs = [j for j in work if j[0] == "log"]
    assert len(logs) == 3 * (50 + 7 * 10)
    assert len(work) - len(logs) == 7 * hz.TRUTH_RUNS
    with_controls = [j for j in logs if len(j[4]) > 1]
    assert len(with_controls) == 3 * 2 * 10
    assert {j[1] for j in with_controls} == set(hz.STATE_SETTINGS)
    assert all(j[3] in hz.LOG_SEEDS[j[2]][:10] and j[2] != "E3" for j in with_controls)
    refits = [j for j in logs if j[5]]
    assert len(refits) == 50 and {(j[1], j[2]) for j in refits} == {("reference", "E2")}
    used = set().union(*(set(v) for v in hz.LOG_SEEDS.values()))
    assert not used & set(sim_ope.LOG_SEEDS) and hz.SURROGATE_SEED not in used


def test_fixed_seeds_give_identical_results() -> None:
    surrogate = hz.fit_surrogate(CTX, Params())
    for design, refits in (("E1", 0), ("E2", 2), ("E3", 0)):
        runs = [
            hz.estimate_log(CTX, Params(), design, 5_010, ("S0", "S_oracle"), refits, surrogate)
            for _ in range(2)
        ]
        for r in runs:
            r.pop("seconds")
        assert runs[0] == runs[1]


TRUE_D = {"N": 0.02, "T_half": -0.01, "T2": -0.02}


def _fake_logs(
    seeds: dict[str, dict[str, tuple[int, ...]]], shift: float
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Per-log results whose D_X estimates sit at ``TRUE_D`` plus
    ``shift`` for every estimator."""
    rng = np.random.default_rng(5)
    keys = {
        "E1": ["DR_day", "PDIS"] + [f"{n}|{s}" for n in ("SMIS", "FQE") for s in hz.STATES],
        "E2": ["PDIS_week"] + [f"{n}|{s}" for n in ("SMIS_week", "FQE_week") for s in hz.STATES],
        "E3": ["arm_means", "surrogate"],
    }
    out = {}
    for setting, by_design in seeds.items():
        for design, design_seeds in by_design.items():
            for seed in design_seeds:
                estimates = {}
                for key in keys[design]:
                    d = {}
                    for x in hz.DIFFS:
                        v = TRUE_D[x] + shift + 0.004 * rng.standard_normal()
                        d[x] = {"value": v, "ci_low": v - 0.01, "ci_high": v + 0.01}
                    estimates[key] = {"D": d}
                out[(setting, design, seed)] = {
                    "estimates": estimates,
                    "arm_users": dict.fromkeys(hz.TARGETS, 1_250),
                }
    return out


def test_summary_of_every_setting() -> None:
    seeds = hz.design_seeds()
    runs = [{t: 0.5 + TRUE_D.get(t, 0.0) for t in hz.TARGETS}] * 2
    truths = {s: hz.truth_summary(runs, None) for s in hz.SETTINGS}
    out = hz.summarize(_fake_logs(seeds, 0.0), truths, seeds)
    assert out["implementation_check"]["passed"]
    assert len(out["implementation_check"]["intervals"]) == 12
    labels = out["primary"]["E2"]["labels"]
    assert set(labels) == {"PDIS_week", "SMIS_week", "FQE_week"}
    assert out["primary"]["E1"]["recovers the ordering"] is True
    found = out["settings"]["reference"]["designs"]["E1"]
    assert found["SMIS|S0"]["N"]["logs"] == 50
    assert found["SMIS|S_oracle"]["N"]["logs"] == found["SMIS|S0|first_10"]["N"]["logs"] == 10
    assert "SMIS|S_oracle" not in out["settings"]["annoyance_0"]["designs"]["E1"]
    reading = out["state_reading"]["reference"]["FQE_week"]["N"]
    assert reading["S_oracle"] and not reading["depends on knowing the memory"]
    projection = out["arm_mean_projection"]["N"]
    assert projection["projected"] == pytest.approx(0.01 * np.sqrt(5_000 / 27_285))
    biased = hz.summarize(_fake_logs(seeds, 0.05), truths, seeds)
    assert not biased["implementation_check"]["passed"]
    assert biased["primary"]["E3"]["labels"]["arm_means"]["N"] == "not validated"
