import numpy as np
import pytest
from scipy.stats import spearmanr

from push_lab.policy import templates as tp
from push_lab.sim import env, world
from push_lab.sim.world import DAYS, SLOTS, Params

from .conftest import CANDIDATES, USERS, make_world

K = 60
FAMILY = np.arange(K) % 6
WORLD_SEED = 3


def _inputs(seed: int = 0) -> tuple[env.Population, np.ndarray, np.ndarray, np.ndarray]:
    pop, static = make_world(seed=seed)
    rng = np.random.default_rng(seed + 1)
    creator = rng.integers(0, 3, (USERS, CANDIDATES)) * (rng.random((USERS, CANDIDATES)) < 0.2)
    rank = np.where(
        rng.random((USERS, CANDIDATES)) < 0.3, rng.integers(1, 101, (USERS, CANDIDATES)), 0
    )
    return pop, static.profile.argmax(axis=1), creator, rank


INPUTS = _inputs()


def _policy(kind: tp.Kind, params: Params, run_seed: int = 7) -> tp.Templated:
    _, slot, creator, rank = INPUTS
    return tp.Templated(
        kind,
        slot,
        FAMILY,
        creator,
        rank,
        run_seed,
        copy=world.copy_effects(params.copy_spread, WORLD_SEED, K),
        novelty=params.novelty,
        novelty_half_life=params.novelty_half_life,
    )


def _run(kind: tp.Kind, params: Params, seed: int = 7) -> tuple[env.Outcome, tp.Templated]:
    policy = _policy(kind, params, seed)
    out = env.simulate(INPUTS[0], params, policy, env.draws(USERS, seed), WORLD_SEED, K)
    return out, policy


def _repeat_rate(out: env.Outcome, days: float = 15) -> float:
    """Share of deliveries whose template reached the same user within ``days``."""
    last = np.full((USERS, K), -np.inf)
    repeats = total = 0
    for d in range(DAYS):
        for s in range(SLOTS):
            users = np.flatnonzero(out.delivered[d, s])
            k = out.template[d, s, users].astype(np.int64)
            t = d + world.SLOT_HOURS[s] / 24
            repeats += int((t - last[users, k] <= days).sum())
            total += len(users)
            last[users, k] = t
    return repeats / total


def test_arm_stats_hand_example() -> None:
    stats = tp.ArmStats(2)
    probs = np.array([[0.5, 0.5], [0.25, 0.75], [0.5, 0.5]])
    stats.add(probs, np.array([0, 1, 1]), np.array([1.0, 0.0, 1.0]))
    mean, size = tp.ArmStats._mean_and_size(stats.plus)
    np.testing.assert_allclose(mean, [1.0, 2 / (4 / 3 + 2)])
    np.testing.assert_allclose(size, [1.0, (10 / 3) ** 2 / (16 / 9 + 4)])
    mean, size = tp.ArmStats._mean_and_size(stats.minus)
    np.testing.assert_allclose(mean, [2 / (4 / 3 + 2), 1.0])
    np.testing.assert_allclose(size, [(10 / 3) ** 2 / (16 / 9 + 4), 1.0])
    # Arm means differ by less than their noise: full shrinkage, scores 0
    np.testing.assert_array_equal(stats.scores(), [0.0, 0.0])


def test_arm_stats_finds_a_clear_winner() -> None:
    stats = tp.ArmStats(2)
    chosen = np.arange(1_000) % 2
    stats.add(np.full((1_000, 2), 0.5), chosen, (chosen == 0).astype(float))
    s = stats.scores()
    assert s[0] > 0 > s[1]
    assert s[1] == pytest.approx(-1, abs=1e-3)
    raw = stats.scores(raw=True)
    assert raw[0] == pytest.approx(1, abs=1e-3) and raw[1] == pytest.approx(-1, abs=1e-3)


def test_uniform_logging_baseline_is_the_other_arms_mean() -> None:
    rng = np.random.default_rng(0)
    chosen = rng.integers(0, 3, 600)
    reward = (rng.random(600) < 0.3).astype(float)
    stats = tp.ArmStats(3)
    stats.add(np.full((600, 3), 1 / 3), chosen, reward)
    mean, _ = tp.ArmStats._mean_and_size(stats.minus)
    for a in range(3):
        assert mean[a] == pytest.approx(reward[chosen != a].mean())


def test_ineligible_and_extreme_probabilities_are_left_out() -> None:
    stats = tp.ArmStats(3)
    probs = np.array([[0.999, 0.001, 0.0]])
    stats.add(probs, np.array([0]), np.array([1.0]))
    np.testing.assert_array_equal(stats.eligible[0], [0, 0, 0])
    np.testing.assert_array_equal(stats.plus[0], [0, 0, 0])


def test_inverse_cdf_never_picks_a_zero_probability() -> None:
    probs = np.tile([0.0, 0.5, 0.0, 0.5], (5, 1))
    u = np.array([0.0, 0.49, 0.5, 0.999, 1 - 1e-17])
    np.testing.assert_array_equal(tp.inverse_cdf(probs, u), [1, 1, 3, 3, 3])


def test_softmax_respects_eligibility() -> None:
    p = tp.softmax(np.array([[0.0, 0.01, 0.02]]), np.array([[True, False, True]]), 0.005)
    assert p[0, 1] == 0
    assert p.sum() == pytest.approx(1)
    assert p[0, 2] / p[0, 0] == pytest.approx(np.exp(4))


def test_all_kinds_send_the_same_and_choose_eligible_templates() -> None:
    params = Params(base_rate=0.2, copy_spread=0.2, novelty=0.03)
    outs = {kind: _run(kind, params)[0] for kind in tp.KINDS}
    for out in outs.values():
        np.testing.assert_array_equal(out.send, outs["U"].send)
        np.testing.assert_array_equal(out.delivered, outs["U"].delivered)
        _assert_eligible(out)


def _assert_eligible(out: env.Outcome) -> None:
    _, _, creator, rank = INPUTS
    count = np.zeros(USERS, dtype=np.int64)
    for d in range(DAYS):
        for s in range(SLOTS):
            users = np.flatnonzero(out.delivered[d, s])
            fam = FAMILY[out.template[d, s, users]]
            j = np.minimum(count[users], CANDIDATES - 1)
            assert (creator[users, j][fam == tp.CREATOR] >= 1).all()
            assert (rank[users, j][fam == tp.POPULAR] > 0).all()
            back = users[fam == tp.BACK]
            if len(back):
                assert d >= 2
                assert not out.visited[d - 2 : d, back].any()
            count[users] += 1


def test_without_copy_effects_every_kind_gives_the_same_opens() -> None:
    params = Params(base_rate=0.2, copy_spread=0.0, novelty=0.0)
    opened = [_run(kind, params)[0].opened for kind in tp.KINDS]
    for o in opened[1:]:
        np.testing.assert_array_equal(o, opened[0])


def test_first_day_choices_are_shared() -> None:
    params = Params(base_rate=0.2, copy_spread=0.2)
    first = [_run(kind, params)[0].template[0] for kind in ("U", "RD", "D0", "RAW")]
    for t in first[1:]:
        np.testing.assert_array_equal(t, first[0])


def test_logged_propensities_match_the_choice_probabilities() -> None:
    params = Params(base_rate=0.2, copy_spread=0.2)
    policy = _policy("RD", params)
    out = env.simulate(INPUTS[0], params, policy, env.draws(USERS, 7), WORLD_SEED, K)
    sent = out.delivered
    assert ((out.propensity[sent] > 0) & (out.propensity[sent] <= 1)).all()
    # Uniform on day 0: propensity is one over the number of eligible templates
    u = _run("U", params)[0]
    p0 = u.propensity[0][u.delivered[0]]
    assert set(np.round(1 / p0).astype(int)) <= {30, 40, 50}


def test_uniform_choices_follow_their_probabilities() -> None:
    _, slot, creator, rank = INPUTS
    policy = tp.Templated("U", slot, FAMILY, creator, rank, 1)
    users = np.arange(USERS)
    counts = np.zeros(K)
    expected = np.zeros(K)
    for day in range(40):
        probs = policy.probabilities(users, 0.0)
        chosen = tp.inverse_cdf(probs, tp.choice_uniforms(1, day, 0, USERS))
        counts += np.bincount(chosen, minlength=K)
        expected += probs.sum(axis=0)
    assert counts[expected == 0].sum() == 0
    seen = expected > 0
    z = (counts[seen] - expected[seen]) / np.sqrt(expected[seen])
    assert np.abs(z).max() < 4.5


def test_oracle_picks_the_best_eligible_template() -> None:
    params = Params(base_rate=0.2, copy_spread=0.2, novelty=0.0)
    out, policy = _run("O", params)
    assert policy.copy is not None
    # Families 2, 3 and 5 are always eligible, so no choice is worse than their best
    best_always = policy.copy[np.isin(FAMILY, [2, 3, 5])].max()
    chosen = out.template[out.delivered]
    assert (policy.copy[chosen] >= best_always).all()
    assert (policy.copy[chosen] > best_always).any()


def test_bandit_learns_large_effects() -> None:
    params = Params(base_rate=0.2, copy_spread=0.6, novelty=0.05)
    copy = world.copy_effects(params.copy_spread, WORLD_SEED, K)
    out_rd, rd = _run("RD", params)
    out_u, _ = _run("U", params)
    assert spearmanr(rd.scores, np.log(copy)).statistic > 0.5
    assert out_rd.opened.sum() > out_u.opened.sum()


def test_the_penalty_lowers_recently_seen_templates() -> None:
    _, slot, creator, rank = INPUTS
    users = np.arange(3)
    probs = {}
    for kind in ("RD", "D0"):
        policy = tp.Templated(kind, slot, FAMILY, creator, rank, 1)
        policy._last_seen[0, 5] = 9.0  # template 5 reached user 0 one day ago
        probs[kind] = policy.probabilities(users, 10.0)
    ratio = probs["RD"][0] / probs["RD"][0, 2]
    assert ratio[5] == pytest.approx(np.exp(-tp.PENALTY * 0.5 ** (1 / 15) / tp.TEMPERATURE))
    np.testing.assert_allclose(probs["D0"][0, 5], probs["D0"][0, 2])
    np.testing.assert_allclose(probs["RD"][1:], probs["D0"][1:])


def test_fixed_seeds_give_identical_runs() -> None:
    params = Params(base_rate=0.2, copy_spread=0.2)
    a, b = _run("RD", params)[0], _run("RD", params)[0]
    np.testing.assert_array_equal(a.template, b.template)
    np.testing.assert_array_equal(a.opened, b.opened)


def test_a_template_policy_must_see_every_day() -> None:
    params = Params()
    policy = _policy("RD", params)
    obs = env.Observation(1, 0, np.zeros(USERS, dtype=bool))
    with pytest.raises(ValueError):
        policy.act(obs)


def test_oracle_needs_the_truth() -> None:
    _, slot, creator, rank = INPUTS
    with pytest.raises(ValueError):
        tp.Templated("O", slot, FAMILY, creator, rank, 1)


def test_finite_shrinkage_matches_the_formula() -> None:
    # Two arms, uniform logging; arm 0 opened 30 of 100 times, arm 1 10 of 100
    stats = tp.ArmStats(2)
    chosen = np.repeat([0, 1], 100)
    reward = np.concatenate([np.arange(100) < 30, np.arange(100) < 10]).astype(float)
    stats.add(np.full((200, 2), 0.5), chosen, reward)
    overall, mu_arm = 0.2, np.array([0.2, 0.2])
    n = 100.0  # Kish size of 100 equal weights
    excess = np.mean((np.array([0.3, 0.1]) - 0.2) ** 2) - np.mean(0.2 * 0.8 / n)
    m = overall * (1 - overall) / excess - 1
    plus = (np.array([0.3, 0.1]) * n + mu_arm * m) / (n + m)
    minus = (np.array([0.1, 0.3]) * n + mu_arm * m) / (n + m)
    np.testing.assert_allclose(stats.scores(), (plus - minus) / minus)
    assert m == pytest.approx(0.16 / 0.0084 - 1)


class _Recorder(tp.Templated):
    """Keeps the probabilities each decision was drawn from."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.drawn: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    def act(self, obs: env.Observation) -> env.Action:
        action = super().act(obs)
        if self._pending and self._pending[-1][0] == obs.slot:
            _, users, probs, _ = self._pending[-1]
            self.drawn.append((obs.day, obs.slot, users, probs))
        return action


@pytest.mark.parametrize("kind", ["RD", "D0", "RAW"])
def test_logged_propensity_is_the_drawn_probability(kind: tp.Kind) -> None:
    params = Params(base_rate=0.2, copy_spread=0.2)
    _, slot, creator, rank = INPUTS
    policy = _Recorder(kind, slot, FAMILY, creator, rank, 7)
    out = env.simulate(INPUTS[0], params, policy, env.draws(USERS, 7), WORLD_SEED, K)
    checked = 0
    for day, s, users, probs in policy.drawn:
        k = out.template[day, s, users].astype(np.int64)
        np.testing.assert_allclose(
            out.propensity[day, s, users], probs[np.arange(len(users)), k], rtol=1e-6
        )
        checked += len(users)
    assert checked == out.delivered.sum()
    # After the first day the learned policies do not choose uniformly
    late = [p for d, _, _, p in policy.drawn if d > 5]
    assert np.ptp(np.concatenate(late)[np.concatenate(late) > 0]) > 0


def test_choices_do_not_depend_on_the_same_day_outcomes() -> None:
    params = Params(base_rate=0.2, copy_spread=0.2)
    day = 10
    draws = env.draws(USERS, 7)
    changed = env.Draws(draws.open.copy(), draws.opt_out, draws.visit.copy())
    changed.open[day] = 1 - changed.open[day]
    changed.visit[day] = 1 - changed.visit[day]
    runs = [
        env.simulate(INPUTS[0], params, _policy("RD", params), d, WORLD_SEED, K)
        for d in (draws, changed)
    ]
    assert not np.array_equal(runs[0].opened[day], runs[1].opened[day])
    np.testing.assert_array_equal(runs[0].template[: day + 1], runs[1].template[: day + 1])
