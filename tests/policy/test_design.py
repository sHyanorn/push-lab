import numpy as np

from push_lab.policy import design


def test_settings_cover_the_sweeps_with_distinct_seeds() -> None:
    settings = design.settings()
    sweeps = [s["sweep"] for s in settings]
    assert len(settings) == 263
    assert {k: sweeps.count(k) for k in set(sweeps)} == {
        "main": 64,
        "annoyance_free": 64,
        "stress": 64,
        "grid": 61,
        "refit": 10,
    }
    assert [s["id"] for s in settings] == list(range(263))
    for s in settings:
        ints = [s["world_seed"], s["explore_seed"], s["perturb_seed"], s["diagnostic_seed"]]
        assert len(set(ints + s["eval_seeds"])) == len(ints) + len(s["eval_seeds"])
        for name, (low, high) in design.RANGES.items():
            if s["sweep"] != "stress" or name != "opt_out":
                assert low <= s["values"][name] <= high


def test_variant_sweeps_change_one_parameter_of_the_main_points() -> None:
    settings = design.settings()
    main = [s for s in settings if s["sweep"] == "main"]
    for sweep, name, value in (("annoyance_free", "annoyance", 0.0), ("stress", "opt_out", 0.005)):
        other = [s for s in settings if s["sweep"] == sweep]
        for a, b in zip(main, other, strict=True):
            assert b["values"] == a["values"] | {name: value}
            assert b["world_seed"] == a["world_seed"] and b["eval_seeds"] == a["eval_seeds"]


def test_hypercube_has_one_point_per_stratum() -> None:
    u = design.hypercube()
    for column in u.T:
        np.testing.assert_array_equal(np.sort(np.floor(column * 64)), np.arange(64))
    np.testing.assert_array_equal(u, design.hypercube())


def test_grid_has_five_points_per_parameter_around_the_reference() -> None:
    grid = [s for s in design.settings() if s["sweep"] == "grid"]
    for name in design.NAMES:
        members = sorted((g[1], s) for s in grid for g in s["grid"] if g[0] == name)
        assert [i for i, _ in members] == list(range(5))
        for _, s in members:
            others = {k: v for k, v in s["values"].items() if k != name}
            assert others == {k: v for k, v in design.REFERENCE.items() if k != name}
    seeds = {(s["world_seed"], s["explore_seed"]) for s in grid}
    assert len(seeds) == 1


def test_users_are_disjoint_and_reproducible() -> None:
    a, b = design.users(27_285), design.users(27_285)
    assert a == b
    sets = [set(v) for v in a.values()]
    assert sum(map(len, sets)) == len(set().union(*sets)) == 17_000
