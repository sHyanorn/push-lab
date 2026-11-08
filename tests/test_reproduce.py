from push_lab import reproduce


def test_differences_skip_timings_and_commits() -> None:
    reference = {"a": 1.0, "commit": "x", "timings_s": {"fit": 2.0}, "fit_s": 3.0, "b": [1, 2]}
    fresh = {"a": 1.0, "commit": "y", "timings_s": {"fit": 9.0}, "fit_s": 4.0, "b": [1, 2]}
    assert list(reproduce.differences(fresh, reference)) == []
    fresh["b"] = [1, 3]
    assert list(reproduce.differences(fresh, reference)) == [".b[1]: rerun 3, committed 2"]


def test_missing_keys_are_top_level_and_skip_optional_ones() -> None:
    reference = {"settings": [{"id": 0}], "label": "a", "post_hoc": {}}
    fresh: dict[str, object] = {"settings": []}
    assert reproduce.missing_keys(fresh, reference, {"post_hoc"}) == ["label"]


def test_hillstrom_part_drops_criteo_and_holm() -> None:
    uplift = {
        "hillstrom": {"visit": 1},
        "criteo": {"visit": 2},
        "primary": {
            "U1_hillstrom": {"value": 1, "holm_rejected": True, "p_value_resolution": 0.002},
            "U3_criteo": {"value": 2},
        },
    }
    assert reproduce.hillstrom_part(uplift) == {
        "hillstrom": {"visit": 1},
        "primary": {"U1_hillstrom": {"value": 1}},
    }
