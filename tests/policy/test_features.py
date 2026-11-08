import numpy as np

from push_lab.policy import features
from push_lab.policy.features import NAMES, History, Static
from push_lab.sim import env
from push_lab.sim.explore import EpsilonGreedy
from push_lab.sim.world import DAYS, SLOTS, Params

from .conftest import make_world


def _logs(users: int = 300) -> tuple[env.Outcome, Static]:
    pop, static = make_world(seed=1, users=users)
    explorer = EpsilonGreedy(static.profile.argmax(axis=1), seed=2)
    out = env.simulate(pop, Params(base_rate=0.2, opt_out=0.01), explorer, env.draws(users, 3))
    return out, static


def test_features_match_a_direct_computation() -> None:
    out, static = _logs()
    history = History(len(out.opted_out_at))
    col = {name: j for j, name in enumerate(NAMES)}
    for d in range(DAYS):
        f = history.features(static)
        delivered, opened = out.delivered[:d], out.opened[:d]
        assert (f[:, :, col["day"]] == d).all()
        np.testing.assert_array_equal(f[:, 0, col["opens"]], opened.sum(axis=(0, 1)))
        count = delivered.sum(axis=(0, 1))
        np.testing.assert_array_equal(
            f[:, 2, col["next_score"]], static.score[np.arange(len(count)), count]
        )
        if d > 0:
            np.testing.assert_allclose(f[:, 1, col["visit_share"]], out.visited[:d].mean(axis=0))
            np.testing.assert_array_equal(f[:, 3, col["visited_yesterday"]], out.visited[d - 1])
            last = np.where(
                delivered.any(axis=1).any(axis=0),
                d - 1 - np.argmax(delivered.any(axis=1)[::-1], axis=0),
                -np.inf,
            )
            np.testing.assert_array_equal(
                f[:, 0, col["days_since_delivery"]], np.minimum(28.0, d - last)
            )
            # Decayed volume at 17:00 of day d over deliveries of earlier days
            times = np.arange(d)[:, None] + np.array([8, 12, 17, 21])[None, :] / 24
            weights = 0.5 ** ((d + 17 / 24 - times) / features.VOLUME_HALF_LIFE)
            expected = np.einsum("ds,dsu->u", weights, delivered.astype(float))
            np.testing.assert_allclose(f[:, 2, col["volume"]], expected)
        else:
            assert np.isnan(f[:, :, col["visit_share"]]).all()
        best = static.profile.argmax(axis=1)
        for s in range(SLOTS):
            np.testing.assert_array_equal(f[:, s, col["is_best"]], best == s)
        history.end_day(out.delivered[d], out.opened[d], out.visited[d])


def test_logged_rows_skip_opted_out_users_and_use_only_earlier_days() -> None:
    out, static = _logs()
    rows = features.logged_rows(out, static)
    oo = out.opted_out_at
    flat_slot = rows.day * SLOTS + rows.slot
    assert ((oo[rows.user] < 0) | (oo[rows.user] >= flat_slot)).all()
    assert (oo >= 0).any()
    # Changing outcomes of day 10 and later leaves earlier rows' features unchanged
    changed = env.Outcome(
        send=out.send,
        delivered=out.delivered,
        opened=np.where(np.arange(DAYS)[:, None, None] >= 10, True, out.opened),
        template=out.template,
        item=out.item,
        propensity=out.propensity,
        opted_out_at=out.opted_out_at,
        visited=np.where(np.arange(DAYS)[:, None] >= 10, True, out.visited),
    )
    other = features.logged_rows(changed, static)
    upto = rows.day <= 10
    np.testing.assert_array_equal(rows.x[upto], other.x[upto])
