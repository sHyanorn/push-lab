import numpy as np

from push_lab.policy.features import NAMES, Rows
from push_lab.policy.gate import Models


def test_each_model_is_fitted_on_its_own_rows() -> None:
    # Same-day visits 0.8 with a delivery and 0.2 without; opens 0.3 of deliveries (METHODS 12.3)
    rng = np.random.default_rng(0)
    n = 4_000
    treated = rng.random(n) < 0.5
    rows = Rows(
        x=rng.random((n, len(NAMES))),
        treated=treated,
        visited=rng.random(n) < np.where(treated, 0.8, 0.2),
        opened=treated & (rng.random(n) < 0.3),
        user=np.arange(n),
        day=np.zeros(n, dtype=np.int64),
        slot=np.zeros(n, dtype=np.int64),
    )
    models = Models.fit(rows)
    probe = rng.random((500, len(NAMES)))
    assert abs(float(np.mean(models.control.predict(probe))) - 0.2) < 0.05
    assert abs(float(np.mean(models.treated.predict(probe))) - 0.8) < 0.05
    assert abs(float(np.mean(models.predict_open(probe))) - 0.3) < 0.05
