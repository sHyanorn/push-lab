"""Partial reproduction for the manual ``reproduce`` workflow (D050).

``synthetic`` writes fixtures into ``data/processed``, runs the pipeline steps that work on
them and checks each result file against the top-level keys of the committed one; deeper
keys depend on the data (tabs, features, the full OBD campaign). ``sample`` reruns
steps on downloaded KuaiRand-Pure, Hillstrom and the OBD small sample and compares their
numbers with the committed results, ignoring timings and commits. Both overwrite files in
``results/``, so run them in a fresh checkout.
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from push_lab.data import fixtures, paths

FIXTURE_ROWS = 4_000
# Steps that run on fixtures; data.quality reports the fixtures' row counts as errors
SYNTHETIC = (
    ("data.quality",),
    ("rank.evaluate",),
    ("uplift.evaluate",),
    ("sim.build",),
    ("ope.obd",),
    ("ope.simulator", "--smoke"),
    ("ope.carryover", "--smoke"),
    ("report.evidence",),
)
SYNTHETIC_FILES = {
    "data_quality.json": "data_quality.json",
    "ranking.json": "ranking.json",
    "uplift.json": "uplift.json",
    "simulator.json": "simulator.json",
    "ope_obd.json": "ope_obd.json",
    "runs/smoke/ope/ope_simulator_amended.json": "ope_simulator_amended.json",
    "runs/smoke/ope_carryover/ope_carryover.json": "ope_carryover.json",
    "evidence.json": "evidence.json",
}
# Keys that other runs add: --reuse-logs and --post-hoc of ope.carryover
OPTIONAL = {"ope_carryover.json": {"logs_commit", "post_hoc"}}
# A main-sweep setting of the frozen policy design
SAMPLE_SETTING = 0
IGNORED = ("timings_s", "commit", "logs_commit")


def committed(root: Path, name: str) -> Any:
    """A results file as committed at HEAD."""
    out = subprocess.run(
        ["git", "show", f"HEAD:{paths.RESULTS.as_posix()}/{name}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    return json.loads(out.stdout)


def current(root: Path, name: str) -> Any:
    return json.loads((root / paths.RESULTS / name).read_text("utf-8"))


def step(root: Path, module: str, *args: str, allowed: Sequence[int] = (0,)) -> None:
    print(f"== {module} {' '.join(args)}", flush=True)
    code = subprocess.run(
        [sys.executable, "-m", f"push_lab.{module}", "--root", str(root), *args], cwd=root
    ).returncode
    if code not in allowed:
        raise SystemExit(f"{module} exited with {code}")


def missing_keys(fresh: dict[str, Any], reference: dict[str, Any], optional: set[str]) -> list[str]:
    return sorted(set(reference) - set(fresh) - optional)


def differences(fresh: Any, reference: Any, path: str = "") -> Iterator[str]:
    """Paths where two results differ, skipping timings and commits."""
    if isinstance(reference, dict) and isinstance(fresh, dict):
        for k in sorted(set(reference) | set(fresh)):
            if k in IGNORED or k.endswith(("_s", "_ms")) or "_fit_s_" in k or "_ms_" in k:
                continue
            if k not in fresh or k not in reference:
                yield f"{path}.{k}: only in {'committed' if k in reference else 'rerun'}"
            else:
                yield from differences(fresh[k], reference[k], f"{path}.{k}")
    elif isinstance(reference, list) and isinstance(fresh, list) and len(reference) == len(fresh):
        for i, (a, b) in enumerate(zip(fresh, reference, strict=True)):
            yield from differences(a, b, f"{path}[{i}]")
    elif fresh != reference:
        yield f"{path}: rerun {fresh!r}, committed {reference!r}"


def synthetic(root: Path) -> int:
    fixtures.write_processed(root, n=FIXTURE_ROWS)
    for module, *args in SYNTHETIC:
        step(root, module, *args, allowed=(0, 1) if module == "data.quality" else (0,))
    failures = 0
    for fresh_name, reference_name in SYNTHETIC_FILES.items():
        optional = OPTIONAL.get(reference_name, set())
        gaps = missing_keys(current(root, fresh_name), committed(root, reference_name), optional)
        print(f"{fresh_name}: {'keys match' if not gaps else f'{len(gaps)} keys missing'}")
        for gap in gaps:
            print(f"  missing {gap}")
        failures += bool(gaps)
    return 1 if failures else 0


def hillstrom_part(uplift: dict[str, Any]) -> dict[str, Any]:
    primary = {
        k: {f: v for f, v in entry.items() if not f.startswith(("holm", "p_value_res"))}
        for k, entry in uplift["primary"].items()
        if "hillstrom" in k
    }
    return {"hillstrom": uplift["hillstrom"], "primary": primary}


def sample(root: Path, setting: int) -> int:
    from push_lab.policy import compare

    step(root, "rank.evaluate")
    step(root, "uplift.evaluate", "--only", "hillstrom")
    step(root, "sim.build")
    step(root, "policy.sweep", "--ids", str(setting), "--no-compare")
    rerun_setting = next(s for s in compare.run(root)["settings"] if s["id"] == setting)
    committed_setting = next(
        s for s in committed(root, "policy.json")["settings"] if s["id"] == setting
    )
    checks = {
        "ranking.json": (current(root, "ranking.json"), committed(root, "ranking.json")),
        "uplift.json (Hillstrom)": (
            hillstrom_part(current(root, "runs/uplift_hillstrom.json")),
            hillstrom_part(committed(root, "uplift.json")),
        ),
        "simulator.json": (current(root, "simulator.json"), committed(root, "simulator.json")),
        f"policy.json (setting {setting})": (rerun_setting, committed_setting),
    }
    failures = 0
    for name, (fresh, reference) in checks.items():
        found = list(differences(fresh, reference))
        print(f"{name}: {'identical' if not found else f'{len(found)} differences'}")
        for line in found[:20]:
            print(f"  {line}")
        failures += bool(found)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rerun part of the pipeline and check it.")
    parser.add_argument("mode", choices=("synthetic", "sample"))
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--setting", type=int, default=SAMPLE_SETTING, help="sample mode only")
    args = parser.parse_args(argv)
    if args.mode == "synthetic":
        return synthetic(args.root)
    return sample(args.root, args.setting)


if __name__ == "__main__":
    sys.exit(main())
