"""Inline SVG line charts with interval whiskers; colors come from the page's CSS tokens."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

WIDTH, HEIGHT = 880, 320
CHART_CSS = """
.chart{width:100%;height:auto;display:block;
font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.chart .grid{stroke:var(--grid,#e1e0d9);stroke-width:1}
.chart .zero{stroke:var(--axis,#c3c2b7);stroke-width:1.5}
.chart .tick{fill:var(--muted,#898781);font-size:12px;font-variant-numeric:tabular-nums}
.chart .axis-title{fill:var(--ink-2,#52514e);font-size:12px}
.chart .line{fill:none;stroke-width:2;stroke-linejoin:round}
.chart .whisker{stroke-width:1.5}
.chart .dot{stroke:var(--surface,#fcfcfb);stroke-width:2}
.chart .direct{fill:var(--ink-2,#52514e);font-size:12px}
.chart .s1 .line,.chart .s1 .whisker{stroke:var(--s1,#2a78d6)}
.chart .s1 .dot{fill:var(--s1,#2a78d6)}
.chart .s2 .line,.chart .s2 .whisker{stroke:var(--s2,#eb6834)}
.chart .s2 .dot{fill:var(--s2,#eb6834)}
.chart .s3 .line,.chart .s3 .whisker{stroke:var(--s3,#1baf7a)}
.chart .s3 .dot{fill:var(--s3,#1baf7a)}
.chart .dot:hover{r:6}
"""

LEFT, RIGHT, TOP, BOTTOM = 64, 150, 16, 48


@dataclass(frozen=True)
class Series:
    name: str
    points: Sequence[dict[str, Any]]  # each with value, ci_low, ci_high


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values covering [low, high], step 1, 2 or 5 times a power of ten."""
    span = high - low or abs(high) or 1.0
    raw = span / count
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw)
    first = math.floor(low / step) * step
    ticks = []
    value = first
    while value <= high + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    if ticks[-1] < high:
        ticks.append(round(ticks[-1] + step, 12))
    return ticks


def _decimals(ticks: Sequence[float]) -> int:
    """Fewest decimals that show every tick exactly."""
    return next(d for d in range(10) if all(abs(round(t, d) - t) < 1e-12 for t in ticks))


def _tick(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return text.replace("-", chr(0x2212)) if value < 0 and float(text) != 0 else text.lstrip("-")


def line_chart(
    xs: Sequence[float],
    series: Sequence[Series],
    x_title: str,
    y_title: str,
    label: str,
    fmt: Any,
    standalone: bool = False,
) -> str:
    """Lines with 95% whiskers, a zero line, direct labels and one tooltip per point.

    ``standalone`` embeds the chart's styles, for use outside the report page.
    """
    lows = [p["ci_low"] for s in series for p in s.points]
    highs = [p["ci_high"] for s in series for p in s.points]
    y_ticks = nice_ticks(min(*lows, 0.0), max(*highs, 0.0))
    x_ticks = list(xs)
    y0, y1 = y_ticks[0], y_ticks[-1]
    x0, x1 = x_ticks[0], x_ticks[-1]
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def px(x: float) -> float:
        return LEFT + (x - x0) / (x1 - x0) * plot_w

    def py(y: float) -> float:
        return TOP + (y1 - y) / (y1 - y0) * plot_h

    yd, xd = _decimals(y_ticks), _decimals(x_ticks)
    out = [
        f'<svg class="chart" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="{escape(label)}">'
    ]
    if standalone:
        out.append(f"<style>{CHART_CSS}</style>")
    for t in y_ticks:
        cls = "zero" if t == 0 else "grid"
        out.append(
            f'<line class="{cls}" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{py(t):.1f}" '
            f'y2="{py(t):.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{LEFT - 8}" y="{py(t) + 4:.1f}" '
            f'text-anchor="end">{_tick(t, yd)}</text>'
        )
    for t in x_ticks:
        out.append(
            f'<text class="tick" x="{px(t):.1f}" y="{TOP + plot_h + 18}" '
            f'text-anchor="middle">{_tick(t, xd)}</text>'
        )
    out.append(
        f'<text class="axis-title" x="{LEFT + plot_w / 2:.1f}" y="{HEIGHT - 8}" '
        f'text-anchor="middle">{escape(x_title)}</text>'
    )
    out.append(
        f'<text class="axis-title" x="14" y="{TOP + plot_h / 2:.1f}" '
        f'text-anchor="middle" transform="rotate(-90 14 {TOP + plot_h / 2:.1f})">'
        f"{escape(y_title)}</text>"
    )
    for i, s in enumerate(series, start=1):
        coords = [(px(x), py(p["value"])) for x, p in zip(xs, s.points, strict=True)]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        out.append(f'<g class="s{i}"><polyline class="line" points="{path}"/>')
        for x, p in zip(xs, s.points, strict=True):
            cx = px(x)
            out.append(
                f'<line class="whisker" x1="{cx:.1f}" x2="{cx:.1f}" '
                f'y1="{py(p["ci_low"]):.1f}" y2="{py(p["ci_high"]):.1f}"/>'
            )
            tip = f"{s.name}, {x_title.lower()} {x:g}: {fmt(p)}"
            out.append(
                f'<circle class="dot" cx="{cx:.1f}" cy="{py(p["value"]):.1f}" r="4">'
                f"<title>{escape(tip)}</title></circle>"
            )
        end_x, end_y = coords[-1]
        out.append(
            f'<text class="direct" x="{end_x + 10:.1f}" y="{end_y + 4:.1f}">'
            f"{escape(s.name)}</text></g>"
        )
    out.append("</svg>")
    return "".join(out)
