"""Writes docs/overview.svg (the framework) and docs/results.svg (test results on the agent
repairs, from artifacts/candidates/test/scores.json). Same style as docs/pipeline.svg."""

import json
from pathlib import Path

FONT = "Segoe UI, Helvetica, Arial, sans-serif"
INK, MUTED, LINE, FILL = "#1d1a26", "#5e5870", "#3d3a8c", "#eeecf8"
ACCENT, ACCENT_FILL = "#b8325f", "#f8e6ed"
GRID = "#e6e4ee"


def box(x, y, w, h, title, lines=(), fill=FILL, stroke=LINE, dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    cx = x + w / 2
    top = y + (h - (18 + 17 * len(lines))) / 2 + 14
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"{dash}/>',
        f'<text x="{cx}" y="{top:.0f}" text-anchor="middle" font-size="14" '
        f'font-weight="600" fill="{INK}">{title}</text>',
    ]
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{cx}" y="{top + 19 + 17 * i:.0f}" text-anchor="middle" '
            f'font-size="12" fill="{MUTED}">{line}</text>'
        )
    return "\n  ".join(out)


def overview() -> str:
    parts = [
        box(12, 86, 150, 78, "Repair task", ("statement and a", "faulty project")),
        box(190, 86, 160, 78, "LLM agent", ("edits the code, runs", "short trainings")),
        box(378, 86, 170, 78, "Harness run", ("stopped and restarted;", "records what trained")),
        box(
            576,
            86,
            196,
            78,
            "RunSleuth v2",
            ("judges each requirement:", "supported, violated, untested"),
            fill=ACCENT_FILL,
            stroke=ACCENT,
        ),
        box(
            806, 42, 182, 44, "Accept", ("every part supported",), fill="#e7f3ec", stroke="#2f7d4f"
        ),
        box(806, 103, 182, 44, "Review", ("some part untested",), fill="#fbf1dc", stroke="#a86b00"),
        box(806, 164, 182, 44, "Reject", ("a part violated",), fill=ACCENT_FILL, stroke=ACCENT),
        box(576, 206, 196, 50, "Probe runs", ("chosen by the evidence gap",)),
        box(
            190,
            206,
            358,
            50,
            "Hidden behavioural tests",
            ("label every candidate; no verifier sees them",),
            fill="#ffffff",
            dashed=True,
        ),
    ]
    arrows = [
        "M162 125 H187",
        "M350 125 H375",
        "M548 125 H573",
        "M772 125 H788 V64 H803",
        "M772 125 H803",
        "M772 125 H788 V186 H803",
        "M652 164 V203",
        "M696 206 V167",
    ]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 272" width="1000" height="272" role="img" aria-labelledby="title desc" font-family="{FONT}">
  <title id="title">Verifying an LLM agent's repair</title>
  <desc id="desc">An LLM agent repairs a faulty training project. The harness runs the candidate, stopping and restarting it and recording what trained. RunSleuth v2 judges each requirement as supported, violated or untested from the run's records and from probe runs chosen by what is still untested, then accepts, sends to review or rejects. Hidden behavioural tests label every candidate and are never seen by a verifier.</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="1000" height="272" rx="10" fill="#ffffff"/>
  {chr(10).join("  " + p for p in parts).strip()}
  <g fill="none" stroke="{LINE}" stroke-width="1.5" marker-end="url(#arrow)">
    {chr(10).join(f'    <path d="{d}"/>' for d in arrows).strip()}
  </g>
  <path d="M270 164 V203" fill="none" stroke="{LINE}" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#arrow)"/>
</svg>
"""


ROWS = (
    ("metric_gate", "Metric gate"),
    ("stage_one_gate", "Stage-one gate"),
    ("runsleuth_v1", "RunSleuth v1"),
    ("llm_reviewer/gemini-4-argon", "Gemini 4 Argon"),
    ("llm_reviewer/claude-sonnet-5.5", "Claude Sonnet 5.5"),
    ("runsleuth_v2", "RunSleuth v2"),
)
PANELS = (
    ("false_acceptance", "Wrong repairs accepted"),
    ("false_rejection", "Correct repairs rejected"),
    ("review_share", "Sent to review"),
)


def results(scores: dict) -> str:
    label_right, panel_x, plot_w, gap = 178, 200, 220, 50
    top, row_h, n = 58, 30, len(ROWS)
    axis_y = top + row_h * n + 12
    height = axis_y + 34
    out = []
    v2_row = [i for i, (name, _) in enumerate(ROWS) if name == "runsleuth_v2"][0]
    y_band = top + row_h * v2_row
    out.append(
        f'<rect x="12" y="{y_band}" width="976" height="{row_h}" rx="6" fill="{ACCENT_FILL}" opacity="0.6"/>'
    )
    for i, (_, label) in enumerate(ROWS):
        weight = ' font-weight="600"' if i == v2_row else ""
        out.append(
            f'<text x="{label_right}" y="{top + row_h * i + 19}" text-anchor="end" '
            f'font-size="13" fill="{INK}"{weight}>{label}</text>'
        )
    for p, (key, title) in enumerate(PANELS):
        x0 = panel_x + p * (plot_w + gap)
        denominator = scores["metric_gate"]["groups"]["agent"][key]["of"]
        out.append(
            f'<text x="{x0}" y="30" font-size="13" font-weight="600" fill="{INK}">{title}</text>'
        )
        out.append(
            f'<text x="{x0}" y="46" font-size="12" fill="{MUTED}">of {denominator} agent repairs</text>'
        )
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            x = x0 + plot_w * t
            out.append(
                f'<line x1="{x}" y1="{top}" x2="{x}" y2="{axis_y}" stroke="{GRID}" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{x}" y="{axis_y + 18}" text-anchor="middle" font-size="11" '
                f'fill="{MUTED}">{int(t * 100)}%</text>'
            )
        for i, (name, _) in enumerate(ROWS):
            group = scores[name]["groups"]["agent"]
            rate = group[key]["count"] / group[key]["of"]
            interval = group["intervals"][key]
            colour = ACCENT if name == "runsleuth_v2" else LINE
            y = top + row_h * i + 15
            lo, hi = x0 + plot_w * interval["low"], x0 + plot_w * interval["high"]
            out.append(
                f'<line x1="{lo:.1f}" y1="{y}" x2="{hi:.1f}" y2="{y}" stroke="{colour}" '
                f'stroke-width="2" stroke-linecap="round"/>'
            )
            out.append(
                f'<circle cx="{x0 + plot_w * rate:.1f}" cy="{y}" r="5" fill="{colour}" '
                f'stroke="#ffffff" stroke-width="2"/>'
            )
    body = "\n  ".join(out)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" width="1000" height="{height}" role="img" aria-labelledby="title desc" font-family="{FONT}">
  <title id="title">Test results on the agent repairs</title>
  <desc id="desc">For each verifier, the share of the 16 wrong agent repairs it accepted, of the 164 correct ones it rejected, and of the 180 it sent to review, each with a 95% interval from a bootstrap over the 12 starting versions.</desc>
  <rect x="0" y="0" width="1000" height="{height}" rx="10" fill="#ffffff"/>
  {body}
</svg>
"""


if __name__ == "__main__":
    scores = json.loads(Path("artifacts/candidates/test/scores.json").read_text(encoding="utf-8"))[
        "verifiers"
    ]
    Path("docs/overview.svg").write_text(overview(), encoding="utf-8", newline="\n")
    Path("docs/results.svg").write_text(results(scores), encoding="utf-8", newline="\n")
    print("ok")
