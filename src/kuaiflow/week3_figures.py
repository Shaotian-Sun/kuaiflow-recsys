"""Generate website-ready Week 3 workflow and experiment-result figures.

The SVGs use only the Python standard library and remain editable.  When
``rsvg-convert`` is available, matching PNG copies are rendered automatically.
Metrics are read from the persisted experiment artifacts instead of duplicated
as hand-maintained chart constants.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


NAVY = "#0B1736"
MUTED = "#526078"
GRID = "#DCE3EC"
BLUE = "#1769B0"
GREEN = "#148A55"
PURPLE = "#6A4CAF"
ORANGE = "#ED9700"
TEAL = "#087F8C"
RED = "#E53935"
PALE_BLUE = "#EBF4FF"
PALE_GREEN = "#EAF8F0"
PALE_PURPLE = "#F2EEFB"
PALE_ORANGE = "#FFF6E3"
PALE_TEAL = "#E9F7F8"


class SVG:
    """Small SVG builder with consistent typography and escaping."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        title: str,
        description: str,
    ) -> None:
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="figure-title figure-description">',
            f'<title id="figure-title">{escape(title)}</title>',
            f'<desc id="figure-description">{escape(description)}</desc>',
            """<defs>
              <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
                <feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#0B1736" flood-opacity="0.13"/>
              </filter>
              <filter id="softShadow" x="-20%" y="-20%" width="140%" height="150%">
                <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0B1736" flood-opacity="0.10"/>
              </filter>
              <linearGradient id="milestone" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0" stop-color="#0B1736"/>
                <stop offset="0.55" stop-color="#173F73"/>
                <stop offset="1" stop-color="#087F8C"/>
              </linearGradient>
              <linearGradient id="blueHeader" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#0D57A1"/><stop offset="1" stop-color="#2582D3"/>
              </linearGradient>
              <linearGradient id="greenHeader" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#08743F"/><stop offset="1" stop-color="#28A56B"/>
              </linearGradient>
              <linearGradient id="purpleHeader" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#56369D"/><stop offset="1" stop-color="#8468C9"/>
              </linearGradient>
              <linearGradient id="orangeHeader" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#D87D00"/><stop offset="1" stop-color="#F5AA19"/>
              </linearGradient>
              <linearGradient id="tealHeader" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#046B76"/><stop offset="1" stop-color="#16A0A8"/>
              </linearGradient>
              <linearGradient id="barBlue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stop-color="#2E85D0"/><stop offset="1" stop-color="#0758A3"/>
              </linearGradient>
              <linearGradient id="barOrange" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stop-color="#FDB31A"/><stop offset="1" stop-color="#E08200"/>
              </linearGradient>
              <linearGradient id="barTeal" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stop-color="#24A8AE"/><stop offset="1" stop-color="#05717F"/>
              </linearGradient>
              <linearGradient id="barPurple" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stop-color="#9B7BD0"/><stop offset="1" stop-color="#6242A9"/>
              </linearGradient>
              <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
                      markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#111827"/>
              </marker>
              <style>
                text { font-family: Inter, Arial, Helvetica, sans-serif; }
              </style>
            </defs>""",
        ]

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "white",
        stroke: str = "none",
        stroke_width: float = 1,
        radius: float = 12,
        opacity: float = 1,
        shadow: bool = False,
    ) -> None:
        effect = ' filter="url(#shadow)"' if shadow else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}" opacity="{opacity}"{effect}/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = NAVY,
        width: float = 2,
        dash: str | None = None,
        arrow: bool = False,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        arrow_attr = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{width}"{dash_attr}{arrow_attr}/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1,
    ) -> None:
        self.add(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def ellipse(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1,
    ) -> None:
        self.add(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1,
    ) -> None:
        encoded = " ".join(f"{x},{y}" for x, y in points)
        self.add(
            f'<polygon points="{encoded}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 18,
        color: str = NAVY,
        weight: int | str = 400,
        anchor: str = "start",
        style: str = "normal",
        letter_spacing: float = 0,
    ) -> None:
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'font-style="{style}" letter-spacing="{letter_spacing}">'
            f'{escape(value)}</text>'
        )

    def lines(
        self,
        x: float,
        y: float,
        values: list[str],
        *,
        size: float = 16,
        line_height: float = 24,
        color: str = NAVY,
        weight: int | str = 400,
        anchor: str = "start",
    ) -> None:
        for index, value in enumerate(values):
            self.text(
                x,
                y + index * line_height,
                value,
                size=size,
                color=color,
                weight=weight,
                anchor=anchor,
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def _section_label(svg: SVG, y: int, label: str, color: str) -> None:
    svg.rect(46, y - 23, 205, 34, fill=color, radius=17)
    svg.text(148, y, label, size=16, color="white", weight=800, anchor="middle")
    svg.line(267, y - 6, svg.width - 46, y - 6, stroke="#AEB9C8", width=1.5, dash="7 7")


def _card(
    svg: SVG,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    header_fill: str,
    body_fill: str,
    border: str,
) -> None:
    svg.rect(x, y, width, height, fill=body_fill, stroke=border, stroke_width=1.5, radius=13, shadow=True)
    svg.add(
        f'<path d="M {x + 13} {y} H {x + width - 13} Q {x + width} {y} '
        f'{x + width} {y + 13} V {y + 58} H {x} V {y + 13} '
        f'Q {x} {y} {x + 13} {y} Z" fill="{header_fill}"/>'
    )
    svg.text(x + width / 2, y + 37, title, size=20, color="white", weight=800, anchor="middle")


def _subbox(
    svg: SVG,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    lines: list[str],
    color: str,
) -> None:
    svg.rect(x, y, width, height, fill="white", stroke=color, stroke_width=1.2, radius=9)
    svg.text(x + 14, y + 25, title, size=15, color=color, weight=800)
    svg.lines(x + 14, y + 49, lines, size=13.5, line_height=20, color=NAVY)


def _flow_arrow(svg: SVG, left: int, right: int, y: int) -> None:
    svg.line(left, y, right, y, stroke="#111827", width=2.2, arrow=True)


def _icon_database(svg: SVG, cx: float, cy: float, color: str, scale: float = 1) -> None:
    width, depth, height = 68 * scale, 15 * scale, 62 * scale
    svg.rect(cx - width / 2, cy - height / 2, width, height, fill=color, radius=3)
    svg.ellipse(cx, cy - height / 2, width / 2, depth, fill="#FFFFFF", stroke=color, stroke_width=3)
    for offset in (-9, 11, 31):
        y = cy - height / 2 + offset * scale
        svg.add(
            f'<path d="M {cx - width / 2} {y} Q {cx} {y + 16 * scale} '
            f'{cx + width / 2} {y}" fill="none" stroke="#FFFFFF" '
            f'stroke-width="{3 * scale}" opacity="0.92"/>'
        )


def _icon_user(svg: SVG, cx: float, cy: float, color: str, scale: float = 1) -> None:
    svg.circle(cx, cy - 22 * scale, 14 * scale, fill=color)
    svg.add(
        f'<path d="M {cx - 28 * scale} {cy + 28 * scale} '
        f'Q {cx - 25 * scale} {cy - 1 * scale} {cx} {cy - 1 * scale} '
        f'Q {cx + 25 * scale} {cy - 1 * scale} {cx + 28 * scale} {cy + 28 * scale} Z" '
        f'fill="{color}"/>'
    )


def _icon_video(svg: SVG, cx: float, cy: float, color: str, scale: float = 1) -> None:
    svg.rect(cx - 36 * scale, cy - 28 * scale, 72 * scale, 56 * scale, fill="white", stroke=color, stroke_width=3, radius=8)
    svg.polygon(
        [
            (cx - 8 * scale, cy - 14 * scale),
            (cx - 8 * scale, cy + 14 * scale),
            (cx + 17 * scale, cy),
        ],
        fill=color,
    )


def _icon_shield(svg: SVG, cx: float, cy: float, color: str, scale: float = 1) -> None:
    svg.add(
        f'<path d="M {cx} {cy - 34 * scale} L {cx + 28 * scale} {cy - 22 * scale} '
        f'V {cy + 2 * scale} Q {cx + 25 * scale} {cy + 28 * scale} {cx} {cy + 39 * scale} '
        f'Q {cx - 25 * scale} {cy + 28 * scale} {cx - 28 * scale} {cy + 2 * scale} '
        f'V {cy - 22 * scale} Z" fill="white" stroke="{color}" stroke-width="{3 * scale}"/>'
    )
    svg.add(
        f'<path d="M {cx - 13 * scale} {cy + 1 * scale} L {cx - 3 * scale} {cy + 12 * scale} '
        f'L {cx + 16 * scale} {cy - 10 * scale}" fill="none" stroke="{color}" '
        f'stroke-width="{5 * scale}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def _icon_target(svg: SVG, cx: float, cy: float, color: str, scale: float = 1) -> None:
    for radius in (31, 20, 9):
        svg.circle(cx, cy, radius * scale, fill="white", stroke=color, stroke_width=3)
    svg.line(cx + 3 * scale, cy - 3 * scale, cx + 37 * scale, cy - 37 * scale, stroke=color, width=4)
    svg.polygon(
        [
            (cx + 37 * scale, cy - 37 * scale),
            (cx + 24 * scale, cy - 34 * scale),
            (cx + 34 * scale, cy - 24 * scale),
        ],
        fill=color,
    )


def _icon_trophy(svg: SVG, cx: float, cy: float, color: str, scale: float = 1) -> None:
    svg.add(
        f'<path d="M {cx - 25 * scale} {cy - 32 * scale} H {cx + 25 * scale} '
        f'V {cy - 10 * scale} Q {cx + 20 * scale} {cy + 15 * scale} {cx} {cy + 18 * scale} '
        f'Q {cx - 20 * scale} {cy + 15 * scale} {cx - 25 * scale} {cy - 10 * scale} Z" '
        f'fill="{color}"/>'
    )
    svg.add(
        f'<path d="M {cx - 25 * scale} {cy - 22 * scale} H {cx - 42 * scale} '
        f'Q {cx - 40 * scale} {cy + 2 * scale} {cx - 20 * scale} {cy + 4 * scale} '
        f'M {cx + 25 * scale} {cy - 22 * scale} H {cx + 42 * scale} '
        f'Q {cx + 40 * scale} {cy + 2 * scale} {cx + 20 * scale} {cy + 4 * scale}" '
        f'fill="none" stroke="{color}" stroke-width="{6 * scale}"/>'
    )
    svg.rect(cx - 5 * scale, cy + 16 * scale, 10 * scale, 20 * scale, fill=color, radius=2)
    svg.rect(cx - 25 * scale, cy + 34 * scale, 50 * scale, 9 * scale, fill=color, radius=4)


def _icon_ranked_list(svg: SVG, cx: float, cy: float, color: str, scale: float = 1) -> None:
    for index, width in enumerate((84, 72, 60), start=1):
        y = cy - 43 * scale + (index - 1) * 36 * scale
        svg.rect(cx - width * scale / 2, y, width * scale, 28 * scale, fill="white", stroke=color, stroke_width=2, radius=7)
        svg.circle(cx - width * scale / 2 + 15 * scale, y + 14 * scale, 10 * scale, fill=color)
        svg.text(cx - width * scale / 2 + 15 * scale, y + 19 * scale, str(index), size=11 * scale, color="white", weight=900, anchor="middle")
        svg.rect(cx - width * scale / 2 + 32 * scale, y + 9 * scale, (width - 41) * scale, 4 * scale, fill=color, radius=2)
        svg.rect(cx - width * scale / 2 + 32 * scale, y + 17 * scale, (width - 50) * scale, 3 * scale, fill="#9EB2C8", radius=1.5)


def _icon_sliders(svg: SVG, cx: float, cy: float, color: str, scale: float = 1) -> None:
    positions = [(-31, -13), (0, 15), (31, -4)]
    for x_offset, knob_offset in positions:
        x = cx + x_offset * scale
        svg.line(x, cy - 37 * scale, x, cy + 37 * scale, stroke=color, width=4)
        svg.circle(x, cy + knob_offset * scale, 9 * scale, fill="white", stroke=color, stroke_width=3)


def _icon_grid(svg: SVG, cx: float, cy: float, color: str, scale: float = 1) -> None:
    for row in range(2):
        for column in range(3):
            x = cx - 42 * scale + column * 42 * scale
            y = cy - 30 * scale + row * 47 * scale
            svg.rect(x - 15 * scale, y - 12 * scale, 30 * scale, 24 * scale, fill="white", stroke=color, stroke_width=2, radius=5)
            svg.polygon(
                [(x - 5 * scale, y - 7 * scale), (x - 5 * scale, y + 7 * scale), (x + 7 * scale, y)],
                fill=color,
            )


def body_fill_for(color: str) -> str:
    return {
        BLUE: PALE_BLUE,
        GREEN: PALE_GREEN,
        PURPLE: PALE_PURPLE,
        ORANGE: PALE_ORANGE,
        TEAL: PALE_TEAL,
    }.get(color, "#F3F5F8")


def build_workflow(results: dict[str, Any], output: Path) -> None:
    """Build the visual-first Week 3 training and serving workflow."""

    svg = SVG(
        1660,
        948,
        title="KuaiFlow Week 3 DeepFM plus MMoE workflow",
        description=(
            "Six illustrated offline stages lead into a five-step candidate-ranking "
            "path that reorders, but does not replace, the Week 2 top 100."
        ),
    )
    train_rows = int(results["training_examples"])
    best_epoch = int(results["optimization"]["best_epoch"])
    svg.rect(0, 0, 1660, 948, fill="#FCFCFD", radius=0)
    svg.text(
        830,
        47,
        "KuaiFlow Week 3 — DeepFM + MMoE Multi-Task Ranking",
        size=38,
        weight=900,
        anchor="middle",
    )
    svg.text(
        830,
        77,
        "One measured path from chronological logs to a multi-objective top-100 order",
        size=17,
        color=MUTED,
        anchor="middle",
    )

    _section_label(svg, 111, "OFFLINE TRAINING", BLUE)
    panel_y, panel_h = 134, 386
    panels = [
        (46, 226, "1. Time-split data", "url(#blueHeader)", BLUE),
        (288, 230, "2. Safe inputs", "url(#greenHeader)", GREEN),
        (534, 250, "3. DeepFM backbone", "url(#purpleHeader)", PURPLE),
        (800, 268, "4. MMoE routing", "url(#orangeHeader)", ORANGE),
        (1084, 250, "5. Ten objectives", "url(#tealHeader)", TEAL),
        (1350, 264, "6. Train + verify", "url(#greenHeader)", GREEN),
    ]
    for x, width, title, header, border in panels:
        _card(svg, x, panel_y, width, panel_h, title, header, "white", border)

    # 1. Chronological data: database plus an explicit train/validation/test timeline.
    _icon_database(svg, 159, 235, BLUE, 0.78)
    svg.text(159, 291, f"{train_rows / 1_000_000:.2f}M train rows", size=15, color=BLUE, weight=900, anchor="middle")
    svg.line(83, 340, 235, 340, stroke="#9AB8DA", width=7)
    split_points = [(88, "TRAIN", BLUE), (166, "VAL", TEAL), (230, "TEST", PURPLE)]
    for x, label, color in split_points:
        svg.circle(x, 340, 13, fill="white", stroke=color, stroke_width=4)
        svg.text(x, 372, label, size=12.5, color=color, weight=900, anchor="middle")
    svg.text(159, 413, "past  →  future", size=16, color=NAVY, weight=800, anchor="middle")
    svg.text(159, 443, "strict chronological split", size=13.5, color=MUTED, anchor="middle")
    svg.rect(77, 469, 164, 30, fill=PALE_BLUE, stroke=BLUE, radius=15)
    svg.text(159, 490, "candidates ≠ labels", size=13, color=BLUE, weight=900, anchor="middle")

    # 2. Inputs: recognizable user/video/context shapes protected by a leakage shield.
    _icon_user(svg, 350, 234, GREEN, 0.72)
    svg.text(403, 244, "+", size=31, color="#93A4B6", weight=500, anchor="middle")
    _icon_video(svg, 460, 234, GREEN, 0.72)
    chip_data = [(318, 301, "USER"), (403, 301, "VIDEO"), (361, 341, "CONTEXT"), (446, 341, "META")]
    for x, y, label in chip_data:
        svg.rect(x - 37, y - 15, 74, 30, fill=PALE_GREEN, stroke=GREEN, radius=15)
        svg.text(x, y + 5, label, size=11.5, color=GREEN, weight=900, anchor="middle")
    _icon_shield(svg, 350, 421, GREEN, 0.5)
    svg.lines(397, 402, ["train-fit only", "10 request-time fields"], size=13.5, line_height=23, color=NAVY, weight=800)
    svg.rect(316, 469, 174, 30, fill="#FFF2F1", stroke=RED, radius=15)
    svg.text(403, 490, "no outcomes · no score", size=12.5, color=RED, weight=900, anchor="middle")

    # 3. DeepFM: feature chips become embeddings, then take linear and FM paths.
    for index, color in enumerate((BLUE, GREEN, ORANGE, TEAL, PURPLE)):
        svg.rect(557 + index * 39, 205, 28, 24, fill=body_fill_for(color), stroke=color, radius=5)
    for row in range(3):
        for column in range(5):
            color = (BLUE, GREEN, ORANGE, TEAL, PURPLE)[column]
            svg.circle(571 + column * 39, 267 + row * 25, 6, fill=color)
    svg.text(659, 337, "shared embeddings", size=13.5, color=PURPLE, weight=900, anchor="middle")
    svg.line(659, 348, 601, 381, stroke=PURPLE, width=2)
    svg.line(659, 348, 717, 381, stroke=PURPLE, width=2)
    svg.circle(600, 401, 29, fill=PALE_PURPLE, stroke=PURPLE, stroke_width=2)
    svg.text(600, 408, "Σwx", size=15, color=PURPLE, weight=900, anchor="middle")
    pair_nodes = [(699, 384), (727, 384), (699, 417), (727, 417)]
    for x, y in pair_nodes:
        svg.circle(x, y, 6, fill=PURPLE)
    for first, second in ((0, 1), (0, 2), (1, 3), (2, 3), (0, 3)):
        svg.line(*pair_nodes[first], *pair_nodes[second], stroke="#9A83CF", width=1.5)
    svg.text(713, 448, "FM pairs", size=13, color=PURPLE, weight=900, anchor="middle")
    svg.line(600, 434, 639, 470, stroke=PURPLE, width=2)
    svg.line(713, 454, 679, 470, stroke=PURPLE, width=2)
    svg.circle(659, 480, 18, fill=PURPLE)
    svg.text(659, 487, "+", size=20, color="white", weight=900, anchor="middle")

    # 4. Four expert stacks route through task gates into ten distinct heads.
    svg.rect(829, 196, 210, 31, fill=PALE_PURPLE, stroke=PURPLE, radius=16)
    svg.text(934, 217, "shared DeepFM vector", size=12.5, color=PURPLE, weight=900, anchor="middle")
    expert_centers = [838, 902, 966, 1030]
    for index, x in enumerate(expert_centers, start=1):
        for layer, width in enumerate((44, 34, 24)):
            svg.rect(x - width / 2, 250 + layer * 23, width, 15, fill="#E7DFF8", stroke=PURPLE, radius=4)
        svg.text(x, 337, f"E{index}", size=12, color=PURPLE, weight=900, anchor="middle")
        svg.line(934, 229, x, 247, stroke="#B19DDA", width=1.4)
    svg.rect(824, 355, 220, 32, fill="#FFF1D3", stroke=ORANGE, radius=16)
    svg.text(934, 377, "10 softmax gates", size=13, color="#9A6200", weight=900, anchor="middle")
    for x in expert_centers:
        svg.line(x, 341, 934, 355, stroke=ORANGE, width=1.3)
    task_labels = ["click", "like", "follow", "comment", "forward", "long", "profile", "hate", "time", "finish"]
    for index, label in enumerate(task_labels):
        column, row = index % 5, index // 5
        x, y = 824 + column * 45, 413 + row * 43
        color = RED if label == "hate" else ORANGE if label in {"time", "finish"} else TEAL
        svg.circle(x + 19, y, 14, fill=body_fill_for(color), stroke=color, stroke_width=1.5)
        svg.text(x + 19, y + 4, str(index + 1), size=9.5, color=color, weight=900, anchor="middle")
        svg.text(x + 19, y + 25, label, size=9.5, color=color, weight=800, anchor="middle")
    svg.text(934, 501, "share  +  specialize", size=13.5, color=PURPLE, weight=900, anchor="middle")

    # 5. Objective family as pictorial badges and a balanced-loss scale.
    _icon_target(svg, 1141, 238, TEAL, 0.58)
    svg.text(1233, 225, "8 action", size=15, color=TEAL, weight=900, anchor="middle")
    svg.text(1233, 247, "probabilities", size=13, color=MUTED, weight=700, anchor="middle")
    svg.circle(1144, 327, 27, fill=PALE_ORANGE, stroke=ORANGE, stroke_width=2.5)
    svg.line(1144, 327, 1144, 310, stroke=ORANGE, width=3)
    svg.line(1144, 327, 1158, 336, stroke=ORANGE, width=3)
    svg.circle(1246, 327, 27, fill=PALE_BLUE, stroke=BLUE, stroke_width=2.5)
    svg.add(f'<path d="M 1229 336 A 20 20 0 0 1 1262 312" fill="none" stroke="{BLUE}" stroke-width="5"/>')
    svg.text(1144, 371, "watch time", size=12.5, color=ORANGE, weight=900, anchor="middle")
    svg.text(1246, 371, "completion", size=12.5, color=BLUE, weight=900, anchor="middle")
    svg.line(1132, 420, 1286, 420, stroke=TEAL, width=4)
    svg.polygon([(1209, 420), (1189, 457), (1229, 457)], fill=TEAL)
    for index in range(10):
        svg.circle(1140 + index * 15.5, 408 - (index % 2) * 9, 5.5, fill=(TEAL if index < 8 else ORANGE))
    svg.text(1209, 482, "10 normalized losses", size=14, color=TEAL, weight=900, anchor="middle")
    svg.text(1209, 503, "collection excluded", size=12.5, color=RED, weight=800, anchor="middle")

    # 6. Verification and measured outcome.
    _icon_trophy(svg, 1424, 237, GREEN, 0.72)
    _icon_shield(svg, 1536, 238, TEAL, 0.58)
    svg.text(1482, 302, f"best epoch {best_epoch}  ·  34 tests", size=14, color=NAVY, weight=900, anchor="middle")
    result_pills = [
        ("+41.8%", "RECALL", BLUE),
        ("+43.4%", "HIT RATE", GREEN),
        ("+27.1%", "NDCG", PURPLE),
    ]
    for index, (value, label, color) in enumerate(result_pills):
        y = 329 + index * 53
        svg.rect(1375, y, 214, 43, fill=body_fill_for(color), stroke=color, stroke_width=1.4, radius=9)
        svg.text(1437, y + 27, value, size=18, color=color, weight=900, anchor="middle")
        svg.text(1548, y + 27, label, size=11.5, color=NAVY, weight=900, anchor="middle")
    svg.text(1482, 501, "composite vs Week 2 retrieval", size=12.5, color=MUTED, weight=700, anchor="middle")

    for left, right in ((272, 288), (518, 534), (784, 800), (1068, 1084), (1334, 1350)):
        _flow_arrow(svg, left + 2, right - 4, 326)

    _section_label(svg, 568, "CANDIDATE RANKING", GREEN)
    bottom_y, bottom_h = 591, 167
    serving_cards = [
        (46, 237, "1. User request", BLUE),
        (299, 252, "2. Retrieve 100", GREEN),
        (567, 252, "3. Ten predictions", PURPLE),
        (835, 252, "4. Utility blend", ORANGE),
        (1103, 511, "5. Reorder the same 100", TEAL),
    ]
    for x, width, title, color in serving_cards:
        svg.rect(x, bottom_y, width, bottom_h, fill="white", stroke=color, stroke_width=1.5, radius=11, shadow=True)
        svg.rect(x, bottom_y, width, 40, fill=color, radius=11)
        svg.rect(x, bottom_y + 29, width, 11, fill=color, radius=0)
        svg.text(x + width / 2, bottom_y + 27, title, size=16, color="white", weight=900, anchor="middle")

    _icon_user(svg, 164, 685, BLUE, 0.76)
    svg.text(164, 742, "ID + live context", size=13, color=NAVY, weight=800, anchor="middle")
    _icon_database(svg, 387, 682, GREEN, 0.55)
    svg.circle(465, 684, 25, fill="white", stroke=GREEN, stroke_width=3)
    svg.line(483, 702, 501, 720, stroke=GREEN, width=5)
    svg.text(425, 742, "Week 2 two-tower + FAISS", size=12.5, color=NAVY, weight=800, anchor="middle")
    for index in range(10):
        angle_x = 606 + (index % 5) * 43
        angle_y = 666 + (index // 5) * 40
        color = RED if index == 7 else ORANGE if index > 7 else PURPLE
        svg.circle(angle_x, angle_y, 14, fill=body_fill_for(color), stroke=color, stroke_width=2)
    svg.text(693, 742, "8 actions + time + finish", size=12.5, color=NAVY, weight=800, anchor="middle")
    _icon_sliders(svg, 961, 687, ORANGE, 0.72)
    svg.text(961, 742, "positive signals − hate", size=12.5, color=NAVY, weight=800, anchor="middle")
    _icon_grid(svg, 1204, 687, BLUE, 0.72)
    svg.line(1270, 687, 1327, 687, stroke=NAVY, width=3, arrow=True)
    _icon_ranked_list(svg, 1438, 684, TEAL, 0.72)
    svg.rect(1495, 648, 91, 54, fill=PALE_GREEN, stroke=GREEN, stroke_width=2, radius=27)
    svg.text(1540, 671, "SAME", size=11.5, color=GREEN, weight=900, anchor="middle")
    svg.text(1540, 688, "100 IDs", size=12.5, color=GREEN, weight=900, anchor="middle")
    svg.text(1358, 742, "membership fixed  ·  order changes", size=13.5, color=TEAL, weight=900, anchor="middle")
    for left, right in ((283, 299), (551, 567), (819, 835), (1087, 1103)):
        _flow_arrow(svg, left + 2, right - 4, 675)

    svg.rect(46, 789, 1568, 111, fill="url(#milestone)", radius=18, shadow=True)
    svg.circle(101, 844, 35, fill="white", stroke="#77E0C1", stroke_width=3)
    svg.text(101, 855, "✓", size=34, color=GREEN, weight=900, anchor="middle")
    svg.text(157, 827, "MAJOR MILESTONE", size=16, color="#78E1C2", weight=900, letter_spacing=1.1)
    svg.text(157, 856, "First measured logs → retrieval → multi-task ranking → final list", size=25, color="white", weight=900)
    svg.text(157, 882, "DeepFM + MMoE scores 1M rows across validation and test", size=14.5, color="#DDEBFF", weight=600)
    for x, label in ((1191, "chronological ✓"), (1360, "same top-100 ✓"), (1522, "finite scores ✓")):
        svg.rect(x - 72, 821, 144, 47, fill="#FFFFFF", opacity=0.12, stroke="#A6EADB", radius=24)
        svg.text(x, 850, label, size=12.5, color="white", weight=800, anchor="middle")
    svg.text(1610, 932, "Week 3 · DeepFM + MMoE", size=13, color=MUTED, style="italic", anchor="end")
    svg.save(output)


def _metric_values(
    mmoe: dict[str, Any], deepfm: dict[str, Any]
) -> dict[str, dict[int, list[float]]]:
    click = mmoe["candidate_ranking"]["test"]["targets"]["click"]
    output: dict[str, dict[int, list[float]]] = {
        "Recall": {},
        "HitRate": {},
        "NDCG": {},
        "Coverage": {},
    }
    keys = {
        "Recall": "recall",
        "HitRate": "hit_rate",
        "NDCG": "ndcg",
        "Coverage": "coverage",
    }
    for cutoff in (20, 50):
        rows = [
            click["retrieval_order"][str(cutoff)],
            deepfm["candidate_ranking"]["test"]["deepfm_order"][str(cutoff)],
            click["mmoe_composite_order"][str(cutoff)],
            click["task_head_order"][str(cutoff)],
        ]
        for label, key in keys.items():
            output[label][cutoff] = [
                100.0 * row[f"{key}@{cutoff}"] for row in rows
            ]
    return output


def _bar_chart(
    svg: SVG,
    x: int,
    y: int,
    width: int,
    height: int,
    label: str,
    values: dict[int, list[float]],
    maximum: float,
    decimals: int,
) -> None:
    svg.rect(x, y, width, height, fill="white", stroke="#C8D2DF", radius=12, shadow=True)
    svg.text(x + 22, y + 31, label, size=20, color=NAVY, weight=900)
    plot_left = x + 62
    plot_right = x + width - 24
    plot_top = y + 55
    plot_bottom = y + height - 52
    plot_height = plot_bottom - plot_top
    for index in range(5):
        fraction = index / 4
        line_y = plot_bottom - fraction * plot_height
        svg.line(plot_left, line_y, plot_right, line_y, stroke=GRID, width=1, dash="4 4")
        svg.text(plot_left - 9, line_y + 5, f"{maximum * fraction:g}%", size=11.5, color=MUTED, anchor="end")
    svg.line(plot_left, plot_top, plot_left, plot_bottom, stroke="#6B7280", width=1.2)
    svg.line(plot_left, plot_bottom, plot_right, plot_bottom, stroke="#6B7280", width=1.2)

    fills = ["url(#barBlue)", "url(#barOrange)", "url(#barTeal)", "url(#barPurple)"]
    strokes = [BLUE, ORANGE, NAVY, PURPLE]
    group_width = (plot_right - plot_left) / 2
    bar_width, gap = 45.0, 10.0
    total_width = 4 * bar_width + 3 * gap
    for group_index, cutoff in enumerate((20, 50)):
        group_center = plot_left + group_width * (group_index + 0.5)
        start = group_center - total_width / 2
        for series_index, value in enumerate(values[cutoff]):
            center = start + series_index * (bar_width + gap) + bar_width / 2
            bar_height = max(1.0, value / maximum * plot_height)
            bar_y = plot_bottom - bar_height
            svg.rect(
                center - bar_width / 2,
                bar_y,
                bar_width,
                bar_height,
                fill=fills[series_index],
                stroke=strokes[series_index],
                stroke_width=3 if series_index == 2 else 1,
                radius=3,
            )
            svg.text(
                center,
                max(plot_top + 12, bar_y - 7),
                f"{value:.{decimals}f}%",
                size=11.5,
                color=NAVY,
                weight=900,
                anchor="middle",
            )
        svg.text(
            group_center,
            plot_bottom + 27,
            f"K = {cutoff}",
            size=13.5,
            color=NAVY,
            weight=900,
            anchor="middle",
        )
    svg.line(plot_left + group_width, plot_top + 10, plot_left + group_width, plot_bottom + 3, stroke="#E4E8EF", width=1)


def build_results(
    mmoe: dict[str, Any],
    deepfm: dict[str, Any],
    output: Path,
) -> None:
    values = _metric_values(mmoe, deepfm)
    svg = SVG(
        1600,
        1180,
        title="KuaiFlow Week 3 end-to-end ranking results",
        description=(
            "Grouped ranking metrics at K 20 and K 50, a fixed-candidate reorder "
            "diagram, and per-target NDCG comparison for the Week 3 rankers."
        ),
    )
    svg.rect(0, 0, 1600, 1180, fill="#FCFCFD", radius=0)
    svg.text(800, 43, "KuaiFlow Week 3 — End-to-End Ranking Results", size=38, weight=900, anchor="middle")
    svg.text(800, 72, "Test set · fixed Week 2 top-100 candidates · ranking depth K = 20 and 50", size=16.5, color=MUTED, anchor="middle")

    svg.rect(62, 89, 1476, 70, fill="url(#milestone)", radius=15, shadow=True)
    svg.circle(106, 124, 24, fill="white", stroke="#77E0C1", stroke_width=2.5)
    svg.text(106, 132, "✓", size=25, color=GREEN, weight=900, anchor="middle")
    svg.text(146, 117, "MAJOR MILESTONE", size=13, color="#78E1C2", weight=900, letter_spacing=1)
    svg.text(146, 142, "First measured logs → retrieval → multi-task ranking → final list", size=19.5, color="white", weight=800)
    svg.rect(1241, 103, 269, 42, fill="#FFFFFF", opacity=0.12, stroke="#A6EADB", radius=21)
    svg.text(1376, 130, "500K test rows · 5K users", size=13.5, color="white", weight=800, anchor="middle")

    legend = [
        (BLUE, "url(#barBlue)", "Week 2 retrieval"),
        (ORANGE, "url(#barOrange)", "DeepFM"),
        (TEAL, "url(#barTeal)", "MMoE composite · FINAL POLICY"),
        (PURPLE, "url(#barPurple)", "MMoE click head · DIAGNOSTIC"),
    ]
    legend_x = [296, 568, 873, 1262]
    for x, (stroke, fill, label) in zip(legend_x, legend):
        svg.rect(x - 106, 176, 27, 17, fill=fill, stroke=NAVY if stroke == TEAL else stroke, stroke_width=2 if stroke == TEAL else 1, radius=2)
        svg.text(x - 68, 190, label, size=13.5, color=NAVY, weight=800 if stroke == TEAL else 600)

    _bar_chart(svg, 62, 211, 714, 256, "A. Recall", values["Recall"], 8.5, 2)
    _bar_chart(svg, 824, 211, 714, 256, "B. HitRate", values["HitRate"], 22.0, 2)
    _bar_chart(svg, 62, 490, 714, 256, "C. NDCG", values["NDCG"], 3.3, 2)
    _bar_chart(svg, 824, 490, 714, 256, "D. Coverage", values["Coverage"], 100.0, 1)
    svg.rect(606, 501, 145, 31, fill=PALE_GREEN, stroke=GREEN, radius=16)
    svg.text(678, 522, "+27.1% at K=20", size=12.5, color=GREEN, weight=900, anchor="middle")
    svg.rect(1360, 501, 153, 31, fill=PALE_ORANGE, stroke=ORANGE, radius=16)
    svg.text(1436, 522, "−42.2 pp at K=20", size=12.5, color="#9A6200", weight=900, anchor="middle")

    # A candidate-grid animation in one still: all 100 IDs survive, only their order changes.
    svg.rect(62, 773, 506, 334, fill="white", stroke="#C8D2DF", radius=13, shadow=True)
    svg.text(86, 806, "E. Fixed-candidate handoff", size=19, color=NAVY, weight=900)
    svg.text(86, 830, "100 retrieved IDs  →  score ×10  →  new order", size=13.5, color=MUTED, weight=700)
    left_start_x, right_start_x, dots_y = 95, 343, 862
    for index in range(100):
        row, column = divmod(index, 10)
        svg.circle(left_start_x + column * 12, dots_y + row * 12, 4.2, fill=BLUE)
        right_color = TEAL if index < 20 else "#CAD4E1"
        svg.circle(right_start_x + column * 12, dots_y + row * 12, 4.2, fill=right_color)
    svg.rect(left_start_x - 11, dots_y - 13, 132, 132, fill="none", stroke=BLUE, stroke_width=1.5, radius=8)
    svg.rect(right_start_x - 11, dots_y - 13, 132, 132, fill="none", stroke=TEAL, stroke_width=1.5, radius=8)
    svg.rect(right_start_x - 8, dots_y - 10, 126, 25, fill="none", stroke=ORANGE, stroke_width=2.5, radius=5)
    svg.line(229, 915, 315, 915, stroke=NAVY, width=3, arrow=True)
    _icon_sliders(svg, 272, 915, ORANGE, 0.36)
    svg.text(150, 1002, "WEEK 2 TOP 100", size=12.5, color=BLUE, weight=900, anchor="middle")
    svg.text(398, 1002, "TOP 20 HIGHLIGHTED", size=12.5, color=TEAL, weight=900, anchor="middle")
    svg.rect(87, 1031, 456, 52, fill=PALE_GREEN, stroke=GREEN, radius=12)
    svg.text(315, 1053, "✓ Recall@100 + HitRate@100 unchanged", size=14, color=GREEN, weight=900, anchor="middle")
    svg.text(315, 1074, "ranking cannot recover a missed candidate", size=12.5, color=MUTED, anchor="middle")

    # Multi-objective view: exact test NDCG@20 positions for the three orderings.
    svg.rect(594, 773, 944, 334, fill="white", stroke="#C8D2DF", radius=13, shadow=True)
    svg.text(618, 806, "F. Target-head NDCG@20", size=19, color=NAVY, weight=900)
    svg.circle(986, 801, 6, fill=BLUE)
    svg.text(998, 806, "retrieval", size=11.5, color=MUTED)
    svg.rect(1068, 795, 12, 12, fill=TEAL, radius=2)
    svg.text(1087, 806, "composite", size=11.5, color=MUTED)
    svg.polygon([(1186, 795), (1193, 802), (1186, 809), (1179, 802)], fill=PURPLE)
    svg.text(1200, 806, "task head", size=11.5, color=MUTED)
    svg.text(1507, 806, "users", size=11.5, color=MUTED, weight=800, anchor="end")
    plot_left, plot_right = 755, 1450
    for tick in range(5):
        x = plot_left + tick / 4 * (plot_right - plot_left)
        svg.line(x, 825, x, 1067, stroke=GRID, width=1, dash="3 4")
        svg.text(x, 1088, f"{tick * 0.875:.1f}%", size=10.5, color=MUTED, anchor="middle")
    target_labels = [
        ("click", "Click"),
        ("like", "Like"),
        ("long_view", "Long-view"),
        ("profile_enter", "Profile-entry"),
        ("follow", "Follow"),
        ("comment", "Comment"),
        ("forward", "Forward"),
    ]
    target_results = mmoe["candidate_ranking"]["test"]["targets"]
    scale_max = 3.5
    for index, (key, label) in enumerate(target_labels):
        row = target_results[key]
        retrieval = 100.0 * row["retrieval_order"]["20"]["ndcg@20"]
        composite = 100.0 * row["mmoe_composite_order"]["20"]["ndcg@20"]
        head = 100.0 * row["task_head_order"]["20"]["ndcg@20"]
        users = int(row["candidate_ground_truth_users"])
        y = 842 + index * 33
        x_retrieval = plot_left + retrieval / scale_max * (plot_right - plot_left)
        x_composite = plot_left + composite / scale_max * (plot_right - plot_left)
        x_head = plot_left + head / scale_max * (plot_right - plot_left)
        svg.text(731, y + 4, label, size=12.5, color=NAVY, weight=800, anchor="end")
        svg.line(min(x_retrieval, x_composite), y, max(x_retrieval, x_composite), y, stroke=TEAL, width=3)
        svg.line(min(x_composite, x_head), y, max(x_composite, x_head), y, stroke=PURPLE, width=1.7, dash="4 3")
        svg.circle(x_retrieval, y, 5.5, fill=BLUE)
        svg.rect(x_composite - 5.5, y - 5.5, 11, 11, fill=TEAL, radius=2)
        svg.polygon([(x_head, y - 7), (x_head + 7, y), (x_head, y + 7), (x_head - 7, y)], fill=PURPLE)
        svg.text(1507, y + 4, f"n={users:,}", size=11.5, color=MUTED, weight=700, anchor="end")
    svg.text(1066, 1100, "NDCG@20  ·  rare-action rows (n=33–84) are high variance", size=12, color=MUTED, style="italic", anchor="middle")

    svg.rect(62, 1125, 1476, 35, fill="#FFF8EA", stroke="#D89B26", radius=9)
    svg.text(800, 1148, "Composite vs DeepFM NDCG@20: +6.6% test, −0.8% validation · equal-weight utility is a transparent baseline, not an optimized policy", size=13, color="#72500D", weight=800, anchor="middle")
    svg.text(1538, 1175, "Higher ranking metrics are better · coverage is a concentration diagnostic", size=11.5, color=MUTED, style="italic", anchor="end")
    svg.save(output)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _render_png(svg_path: Path, width: int, height: int) -> Path | None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        return None
    output = svg_path.with_suffix(".png")
    subprocess.run(
        [converter, "-w", str(width), "-h", str(height), "-o", str(output), str(svg_path)],
        check=True,
    )
    return output


def generate_week3_figures(repo_root: str | Path, render_png: bool = True) -> list[Path]:
    root = Path(repo_root)
    mmoe = _load_json(root / "artifacts" / "week3_mmoe_results.json")
    deepfm = _load_json(root / "artifacts" / "week3_deepfm_results.json")
    figures = root / "figures"
    workflow = figures / "Week_3_workflow_diagram.svg"
    results = figures / "Week_3_experiment_results.svg"
    build_workflow(mmoe, workflow)
    build_results(mmoe, deepfm, results)
    outputs = [workflow, results]
    if render_png:
        for path, width, height in ((workflow, 1660, 948), (results, 1600, 1180)):
            rendered = _render_png(path, width, height)
            if rendered is not None:
                outputs.append(rendered)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--svg-only", action="store_true")
    args = parser.parse_args()
    for path in generate_week3_figures(args.repo_root, render_png=not args.svg_only):
        print(path)


if __name__ == "__main__":
    main()
