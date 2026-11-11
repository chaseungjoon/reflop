# SPDX-License-Identifier: Apache-2.0
"""Render docs/images/how-it-works-{dark,light}.png for the README (needs google-chrome and the Inter font).

    python3 docs/images/how-it-works.py docs/images
"""
import pathlib, subprocess, sys

OUT = pathlib.Path(sys.argv[1])
HERE = pathlib.Path(__file__).parent
W, H = 960, 720

THEMES = {
    "dark": dict(text="#e6edf3", muted="#9198a1", box="#151b23", stroke="#8b949e", rule="#2a313c",
                 dash="#6e7681", green="#3fb950", gtext="#56d364", gbox="#0b2516", gstroke="#3fb950"),
    "light": dict(text="#1f2328", muted="#59636e", box="#f6f8fa", stroke="#8c959f", rule="#d1d9e0",
                  dash="#8c959f", green="#1a7f37", gtext="#1a7f37", gbox="#ecfdf0", gstroke="#2da44e"),
}


def box(x, y, w, h, title, lines, fill, stroke, c, sw=2):
    cx = x + w / 2
    n = len(lines)
    top = y + h / 2 - (34 + 30 * n) / 2 + 24
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>',
           f'<text x="{cx}" y="{top}" class="t" fill="{c["text"]}">{title}</text>']
    for i, ln in enumerate(lines):
        out.append(f'<text x="{cx}" y="{top + 40 + 30 * i}" class="b" fill="{c["text"]}">{ln}</text>')
    return "\n".join(out)


def arrow(x1, x2, y, color, dashed, width=3.5):
    d = 1 if x2 > x1 else -1
    tip = x2
    shaft_end = x2 - d * 6
    dash = ' stroke-dasharray="12 9"' if dashed else ""
    head = (f'<path d="M{tip - d * 14},{y - 11} L{tip},{y} L{tip - d * 14},{y + 11}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')
    return (f'<line x1="{x1}" y1="{y}" x2="{shaft_end}" y2="{y}" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round"{dash}/>' + head)


def svg(c):
    L, R, BW = 24, 656, 280  # left box x, right box x, box width
    a1, a2 = L + BW + 26, R - 26
    parts = [
        f'<text x="{L}" y="52" class="h" fill="{c["text"]}">Conventional remote GPU execution</text>',
        box(L, 100, BW, 150, "Client", ["Code + data"], c["box"], c["stroke"], c),
        box(R, 100, BW, 150, "GPU server", ["Runs code"], c["box"], c["stroke"], c),
        arrow(a1, a2, 175, c["dash"], True),
        f'<line x1="{L}" y1="322" x2="{W - L}" y2="322" stroke="{c["rule"]}" stroke-width="1.5"/>',
        f'<text x="{L}" y="400" class="h" fill="{c["text"]}">reFLOP’s approach</text>',
        box(L, 450, BW, 230, "Client", ["Python runs", "locally"], c["box"], c["stroke"], c),
        box(R, 450, BW, 230, "GPU server", ["Validates ops", "Runs tensor", "operations"], c["gbox"],
            c["gstroke"], c, 2.5),
        f'<text x="{(a1 + a2) / 2}" y="522" class="l" fill="{c["gtext"]}">Tensor graphs</text>',
        arrow(a1, a2, 540, c["green"], False),
        arrow(a2, a1, 612, c["dash"], True),
        f'<text x="{(a1 + a2) / 2}" y="652" class="l" fill="{c["muted"]}">Results</text>',
    ]
    body = "\n".join(parts)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<style>
text {{ font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif; }}
.h {{ font-size: 34px; font-weight: 500; letter-spacing: -0.3px; }}
.t {{ font-size: 27px; font-weight: 600; text-anchor: middle; }}
.b {{ font-size: 23px; font-weight: 400; text-anchor: middle; }}
.l {{ font-size: 21px; font-weight: 450; text-anchor: middle; }}
</style>
{body}
</svg>'''


for name, c in THEMES.items():
    s = svg(c)
    html = HERE / f"{name}.html"
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;'
                    f'background:transparent}}svg{{display:block}}</style></head><body>{s}</body></html>')
    png = OUT / f"how-it-works-{name}.png"
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--screenshot={png}", f"--window-size={W},{H}", "--force-device-scale-factor=2",
                    "--default-background-color=00000000", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(png)
