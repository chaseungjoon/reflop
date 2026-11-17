# SPDX-License-Identifier: Apache-2.0
"""Render docs/images/how-it-works-{dark,light}.png for the README (needs google-chrome and the Inter font).

    python3 docs/images/how-it-works.py docs/images
"""
import math
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

THEMES = {
    "light": NS(text="#1f2328", muted="#59636e", faint="#d1d9e0", surf="#f6f8fa", bg="#ffffff",
                stroke="#8c959f", ink="#424a53", green="#1a7f37", gfill="#dafbe1", gstroke="#2da44e",
                blue="#0969da", bfill="#ddf4ff", red="#cf222e", rfill="#ffebe9", purple="#8250df",
                orange="#bc4c00", screen="#24292f"),
    "dark": NS(text="#e6edf3", muted="#9198a1", faint="#30363d", surf="#151b23", bg="#0d1117",
               stroke="#8b949e", ink="#b7bdc8", green="#3fb950", gfill="#0f2a19", gstroke="#3fb950",
               blue="#4493f8", bfill="#0d2142", red="#f85149", rfill="#3a1418", purple="#ab7df8",
               orange="#db6d28", screen="#010409"),
}

SANS = 'Inter, "Segoe UI", Helvetica, Arial, sans-serif'
MONO = '"JetBrains Mono", "JetBrainsMono Nerd Font Mono", ui-monospace, Menlo, monospace'


def text(x, y, s, c, size=16, fill=None, weight=500, anchor="middle", mono=False, spacing=0, halo=False):
    fam = MONO if mono else SANS
    h = (f' stroke="{c.bg}" stroke-width="8" paint-order="stroke" stroke-linejoin="round"' if halo else "")
    return (f'<text x="{x}" y="{y}" font-family=\'{fam}\' font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" letter-spacing="{spacing}" fill="{fill or c.text}"{h}>{s}</text>')


def caps(x, y, s, c, anchor="middle", fill=None):
    return text(x, y, s.upper(), c, 14, fill or c.muted, 600, anchor, spacing=2.2)


def arrow(x1, y1, x2, y2, color, w=3, dashed=False, head=12):
    ang = math.atan2(y2 - y1, x2 - x1)
    ex, ey = x2 - math.cos(ang) * 4, y2 - math.sin(ang) * 4
    d = ' stroke-dasharray="9 8"' if dashed else ""
    hx1, hy1 = x2 - head * math.cos(ang - 0.6), y2 - head * math.sin(ang - 0.6)
    hx2, hy2 = x2 - head * math.cos(ang + 0.6), y2 - head * math.sin(ang + 0.6)
    return (f'<line x1="{x1}" y1="{y1}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{color}" stroke-width="{w}" '
            f'stroke-linecap="round"{d}/>'
            f'<path d="M{hx1:.1f},{hy1:.1f} L{x2},{y2} L{hx2:.1f},{hy2:.1f}" fill="none" stroke="{color}" '
            f'stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round"/>')


CODE_ROWS = [(0, .30, "purple", .22, "blue"), (1, .50, "grey", 0, None), (1, .20, "orange", .30, "grey"),
             (2, .44, "green", 0, None), (1, .34, "grey", .16, "blue"), (0, .28, "purple", .18, "grey")]
TENSOR_ROWS = {1, 3, 4}


def code_lines(bx, by, bw, bh, c, hl=None):
    """Abstract code in a screen box. Rows in hl (tensor ops) are drawn blue on a band.
    Returns (svg, [row y, ...])."""
    hl = hl or set()

    def col(k):
        return "#8b949e" if k == "grey" else getattr(c, k)

    py = bh * 0.16
    lh = (bh - 2 * py) / (len(CODE_ROWS) - 1)
    lx, scale, sw = bx + bw * 0.09, bw * 0.82, max(3, bw * 0.036)
    out, ys = [], []
    for i, (ind, a, ca, b, cb) in enumerate(CODE_ROWS):
        yy = by + py + i * lh
        ys.append(yy)
        sx = lx + ind * bw * 0.08
        if i in hl:
            out.append(f'<rect x="{bx+bw*0.03:.1f}" y="{yy-lh*0.42:.1f}" width="{bw*0.94:.1f}" '
                       f'height="{lh*0.84:.1f}" rx="2" fill="{c.blue}" opacity="0.28"/>')
            ca = cb = "blue"
        out.append(f'<line x1="{sx:.1f}" y1="{yy:.1f}" x2="{sx+a*scale:.1f}" y2="{yy:.1f}" stroke="{col(ca)}" '
                   f'stroke-width="{sw:.1f}" stroke-linecap="round"/>')
        if b:
            x2 = sx + a * scale + bw * 0.06
            out.append(f'<line x1="{x2:.1f}" y1="{yy:.1f}" x2="{x2+b*scale:.1f}" y2="{yy:.1f}" '
                       f'stroke="{col(cb)}" stroke-width="{sw:.1f}" stroke-linecap="round"/>')
    return "\n".join(out), ys


def laptop(x, y, w, c, code=True, hl=None, ret_rows=False):
    """Laptop whose screen shows abstract Python, or an idle cursor. (x, y) is the screen's top-left corner."""
    h = w * 0.64
    sx, sy, sw_, sh = x + w * 0.06, y + w * 0.06, w * 0.88, h - w * 0.12
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h:.1f}" rx="{w*0.05:.1f}" fill="{c.surf}" '
           f'stroke="{c.ink}" stroke-width="3"/>',
           f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{sw_:.1f}" height="{sh:.1f}" rx="{w*0.025:.1f}" '
           f'fill="{c.screen}"/>',
           f'<path d="M{x-w*0.1:.1f},{y+h+3:.1f} H{x+w*1.1:.1f} l{-w*0.06:.1f},{w*0.07:.1f} '
           f'H{x-w*0.04:.1f} Z" fill="{c.surf}" stroke="{c.ink}" stroke-width="3" stroke-linejoin="round"/>']
    ys = []
    if code:
        svg, ys = code_lines(sx, sy, sw_, sh, c, hl)
        out.append(svg)
    else:
        out.append(f'<rect x="{sx+sw_*0.09:.1f}" y="{sy+sh*0.12:.1f}" width="{w*0.05:.1f}" height="{sh*0.14:.1f}" '
                   f'fill="#8b949e" opacity="0.8"/>')
    return ("\n".join(out), ys) if ret_rows else "\n".join(out)


def code_window(x, y, w, c, hl=None, col=None):
    """Editor window full of code, with a title bar."""
    col = col or c.ink
    h = w * 0.8
    tb = h * 0.2
    svg, _ = code_lines(x, y + tb, w, h - tb, c, hl)
    dots = "".join(f'<circle cx="{x+w*0.09+i*w*0.08:.1f}" cy="{y+tb/2:.1f}" r="{w*0.025:.1f}" fill="#8b949e"/>'
                   for i in range(3))
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h:.1f}" rx="9" fill="{c.screen}" stroke="{col}" '
            f'stroke-width="3"/>'
            f'<line x1="{x}" y1="{y+tb:.1f}" x2="{x+w}" y2="{y+tb:.1f}" stroke="{col}" stroke-width="1.5" '
            f'opacity="0.6"/>' + dots + svg)


def run_badge(cx, cy, c, col):
    """'▶ python': where the Python program runs."""
    w, h = 104, 28
    t = f'M{cx-w/2+14},{cy-6} L{cx-w/2+24},{cy} L{cx-w/2+14},{cy+6} Z'
    fill = c.gfill if col == c.green else c.surf
    return (f'<rect x="{cx-w/2}" y="{cy-h/2}" width="{w}" height="{h}" rx="{h/2}" fill="{fill}" stroke="{col}" '
            f'stroke-width="2"/><path d="{t}" fill="{col}" stroke="{col}" stroke-width="1.5" '
            f'stroke-linejoin="round"/>' + text(cx + 10, cy + 5, "python", c, 15, col, 600, mono=True))


def wordmark(x, y, c, size=18, anchor="start"):
    return text(x, y, 're<tspan font-weight="800">FLOP</tspan>', c, size, c.green, 500, anchor)


def sandboxed_gpu(x, y, w, c):
    h = w * 0.48
    return (f'<rect x="{x-24}" y="{y-30}" width="{w+48}" height="{h+60:.1f}" rx="18" fill="none" '
            f'stroke="{c.green}" stroke-width="2.5" stroke-dasharray="7 7"/>'
            + gpu(x, y, w, c) + lock(x + w + 24, y - 30, 11, c))


def fan(cx, cy, r, c, color):
    blade = (f'M0,{-r*.22:.1f} C{r*.42:.1f},{-r*.32:.1f} {r*.55:.1f},{-r*.70:.1f} {r*.16:.1f},{-r*.90:.1f} '
             f'C{r*.02:.1f},{-r*.66:.1f} {-r*.06:.1f},{-r*.42:.1f} 0,{-r*.22:.1f} Z')
    blades = "".join(f'<path d="{blade}" transform="rotate({i*51.4:.1f})" fill="{color}" opacity="0.9"/>'
                     for i in range(7))
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c.bg}" stroke="{c.ink}" stroke-width="2.5"/>'
            f'<g transform="translate({cx},{cy})">{blades}</g>'
            f'<circle cx="{cx}" cy="{cy}" r="{r*.2:.1f}" fill="{c.surf}" stroke="{c.ink}" stroke-width="2"/>')


def gpu(x, y, w, c, accent=None):
    """Graphics card, two fans. (x, y) is the top-left of the shroud."""
    h = w * 0.48
    a = accent or c.green
    r = h * 0.36
    out = [f'<rect x="{x-w*0.05:.1f}" y="{y-h*0.08:.1f}" width="{w*0.035:.1f}" height="{h*1.16:.1f}" rx="2" '
           f'fill="{c.surf}" stroke="{c.ink}" stroke-width="2.5"/>',
           f'<rect x="{x}" y="{y}" width="{w}" height="{h:.1f}" rx="{h*0.14:.1f}" fill="{c.surf}" '
           f'stroke="{c.ink}" stroke-width="3"/>',
           f'<line x1="{x+w*0.04:.1f}" y1="{y+h*0.86:.1f}" x2="{x+w*0.96:.1f}" y2="{y+h*0.86:.1f}" '
           f'stroke="{a}" stroke-width="3" stroke-linecap="round"/>']
    for i in range(9):
        fx = x + w * 0.30 + i * w * 0.045
        out.append(f'<rect x="{fx:.1f}" y="{y+h+2:.1f}" width="{w*0.028:.1f}" height="{h*0.12:.1f}" rx="1" '
                   f'fill="{c.orange}" opacity="0.85"/>')
    out.append(fan(x + w * 0.29, y + h * 0.45, r, c, a))
    out.append(fan(x + w * 0.71, y + h * 0.45, r, c, a))
    return "\n".join(out)


DAG_NODES = {"a": (0, 18, "x"), "b": (0, 72, "W"), "c": (58, 45, "@"), "d": (112, 18, "σ"),
             "e": (112, 72, "+"), "f": (166, 45, "Σ")}
DAG_EDGES = [("a", "c"), ("b", "c"), ("c", "d"), ("c", "e"), ("d", "f"), ("e", "f"), ("b", "e")]


def dag(x, y, c, s=1.0, color=None, glyphs=True):
    """Small tensor graph: x, W -> matmul -> (sigmoid, add) -> sum. Box is ~166*s by 90*s."""
    col = color or c.blue
    P = {k: (x + px * s, y + py * s) for k, (px, py, _) in DAG_NODES.items()}
    out = []
    for u, v in DAG_EDGES:
        (x1, y1), (x2, y2) = P[u], P[v]
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
                   f'stroke-width="{2.5*s:.1f}" opacity="0.7"/>')
    for k, (px, py) in P.items():
        g = DAG_NODES[k][2]
        r = 15 * s
        leaf = k in "ab"
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" fill="{c.bfill if not leaf else c.bg}" '
                   f'stroke="{col}" stroke-width="{2.5*s:.1f}"/>')
        if glyphs:
            out.append(text(f"{px:.1f}", f"{py+5.5*s:.1f}", g, c, 16 * s, col, 600, mono=True))
    return "\n".join(out)


def lock(cx, cy, s, c, color=None):
    col = color or c.green
    return (f'<circle cx="{cx}" cy="{cy}" r="{s*1.25:.1f}" fill="{c.bg}" stroke="{col}" stroke-width="2.5"/>'
            f'<path d="M{cx-s*0.38:.1f},{cy-s*0.05:.1f} V{cy-s*0.38:.1f} a{s*0.38:.1f},{s*0.38:.1f} 0 0 1 '
            f'{s*0.76:.1f},0 V{cy-s*0.05:.1f}" fill="none" stroke="{col}" stroke-width="2.5"/>'
            f'<rect x="{cx-s*0.58:.1f}" y="{cy-s*0.08:.1f}" width="{s*1.16:.1f}" height="{s*0.78:.1f}" rx="3" '
            f'fill="{col}"/>')


def badge(cx, cy, r, c, kind):
    """Round check or cross mark."""
    if kind == "ok":
        return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c.gfill}" stroke="{c.green}" stroke-width="3"/>'
                f'<path d="M{cx-r*0.42:.1f},{cy+r*0.02:.1f} L{cx-r*0.1:.1f},{cy+r*0.34:.1f} '
                f'L{cx+r*0.45:.1f},{cy-r*0.3:.1f}" fill="none" stroke="{c.green}" stroke-width="4" '
                f'stroke-linecap="round" stroke-linejoin="round"/>')
    k = r * 0.36
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c.rfill}" stroke="{c.red}" stroke-width="3"/>'
            f'<path d="M{cx-k:.1f},{cy-k:.1f} L{cx+k:.1f},{cy+k:.1f} M{cx+k:.1f},{cy-k:.1f} L{cx-k:.1f},{cy+k:.1f}" '
            f'stroke="{c.red}" stroke-width="4" stroke-linecap="round"/>')


def code_file(x, y, w, c, col=None):
    """Document with a folded corner and </> on it."""
    col = col or c.ink
    h = w * 1.25
    f = w * 0.28
    return (f'<path d="M{x},{y+6} a6,6 0 0 1 6,-6 H{x+w-f:.1f} L{x+w},{y+f:.1f} V{y+h-6:.1f} a6,6 0 0 1 -6,6 '
            f'H{x+6} a6,6 0 0 1 -6,-6 Z" fill="{c.surf}" stroke="{col}" stroke-width="3" stroke-linejoin="round"/>'
            f'<path d="M{x+w-f:.1f},{y} V{y+f:.1f} H{x+w}" fill="none" stroke="{col}" stroke-width="3" '
            f'stroke-linejoin="round"/>'
            + text(x + w / 2, f"{y+h*0.62:.1f}", "&lt;/&gt;", c, w * 0.3, col, 700, mono=True))


def how_it_works(c):
    """The usual way and reFLOP, laid over 'your machine' and 'remote GPU' zones."""
    W, H = 1000, 440
    ya, yb = 150, 336
    L, R = (16, 430), (482, 984)
    p = [f'<rect x="{L[0]}" y="56" width="{L[1]-L[0]}" height="372" rx="22" fill="{c.surf}"/>',
         f'<rect x="{R[0]}" y="56" width="{R[1]-R[0]}" height="372" rx="22" fill="{c.surf}"/>',
         caps((L[0] + L[1]) / 2, 38, "your machine", c), caps((R[0] + R[1]) / 2, 38, "remote GPU", c),
         f'<line x1="40" y1="243" x2="406" y2="243" stroke="{c.faint}" stroke-width="1.5"/>',
         f'<line x1="506" y1="243" x2="960" y2="243" stroke="{c.faint}" stroke-width="1.5"/>',
         # usual
         caps(40, 88, "usual", c, "start"),
         laptop(80, ya - 40, 130, c, code=False),
         arrow(232, ya, 410, ya, c.stroke, dashed=True),
         code_file(423, ya - 42, 66, c),
         arrow(500, ya, 545, ya, c.stroke, dashed=True),
         code_window(558, ya - 50, 124, c),
         run_badge(620, ya + 70, c, c.ink),
         gpu(720, ya - 40, 160, c, accent=c.stroke),
         badge(935, ya, 24, c, "no"),
         # reFLOP
         wordmark(40, 276, c),
         laptop(80, yb - 40, 130, c),
         run_badge(145, yb + 70, c, c.green),
         arrow(232, yb, 374, yb, c.blue),
         dag(388, yb - 37, c, 0.82),
         text(456, yb + 70, "tensor ops", c, 15, c.blue, 600),
         arrow(536, yb, 690, yb, c.blue),
         sandboxed_gpu(720, yb - 38, 160, c),
         badge(935, yb, 24, c, "ok")]
    return W, H, p


def render(fn, c, png):
    W, H, parts = fn(c)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           + "\n".join(parts) + "</svg>")
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;'
                f'background:transparent}}svg{{display:block}}</style></head><body>{svg}</body></html>')
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--screenshot={png}", f"--window-size={W},{H}", "--force-device-scale-factor=2",
                    "--default-background-color=00000000", pathlib.Path(f.name).as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pathlib.Path(f.name).unlink()


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1])
    for theme, c in THEMES.items():
        png = out / f"how-it-works-{theme}.png"
        render(how_it_works, c, png)
        print(png)
