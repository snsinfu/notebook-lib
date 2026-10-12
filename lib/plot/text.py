"""
Text formatting for plotting labels.
"""

from __future__ import annotations


def texsci(x: float, n: int) -> str:
    """
    Format `x` with `n` significant digits as `a \\times 10^{b}`.

    For use in text labels (e.g. `Axes.text`, titles, annotations), not as a
    matplotlib tick `Formatter`. Non-scientific values are formatted plainly.
    """
    s = f"{{:.{n}g}}".format(x)
    if "e+" in s:
        s = s.replace("e+", r" \times 10^{").replace("{0", "{") + "}"
    if "e-" in s:
        s = s.replace("e-", r" \times 10^{-").replace("{-0", "{-") + "}"
    return s
