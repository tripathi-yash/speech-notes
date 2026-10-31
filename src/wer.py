"""Word Error Rate (WER) computed with a plain Levenshtein alignment.

No third-party dependencies: the metric is small enough to implement and
read directly, which also makes it easy to explain in the report.
"""
import re
from dataclasses import dataclass, field


@dataclass
class WerResult:
    ref_words: int
    substitutions: int
    deletions: int
    insertions: int
    ops: list = field(default_factory=list)  # ("OK"|"S"|"D"|"I", ref_word, hyp_word)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.ref_words == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.ref_words


import unicodedata

ABBREVIATIONS = {"mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor"}

def normalize(text: str) -> list[str]:
    """Lower-case, drop punctuation in any script, keep marks, expand abbreviations."""
    chars = []
    for ch in text.lower():
        cat = unicodedata.category(ch)
        if ch.isspace() or ch == "'" or cat[0] in "LMN":  # letters, marks, numbers
            chars.append(ch)
        else:
            chars.append(" ")
    return [ABBREVIATIONS.get(w, w) for w in "".join(chars).split()]

def align(ref: list[str], hyp: list[str]) -> WerResult:
    """Align two word lists and count substitutions, deletions and insertions."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         # deletion
                dp[i][j - 1] + 1,         # insertion
                dp[i - 1][j - 1] + cost,  # match or substitution
            )

    # Backtrace to recover which operation produced each step.
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            kind = "OK" if ref[i - 1] == hyp[j - 1] else "S"
            ops.append((kind, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", hyp[j - 1]))
            j -= 1
    ops.reverse()

    subs = sum(1 for k, _, _ in ops if k == "S")
    dels = sum(1 for k, _, _ in ops if k == "D")
    ins = sum(1 for k, _, _ in ops if k == "I")
    return WerResult(ref_words=n, substitutions=subs, deletions=dels,
                     insertions=ins, ops=ops)


def compute_wer(reference: str, hypothesis: str) -> WerResult:
    """Normalise both texts, then align them."""
    return align(normalize(reference), normalize(hypothesis))


def error_lines(result: WerResult) -> list[str]:
    """Human-readable list of the mismatches, used in the error review file."""
    lines = []
    for kind, r, h in result.ops:
        if kind == "S":
            lines.append(f'substitution: expected "{r}", got "{h}"')
        elif kind == "D":
            lines.append(f'deletion: missing "{r}"')
        elif kind == "I":
            lines.append(f'insertion: extra "{h}"')
    return lines
