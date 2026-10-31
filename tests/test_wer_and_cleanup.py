"""Tests for the parts that do not need a speech model. Run with:

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wer import compute_wer, normalize  # noqa: E402
from cleanup import clean_transcript  # noqa: E402


class WerTests(unittest.TestCase):
    def test_identical_text_has_zero_wer(self):
        self.assertEqual(compute_wer("the cat sat", "the cat sat").wer, 0.0)

    def test_one_substitution_in_four_words(self):
        r = compute_wer("the cat sat down", "the dog sat down")
        self.assertEqual((r.substitutions, r.deletions, r.insertions), (1, 0, 0))
        self.assertAlmostEqual(r.wer, 0.25)

    def test_deletion_and_insertion(self):
        r = compute_wer("one two three four", "one three four five")
        self.assertEqual(r.deletions, 1)
        self.assertEqual(r.insertions, 1)
        self.assertAlmostEqual(r.wer, 0.5)

    def test_case_and_punctuation_are_ignored(self):
        self.assertEqual(compute_wer("Hello, world.", "hello world").wer, 0.0)

    def test_apostrophes_are_kept(self):
        self.assertEqual(normalize("Don't stop"), ["don't", "stop"])

    def test_wer_can_exceed_one_with_many_insertions(self):
        self.assertGreater(compute_wer("hi", "hi there friend again").wer, 1.0)

    def test_empty_reference(self):
        self.assertEqual(compute_wer("", "").wer, 0.0)
        self.assertEqual(compute_wer("", "something").wer, 1.0)


class CleanupTests(unittest.TestCase):
    def test_capitalises_and_adds_final_stop(self):
        self.assertEqual(clean_transcript("  this is a test "), "This is a test.")

    def test_one_sentence_per_line(self):
        out = clean_transcript("first point. second point! third one?")
        self.assertEqual(out.split("\n"), ["First point.", "Second point!", "Third one?"])

    def test_removes_space_before_punctuation(self):
        self.assertEqual(clean_transcript("hello , world ."), "Hello, world.")

    def test_empty_input(self):
        self.assertEqual(clean_transcript("   "), "")


if __name__ == "__main__":
    unittest.main()
