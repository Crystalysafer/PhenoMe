"""Which organ the app picks when the prompt named none.

The decision is not "whatever scores highest in the image": a language model narrows the
field first, because a word can only mean certain organs -- 扁豆 is the pod or the seed,
never the flower. Measured on one photograph of three pods, the unrestricted probe put a
flower first on box count (4 boxes against the pods' 3) and only lost on score, so
"highest wins" is right by luck there and wrong in general.

These tests need no model and no GPU: ``organ_scores`` is the only part that touches
Grounding DINO, and it is replaced. What is under test is the ranking and the refusals.

    python -m unittest pheno.test_detector
"""

from __future__ import annotations

import unittest

from pheno.TextDetector import TextDetector


def summary(**organs) -> dict:
    """A probe result, in the shape ``summarize()`` returns."""
    return {
        name: {"count": count, "best": best, "scores": [best]}
        for name, (count, best) in organs.items()
    }


#: What the four-organ probe returned for a three-pod bean photograph:
#: the correct organ wins on score, a flower wins on count and scores close behind.
BEAN_PHOTO = summary(pod=(3, 0.449), seed=(1, 0.333), flower=(4, 0.302), leaf=(0, 0.0))


class OrganChoiceTestCase(unittest.TestCase):
    def detector(self, probe: dict, single: dict | None = None) -> TextDetector:
        """A detector with the model replaced.

        ``probe`` is what the four-organ probe returns. ``single`` maps an organ to the score
        a single-organ detection gives it -- the comparison the tie-break makes when the probe
        cannot separate the candidates -- and an organ missing from it detects nothing.

        ``__new__`` on purpose: ``__init__`` loads Grounding DINO, and nothing under test
        needs it.
        """
        detector = TextDetector.__new__(TextDetector)
        detector.organ_scores = lambda rgb, **kwargs: probe
        scores = dict(single or {})
        detector.detect = lambda rgb, word, **kwargs: (
            [{"label": word, "score": scores[word], "box": [0, 0, 1, 1]}]
            if word in scores
            else []
        )
        return detector

    def resolve(self, prompt, probe=BEAN_PHOTO, candidates=None, single=None):
        return self.detector(probe, single).resolve_organ(None, prompt, candidates=candidates)

    # -- the tie-break --------------------------------------------------------

    def test_a_close_probe_is_settled_by_single_organ_detection(self):
        """The bug this was written for: a seed image came back labelled as pods.

        On that image the four-organ probe gave seed 0.351 against pod 0.297 -- a 0.054
        margin against a 0.05 threshold, a coin flip. Asked one organ at a time the same
        image is not close at all.
        """
        seed_probe = summary(seed=(1, 0.351), pod=(2, 0.297), leaf=(0, 0.0), flower=(0, 0.0))

        organ, choice, _ = self.resolve(
            "扁豆",
            probe=seed_probe,
            candidates=["pod", "seed"],
            single={"seed": 0.61, "pod": 0.52},
        )

        self.assertEqual(organ, "seed")
        self.assertEqual(choice["separated_by"], "single-organ detection")
        self.assertEqual(choice["score"], 0.61)

    def test_a_decisive_probe_does_not_pay_for_the_extra_inferences(self):
        probe = summary(pod=(3, 0.62), seed=(1, 0.20))

        organ, choice, _ = self.resolve("扁豆", probe=probe, candidates=["pod", "seed"])

        self.assertEqual(organ, "pod")
        self.assertEqual(choice["separated_by"], "four-organ probe")

    def test_a_tie_break_that_detects_nothing_leaves_the_prompt_alone(self):
        """No candidate is detectable either way: refuse rather than pick on noise."""
        probe = summary(pod=(3, 0.41), seed=(3, 0.40))

        organ, choice, _ = self.resolve("扁豆", probe=probe, candidates=["pod", "seed"])

        self.assertEqual(organ, "扁豆")
        self.assertIsNone(choice)

    # -- candidates -----------------------------------------------------------

    def test_candidates_keep_the_implausible_organ_out(self):
        """扁豆 is the pod or the seed; a flower is not what anyone means by it.

        Blunt, but this is the whole reason candidates exist: without them a flower can win
        on a photograph of pods.
        """
        organ, choice, _ = self.resolve(
            "扁豆", candidates=["pod", "seed"], single={"pod": 0.72, "seed": 0.61}
        )

        self.assertEqual(organ, "pod")
        self.assertEqual(choice["candidates"], ["pod", "seed"])
        # A flower is nowhere in the answer, because it was never a candidate.
        self.assertNotIn("flower", choice["others"])

    def test_a_flower_that_scores_highest_is_still_not_chosen_when_excluded(self):
        probe = summary(flower=(5, 0.62), pod=(3, 0.41), seed=(1, 0.2))

        organ, choice, _ = self.resolve("扁豆", probe=probe, candidates=["pod", "seed"])

        self.assertEqual(organ, "pod")
        self.assertNotIn("flower", choice["others"])

    def test_without_candidates_it_falls_back_to_all_four(self):
        """A caller that offers no narrowing still gets an answer, just a less informed one."""
        organ, choice, _ = self.resolve(
            "扁豆", single={"pod": 0.72, "seed": 0.61, "flower": 0.3}
        )

        self.assertEqual(organ, "pod")
        self.assertNotIn("candidates", choice)
        self.assertIn("seed", choice["others"])

    # -- refusals -------------------------------------------------------------

    def test_an_explicit_organ_is_left_alone(self):
        for prompt in ("pod", "seed", "豆荚", "种子", "扁豆豆荚"):
            organ, choice, _ = self.resolve(prompt, candidates=["pod", "seed"])
            self.assertEqual(organ, prompt, prompt)
            self.assertIsNone(choice, prompt)

    def test_a_calibration_object_is_left_alone(self):
        """A ruler names no organ either, and choosing one for it would be nonsense."""
        for prompt in ("ruler", "color checker", "标尺", "色卡"):
            organ, choice, _ = self.resolve(prompt, candidates=["pod", "seed"])
            self.assertEqual(organ, prompt, prompt)
            self.assertIsNone(choice, prompt)

    def test_a_weak_image_is_left_alone(self):
        probe = summary(pod=(2, 0.22), seed=(1, 0.18))

        organ, choice, _ = self.resolve("扁豆", probe=probe, candidates=["pod", "seed"])

        self.assertEqual(organ, "扁豆")
        self.assertIsNone(choice)

    def test_a_close_call_is_left_alone(self):
        """If the image cannot separate them it has no opinion worth acting on."""
        probe = summary(pod=(3, 0.41), seed=(3, 0.40))

        organ, choice, _ = self.resolve("扁豆", probe=probe, candidates=["pod", "seed"])

        self.assertEqual(organ, "扁豆")
        self.assertIsNone(choice)

    def test_unknown_candidates_are_ignored_rather_than_trusted(self):
        """The model may name anything; only the four organs can be chosen."""
        organ, choice, _ = self.resolve("扁豆", candidates=["fruit", "pod"])

        self.assertEqual(organ, "pod")
        self.assertEqual(choice["candidates"], ["pod"])

    def test_a_probe_failure_is_not_fatal(self):
        detector = TextDetector.__new__(TextDetector)

        def boom(rgb, **kwargs):
            raise RuntimeError("no GPU today")

        detector.organ_scores = boom
        organ, choice, probe = detector.resolve_organ(None, "扁豆", candidates=["pod"])

        self.assertEqual(organ, "扁豆")
        self.assertIsNone(choice)
        self.assertIsNone(probe)


if __name__ == "__main__":
    unittest.main()
