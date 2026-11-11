import re
import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Chinese prompt vocabulary. Imported at module level (it is pure data, no torch,
# no network) so both front ends get the same behaviour from one place.
from pheno.prompt_zh import has_chinese as prompt_has_chinese
from pheno.prompt_zh import split_conjunctions as prompt_split_conjunctions
from pheno.prompt_zh import translate as prompt_translate_zh

log = logging.getLogger("phenome.detect")


class TextDetector:
    MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    BASE_CATEGORIES = ("seed", "pod", "leaf", "flower")
    PLURAL_FORMS = {
        "seeds": "seed",
        "pods": "pod",
        "leaves": "leaf",
        "leafs": "leaf",
        "flowers": "flower",
    }
    # Domain terms the user is likely to type, mapped to wording Grounding
    # DINO can actually ground. The model is trained on general-purpose
    # captions, so botanical vocabulary ("panicle", "silique", "caryopsis")
    # has almost no visual grounding: measured on real rice-field photos,
    # "panicle" missed every spike and instead scored the colour checker
    # highest, while "rice spike" hit the spike at 0.61-0.70. Adding a
    # species qualifier also narrows the model's attention markedly
    # ("spike" gave 4 boxes on one image, "rice spike" gave 1, the right one).
    PROMPT_ALIASES = {
        "seed": "bean seed",
        "pod": "bean pod",
        "leaf": "bean leaf",
        "flower": "bean flower",
        # --- cereal inflorescence: "panicle" is the standard term for a rice
        # head but is invisible to the model, so ask for "rice spike".
        "panicle": "rice spike",
        "panicles": "rice spike",
        "rice panicle": "rice spike",
        "spike": "rice spike",
        "spikes": "rice spike",
        "spike of rice": "rice spike",
        "ear": "rice spike",
        "ear of rice": "rice spike",
        "rice ear": "rice spike",
        "head": "rice spike",
        "rice head": "rice spike",

        # --- other botanical terms with poor grounding
        "silique": "pod",
        "siliques": "pod",
        "legume": "bean pod",
        "caryopsis": "grain",
        "grain": "rice grain",
        "spikelet": "rice grain",
        "spikelets": "rice grain",
    }

    # Chinese is deliberately absent from this table: pheno/prompt_zh.py handles it
    # with longest-match scanning, which composes ("绿色豆荚") where a flat lookup
    # only matched whole strings. Entries here would shadow that and reintroduce the
    # inconsistency where 叶片 gave "leaf" while 叶 gave "bean leaf".

    # Calibration and staging objects that share the frame with the sample.
    # They are detected alongside the user's target purely so their boxes can
    # veto overlapping target boxes; they never become objects themselves.
    # "hand" is included because a hand holding the sample is common and the
    # model happily labels it as the target.
    DISTRACTOR_PROMPT = (
        "color checker. color chart. ruler. label. paper card. hand."
    )
    # A distractor box covering more than this fraction of the frame is the
    # model framing the whole photo, not an object; ignoring those keeps the
    # veto from wiping out every real detection.
    DISTRACTOR_MAX_FRAME_RATIO = 0.5
    # How much of a target box a distractor must cover to veto it.
    DISTRACTOR_COVER_RATIO = 0.60
    # Absolute confidence a distractor phrase needs before it may veto.
    # Scores from different prompts are NOT comparable: a word the model
    # cannot ground ("panicle") faces no competing concept and can score
    # spuriously high - measured 0.595 on a colour checker, above the
    # "color chart" score of 0.462 for the same region. So the veto keys off
    # the distractor's own confidence rather than a score comparison.
    DISTRACTOR_MIN_SCORE = 0.35
    # Floors for resolve_organ(), which picks an organ for a prompt that named none.
    #
    # These are NOT the single-word numbers and must not be tuned against them: the probe
    # asks for all four categories at once and scores far lower. Measured on one bean
    # photograph, the probe gave pod 0.449 (three boxes, the three pods), seed 0.333,
    # flower 0.302 and leaf 0.000, while a single-word detection of "bean pod" on the same
    # picture scored 0.720. What ranks correctly is `best` -- not the count, which put
    # flower above pod (four boxes against three) on an image with no flowers in it.
    #
    # So: a low absolute floor, plus the winner having to actually beat the runner-up. A
    # weak image scores every category alike and gets handed to the detector with the
    # caller's original wording, which is the safe outcome.
    MIN_ORGAN_SCORE = 0.30
    MIN_ORGAN_MARGIN = 0.05

    #: Below this margin the four-organ probe is not trusted to choose, and the candidates
    #: are compared by single-organ detection instead. Both measured margins fall under it:
    #: 0.116 on a photograph of pods, 0.054 on a photograph of seeds, and the second is a
    #: coin flip -- which is how a seed image came back labelled as pods.
    PROBE_TIE_MARGIN = 0.15
    # ...unless the target wins by this much, which means the model is far
    # more certain about the sample than about the calibration object.
    DISTRACTOR_DOMINANCE_MARGIN = 0.25
    # A distractor this many times larger than the target is a *surface the
    # sample rests on* - a light-box back panel, a paper card, a tray - not an
    # object competing for the same pixels. Measured on cut-leaf photographs:
    # the lit light-box panel scored "paper card" 0.41 while the three leaf
    # blades on it scored 0.48-0.56, so the veto deleted all three blades on
    # every image of that set, and no prompt wording could clear the bar (the
    # target would have needed >0.66). A sample lying on a surface is the
    # normal case, so a distractor that dwarfs the target never vetoes it.
    DISTRACTOR_MAX_SIZE_RATIO = 3.0
    # When the target and the distractor are *both* slim bars, the two labels
    # are shape-ambiguous rather than competing for the same pixels: a detached
    # leaf blade and a straight ruler are the same shape, so the model has no
    # way to tell them apart. Measured on one cut-leaf photograph, DINO boxed
    # all three blades as "ruler" (0.33-0.36) with boxes matching the "rice
    # leaf" boxes (0.42-0.46) one for one, and two of the three blades survived
    # only because their ruler score missed DISTRACTOR_MIN_SCORE by 0.009.
    # Vetoing on such a label is a coin flip, so the higher-scoring label wins
    # here and the mask-level size/shape filter still removes real junk after.
    DISTRACTOR_SLIM_ASPECT = 3.0
    # Words that make the user's prompt itself a calibration object, in which
    # case the veto must stand down - detecting the checker or the ruler on
    # purpose is a supported use (see the 色卡 / 标尺 aliases).
    DISTRACTOR_WORDS = (
        "color checker", "colour checker", "color chart", "colour chart",
        "checker", "chart", "ruler", "scale bar", "label", "tag",
        "paper card", "card", "hand", "finger",
    )

    @classmethod
    def prompt_is_distractor(cls, prompt):
        """True when the user is deliberately asking for a calibration object."""
        for _raw, normalized in cls.split_prompt_pairs(prompt):
            lowered = normalized.strip().lower().rstrip(".")
            if not lowered:
                continue
            for word in cls.DISTRACTOR_WORDS:
                if word in lowered or lowered in word:
                    return True
        return False
    # A target box this large is the model boxing the whole scene instead of
    # an object. Measured on rice photos: real spikes covered 3-13% of the
    # frame while the junk detection covered 87%. Only dropped when a smaller
    # box also survived, so a legitimate close-up stays the sole result.
    FRAME_BOX_RATIO = 0.60

    def __init__(self, device="cpu", model_dir=None):
        project_root = Path(__file__).resolve().parent.parent
        self.model_dir = Path(model_dir or project_root / "models" / "grounding-dino-tiny")
        self.device = torch.device(device if str(device).startswith("cuda") else "cpu")
        self.processor = None
        self.model = None
        # Distractor boxes depend only on the image, so a second detection on
        # the same frame (another prompt, another batch step) reuses them.
        self._distractor_key = None
        self._distractor_cache = []

    @staticmethod
    def normalize_prompt(prompt):
        parts = [part for part in TextDetector.split_prompt(prompt) if part.strip()]
        if not parts:
            # Reached when every fragment was Chinese that the vocabulary could not
            # place. Sending that to the model would produce a confident-looking
            # detection of nothing in particular, so say so instead.
            if prompt_has_chinese(prompt):
                raise ValueError(
                    "无法识别这个中文描述，请换个说法（例如：豆荚、种子、叶片、绿色豆荚），"
                    "或直接输入英文。"
                )
            raise ValueError("Please enter a text prompt.")
        return ". ".join(parts) + "."

    def _ensure_model_files(self):
        required_files = (
            self.model_dir / "config.json",
            self.model_dir / "preprocessor_config.json",
            self.model_dir / "tokenizer_config.json",
        )
        has_weights = any(
            (self.model_dir / filename).exists()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        has_tokenizer = any(
            (self.model_dir / filename).exists()
            for filename in ("tokenizer.json", "vocab.txt")
        )
        if has_weights and has_tokenizer and all(path.exists() for path in required_files):
            return

        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=self.MODEL_ID,
                local_dir=str(self.model_dir),
            )
        except Exception as error:
            missing = [path.name for path in required_files if not path.exists()]
            if not has_tokenizer:
                missing.append("tokenizer.json or vocab.txt")
            raise RuntimeError(
                "Grounding DINO Tiny is not available locally and could not be "
                "downloaded. Missing: %s. Check the network or VPN, then run "
                "Semantic Detect again." % ", ".join(missing)
            ) from error

    def _load_model(self):
        if self.model is not None:
            return

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        # HF's GroundingDino module JIT-compiles its deformable-attention
        # CUDA kernel whenever ninja is on PATH. In the packaged app no MSVC
        # toolchain exists, so every attention layer retries the build and
        # dies spawning console windows. Force the pure-PyTorch fallback the
        # upstream try/except already selects on failure.
        try:
            from transformers.models.grounding_dino import modeling_grounding_dino

            modeling_grounding_dino.is_ninja_available = lambda: False
        except Exception:
            pass

        self._ensure_model_files()
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(self.model_dir),
            torch_dtype=model_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def split_prompt(prompt):
        """Split a raw user prompt into its individual normalized concepts."""
        return [normalized for _raw, normalized in TextDetector.split_prompt_pairs(prompt)]

    @staticmethod
    def split_prompt_pairs(prompt):
        """Split a prompt into ``(as typed, normalized)`` concept pairs.

        The raw half is what the user actually typed and is what object names
        are built from; the normalized half is what the model is asked about.
        Alias lookup is case-insensitive so "Panicle" behaves like "panicle".

        Chinese is supported here rather than in the callers, so the desktop app
        and the web service get it from the same place. Order matters: the
        Chinese-to-English table produces plain words ("pod"), and PROMPT_ALIASES
        then adds the qualifier the model actually grounds ("bean pod").
        """
        pairs = []
        text = prompt_split_conjunctions(prompt)
        for part in re.split(r"[,;，；。\s]*[,;，；。][,;，；。\s]*", text):
            part = part.strip()
            if not part:
                continue
            alias = TextDetector.PROMPT_ALIASES.get(part)
            if alias is None:
                alias = TextDetector.PROMPT_ALIASES.get(part.lower())
            if alias is None and prompt_has_chinese(part):
                english = prompt_translate_zh(part)
                # A second pass through the alias table: the translation emits
                # bare words, and "pod" alone grounds far worse than "bean pod".
                alias = TextDetector.PROMPT_ALIASES.get(english.lower(), english)
            if alias is None:
                alias = part
            pairs.append((part, alias))
        return pairs

    @staticmethod
    def alias_notes(prompt):
        """``[(as typed, normalized), ...]`` for fragments an alias rewrote.

        Used to tell the user their wording was translated for the model, so
        a surprising result is explainable rather than mysterious.
        """
        notes = []
        for raw, normalized in TextDetector.split_prompt_pairs(prompt):
            if normalized.strip().lower() != raw.strip().lower():
                notes.append((raw, normalized))
        return notes

    @staticmethod
    def term_for_label(label, pairs):
        """Map a model label back to the concept the user typed.

        Falls back to the label itself when no recorded concept matches, and to
        the single requested concept when the label is unusable.
        """
        text = str(label or "").strip().rstrip(".")
        lowered = text.lower()
        if lowered:
            for raw, normalized in pairs:
                if lowered == normalized.strip().lower():
                    return raw
            for raw, normalized in pairs:
                candidate = normalized.strip().lower()
                if candidate and (candidate in lowered or lowered in candidate):
                    return raw
            for raw, _normalized in pairs:
                candidate = raw.strip().lower()
                if candidate and (candidate in lowered or lowered in candidate):
                    return raw
        if text:
            return text
        return pairs[0][0] if len(pairs) == 1 else ""

    @staticmethod
    def _category_of(text):
        """Map a prompt fragment or model label to one of BASE_CATEGORIES."""
        lowered = str(text).strip().lower().rstrip(".")
        if not lowered:
            return None
        words = re.findall(r"[a-z]+", lowered)
        for word in words:
            word = TextDetector.PLURAL_FORMS.get(word, word)
            if word in TextDetector.BASE_CATEGORIES:
                return word
        return None

    @classmethod
    def target_categories(cls, prompt):
        """Base categories the user asked for; empty when not comparable."""
        categories = []
        for part in cls.split_prompt(prompt):
            category = cls._category_of(part)
            if category and category not in categories:
                categories.append(category)
        return categories

    @staticmethod
    def summarize(detections):
        """Group detections into per-base-category count/score statistics."""
        summary = {}
        for detection in detections:
            category = TextDetector._category_of(detection.get("label", ""))
            if category is None:
                continue
            entry = summary.setdefault(
                category, {"count": 0, "best": 0.0, "scores": []}
            )
            score = float(detection.get("score", 0.0))
            entry["count"] += 1
            entry["scores"].append(score)
            entry["best"] = max(entry["best"], score)
        return summary

    # rival_min_score was 0.35, which sits inside this model's noise band on
    # real photographs. Asking "seed, pod, leaf, flower" about a cut-leaf image
    # returned pod 0.32-0.40 and nothing at all for leaf, while the same image
    # answered a direct "rice leaf" with genuine boxes at 0.43-0.56 - so a
    # correct result was reported as a prompt/image mismatch and no object was
    # created. Real groundings on these images start around 0.43, so the rival
    # bar now sits just above the noise: a rival must reach 0.45 to be believed.
    def organ_scores(self, rgb_image, box_threshold=0.25, text_threshold=0.20):
        """Per-organ count and best score, from one probe inference.

        The probe asks for all four base categories in a single pass, so this costs one
        inference rather than four. ``cross_check`` already made exactly this call, which
        is why it is a method now: the choice :meth:`resolve_organ` makes and the warning
        :meth:`cross_check` raises have to agree about what the image contains.
        """
        detections = self.detect(
            rgb_image,
            ", ".join(self.BASE_CATEGORIES),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            # This probe only asks "what is in this image at all"; running the
            # distractor veto here would cost a second inference and skew the
            # category statistics it is about to compute.
            reject_distractors=False,
        )
        return self.summarize(detections)

    def is_calibration_prompt(self, prompt) -> bool:
        """True for the objects that exist to be measured against.

        A ruler and a colour checker name no organ either, but choosing one for them would
        be nonsense: the caller asked for a ruler, and the answer is a ruler or nothing.
        """
        text = str(prompt or "").strip().lower()
        if not text:
            return False
        for phrase in self.DISTRACTOR_PROMPT.split("."):
            phrase = phrase.strip()
            if phrase and (phrase in text or text in phrase):
                return True
        return False

    def resolve_organ(self, rgb_image, prompt, candidates=None, **probe_kwargs):
        """Replace a prompt that names no organ with the one this image scores highest.

        Two things know something the other does not. The caller's language model knows
        what a word can *mean* -- 扁豆 means the pod or the seed, never the flower -- and
        the image knows what it is actually *showing*. So the model narrows the field with
        ``candidates`` and the four-organ probe decides inside it.

        The narrowing matters more than it looks. Measured on one photograph of three pods,
        the unrestricted probe put a **flower** first on box count (4 boxes against the
        pods' 3) and only lost on score (0.302 against 0.449). Right answer, wrong reason:
        on another photograph the same noise could win outright.

        Returns ``(prompt, choice, summary)``. ``choice`` is ``None`` -- and the original
        prompt is returned unchanged -- when it already names an organ, when it is a
        calibration object, when the image scores nothing convincingly, or when the probe
        fails. In all of those the caller carries on and the ordinary mismatch path still
        has its say. ``summary`` is the probe result, so a caller that goes on to
        :meth:`cross_check` does not pay for the same inference twice.
        """
        if self.target_categories(prompt):
            return prompt, None, None

        english = prompt_translate_zh(prompt) or prompt
        if self.is_calibration_prompt(english):
            return prompt, None, None

        # Only the four organs can be chosen, whatever the caller suggests.
        wanted = [name for name in (candidates or ()) if name in self.BASE_CATEGORIES]

        try:
            summary = self.organ_scores(rgb_image, **probe_kwargs)
        except Exception as exc:  # noqa: BLE001 - a probe must not block the detection
            log.warning("organ probe failed, using the prompt as given: %r", exc)
            return prompt, None, None

        considered = {
            name: stats for name, stats in summary.items() if not wanted or name in wanted
        }
        ranked = sorted(
            considered.items(),
            key=lambda item: (
                -float(item[1].get("best", 0.0)),
                -int(item[1].get("count", 0)),
            ),
        )
        if not ranked:
            return prompt, None, summary

        organ, stats = ranked[0]
        best = float(stats.get("best", 0.0))
        if not stats.get("count") or best < self.MIN_ORGAN_SCORE:
            # Nothing convincing enough to act on. Better to hand the original prompt to
            # the detector than to pick an organ out of noise.
            return prompt, None, summary

        runner_up = float(ranked[1][1].get("best", 0.0)) if len(ranked) > 1 else 0.0
        separated_by = "four-organ probe"
        others = {
            name: round(float(value.get("best", 0.0)), 3) for name, value in ranked[1:]
        }
        if best - runner_up < self.PROBE_TIE_MARGIN:
            # Too close to call. The probe asks for all four organs at once and each gets a
            # share of the model's attention; asked one at a time they score far higher and
            # further apart, which is worth two extra inferences when the choice is 50/50.
            decided = self._separate_by_detection(
                rgb_image, [name for name, _ in ranked]
            )
            if decided is None:
                # The tie-break could not run either. Leave the prompt alone rather than
                # pick on noise.
                return prompt, None, summary
            organ, best, count, others = decided
            separated_by = "single-organ detection"
            if best < self.MIN_ORGAN_SCORE:
                return prompt, None, summary
            stats = {"best": best, "count": count}

        choice = {
            "from": prompt,
            "organ": organ,
            "score": round(float(stats["best"]), 3),
            "count": int(stats["count"]),
            "others": others,
            "separated_by": separated_by,
        }
        if wanted:
            choice["candidates"] = wanted
        return organ, choice, summary

    def _separate_by_detection(self, rgb_image, organs):
        """One single-organ detection per candidate; the strongest wins.

        ``(organ, best, count, others)``, or ``None`` when no candidate could be detected.

        The four-organ probe asks for everything at once, so each category gets a share of
        the model's attention and the scores sit close to the noise band -- measured, pod
        0.449 against seed 0.333 on a photograph of pods, and seed 0.351 against pod 0.297 on
        a photograph of seeds, where a 0.05 margin decided it. Asked one at a time the same
        organs score much higher and separate cleanly (``bean pod`` 0.720), which is what
        makes this worth two inferences when the probe cannot choose.
        """
        scored = []
        for organ in organs:
            try:
                detections = self.detect(rgb_image, organ)
            except Exception as exc:  # noqa: BLE001 - fall back to the probe's order
                log.warning("tie-break detection for %r failed: %r", organ, exc)
                continue
            best = max((float(d.get("score", 0.0)) for d in detections), default=0.0)
            scored.append((best, len(detections), organ))
        if not scored:
            return None

        scored.sort(reverse=True)
        best, count, organ = scored[0]
        others = {name: round(value, 3) for value, _n, name in scored[1:]}
        return organ, best, count, others

    def cross_check(
        self,
        rgb_image,
        prompt,
        box_threshold=0.25,
        text_threshold=0.20,
        rival_min_count=2,
        rival_min_score=0.45,
        dominance_ratio=0.60,
        target_min_score=0.35,
        summary=None,
    ):
        """Compare the requested category against the other base categories.

        Returns ``None`` when the prompt is not comparable (ruler, color
        checker, custom wording) or when the result looks plausible. Otherwise
        a dict describing the mismatch is returned so the caller can warn the
        user before any object is created.

        Pass ``summary`` to reuse a probe :meth:`resolve_organ` has already run.
        """
        targets = self.target_categories(prompt)
        if not targets:
            return None

        if summary is None:
            summary = self.organ_scores(
                rgb_image, box_threshold=box_threshold, text_threshold=text_threshold
            )
        if not summary:
            # Nothing recognizable at all: this is a weak image rather than a
            # wrong prompt, so do not raise a mismatch warning.
            return None

        target_best = max(
            (summary.get(target, {}).get("best", 0.0) for target in targets),
            default=0.0,
        )
        target_count = sum(
            summary.get(target, {}).get("count", 0) for target in targets
        )

        if target_count and target_best >= target_min_score:
            # The requested category is found with decent confidence, so even a
            # stronger rival category is not enough to call this a mismatch.
            return None

        rivals = [
            (category, stats)
            for category, stats in summary.items()
            if category not in targets
        ]
        if not rivals:
            return None

        rival_category, rival_stats = max(
            rivals, key=lambda item: (item[1]["best"], item[1]["count"])
        )
        if (
            rival_stats["count"] < rival_min_count
            or rival_stats["best"] < rival_min_score
        ):
            return None

        if target_count == 0 or target_best < dominance_ratio * rival_stats["best"]:
            return {
                "requested": targets,
                "requested_count": target_count,
                "requested_score": target_best,
                "suggested": rival_category,
                "suggested_count": rival_stats["count"],
                "suggested_score": rival_stats["best"],
                "summary": summary,
            }

        return None

    def detect(self, rgb_image, prompt, box_threshold=0.30, text_threshold=0.25,
               reject_distractors=True):
        self._load_model()
        detections = self._detect_raw(
            rgb_image, self.normalize_prompt(prompt),
            box_threshold, text_threshold,
        )
        if reject_distractors and detections \
                and not self.prompt_is_distractor(prompt):
            detections = self.drop_frame_boxes(rgb_image, detections)
            detections, _vetoed = self.reject_distractors(
                rgb_image, detections, box_threshold, text_threshold)
        return detections

    def detect_with_report(self, rgb_image, prompt, box_threshold=0.30,
                           text_threshold=0.25):
        """Like :meth:`detect` but also returns what was filtered out."""
        self._load_model()
        detections = self._detect_raw(
            rgb_image, self.normalize_prompt(prompt),
            box_threshold, text_threshold,
        )
        if not detections:
            return [], []
        detections, oversized = self.drop_frame_boxes(
            rgb_image, detections, report=True)
        if self.prompt_is_distractor(prompt):
            # The calibration object *is* the target here.
            return detections, list(oversized)
        kept, vetoed = self.reject_distractors(
            rgb_image, detections, box_threshold, text_threshold)
        return kept, list(oversized) + list(vetoed)

    def drop_frame_boxes(self, rgb_image, detections, report=False):
        """Remove boxes that cover most of the frame when smaller ones exist.

        Grounding DINO sometimes answers "the whole photo" for a prompt it
        cannot localise. Such a box is useless for phenotyping and, once
        segmented, produces one giant object swallowing the real ones.
        """
        height, width = rgb_image.shape[:2]
        frame_area = float(max(height * width, 1))
        oversized, normal = [], []
        for detection in detections:
            ratio = self._box_area(detection["box"]) / frame_area
            (oversized if ratio >= self.FRAME_BOX_RATIO else normal).append(
                (detection, ratio))

        # Keep the oversized ones only when nothing smaller survived, so a
        # genuine close-up is not filtered down to nothing.
        if not normal:
            kept = [item for item, _ratio in oversized]
            return (kept, []) if report else kept

        kept = [item for item, _ratio in normal]
        dropped = []
        for item, ratio in oversized:
            entry = dict(item)
            entry["vetoed_by"] = "whole frame"
            entry["vetoed_score"] = ratio
            dropped.append(entry)
        return (kept, dropped) if report else kept

    @staticmethod
    def _image_fingerprint(rgb_image):
        """Cheap content key: shape plus a strided pixel sample.

        A full md5 of 24 MP costs ~150 ms; sampling every 97th pixel costs
        ~2 ms and still distinguishes one photo from the next for caching.
        """
        height, width = rgb_image.shape[:2]
        sample = rgb_image[::97, ::97]
        return (height, width, hashlib.md5(sample.tobytes()).hexdigest())

    def reject_distractors(self, rgb_image, detections,
                           box_threshold=0.30, text_threshold=0.25):
        """Drop target boxes that a better-scoring calibration object covers.

        Photos of a sample almost always include a colour checker, a ruler, a
        label card or the photographer's hand. Grounding DINO readily labels
        those as the requested plant part - measured on real rice images,
        "panicle" scored the colour checker 0.595 while missing the spike
        entirely. Scores from different prompts are not directly comparable in
        general, but a distractor phrase scoring higher than the target phrase
        on the very same region is strong evidence the region is the
        distractor, so that region's target boxes are removed.

        Returns ``(kept, vetoed)``.
        """
        key = self._image_fingerprint(rgb_image)
        if self._distractor_key != key:
            self._distractor_key = key
            self._distractor_cache = self._detect_raw(
                rgb_image, self.DISTRACTOR_PROMPT, box_threshold, text_threshold)
        distractors = self._distractor_cache
        if not distractors:
            return list(detections), []

        height, width = rgb_image.shape[:2]
        frame_area = float(max(height * width, 1))
        usable = [
            item for item in distractors
            if self._box_area(item["box"]) / frame_area
            <= self.DISTRACTOR_MAX_FRAME_RATIO
            and item["score"] >= self.DISTRACTOR_MIN_SCORE
        ]
        if not usable:
            return list(detections), []

        kept, vetoed = [], []
        for detection in detections:
            veto = None
            for distractor in usable:
                covered = self._cover_ratio(detection["box"], distractor["box"])
                if covered < self.DISTRACTOR_COVER_RATIO:
                    continue
                # A distractor far larger than the target is the surface the
                # sample lies on, not a rival for the same pixels.
                if self._box_area(distractor["box"]) > (
                        self.DISTRACTOR_MAX_SIZE_RATIO
                        * self._box_area(detection["box"])):
                    continue
                # Both boxes slim bars: shape ambiguity between two labels for
                # one object (leaf blade vs ruler), not a calibration object
                # covering the sample. See DISTRACTOR_SLIM_ASPECT.
                if (self._box_aspect(detection["box"]) >= self.DISTRACTOR_SLIM_ASPECT
                        and self._box_aspect(distractor["box"])
                        >= self.DISTRACTOR_SLIM_ASPECT):
                    continue
                # Let a target through only when it dominates decisively.
                if detection["score"] >= (distractor["score"]
                                          + self.DISTRACTOR_DOMINANCE_MARGIN):
                    continue
                veto = distractor
                break
            if veto is None:
                kept.append(detection)
            else:
                item = dict(detection)
                item["vetoed_by"] = str(veto["label"])
                item["vetoed_score"] = float(veto["score"])
                vetoed.append(item)
        return kept, vetoed

    @staticmethod
    def _box_area(box):
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _box_aspect(box):
        """Longer side over shorter side of an XYXY box."""
        x1, y1, x2, y2 = box
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return max(width, height) / max(min(width, height), 1e-6)

    @staticmethod
    def _cover_ratio(inner, outer):
        """Fraction of ``inner``'s area that lies inside ``outer``."""
        ax1, ay1, ax2, ay2 = inner
        bx1, by1, bx2, by2 = outer
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area = TextDetector._box_area(inner)
        return intersection / area if area > 0 else 0.0

    def _detect_raw(self, rgb_image, normalized_prompt,
                    box_threshold=0.30, text_threshold=0.25):
        self._load_model()
        pil_image = Image.fromarray(np.ascontiguousarray(rgb_image))
        inputs = self.processor(
            images=pil_image,
            text=normalized_prompt,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        result = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        boxes = result["boxes"].detach().cpu()
        scores = result["scores"].detach().cpu()
        labels = (
            result["text_labels"]
            if "text_labels" in result
            else result.get("labels", [])
        )
        if len(boxes) == 0:
            return []
        if len(labels) != len(boxes):
            labels = [normalized_prompt.rstrip(".")] * len(boxes)

        from torchvision.ops import nms

        keep = []
        label_groups = {}
        for index, label in enumerate(labels):
            label_groups.setdefault(str(label), []).append(index)
        for indices in label_groups.values():
            index_tensor = torch.tensor(indices, dtype=torch.long)
            kept_indices = nms(boxes[index_tensor], scores[index_tensor], 0.50)
            keep.extend(index_tensor[kept_indices].tolist())
        keep.sort(key=lambda index: float(scores[index]), reverse=True)
        height, width = rgb_image.shape[:2]
        detections = []
        for index in keep:
            x1, y1, x2, y2 = boxes[index].tolist()
            box = (
                max(0.0, min(float(width - 1), x1)),
                max(0.0, min(float(height - 1), y1)),
                max(0.0, min(float(width - 1), x2)),
                max(0.0, min(float(height - 1), y2)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            label = labels[index] if index < len(labels) else normalized_prompt.rstrip(".")
            if hasattr(label, "item"):
                label = label.item()
            detections.append(
                {
                    "box": box,
                    "score": float(scores[index]),
                    "label": str(label),
                }
            )

        return detections
