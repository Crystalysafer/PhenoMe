"""Model-backed fallback for Chinese words the dictionaries do not know.

``pheno/prompt_zh.py`` translates Chinese detection prompts from hand-written
dictionaries. That is instant and deterministic, which matters more than it sounds:
a measurement that changes between runs is not a measurement. But a dictionary
cannot hold every plant name, and an unknown term is currently *dropped*, which
silently changes what is being asked for.

This module fills that gap and nothing else:

* **Only for terms the dictionaries missed.** ``translate`` is never consulted for
  a word that is already known, so the common path stays dictionary-only.
* **One term, one answer, forever.** Every translation is written to a JSON cache
  next to the dictionaries. The model does the work once; from then on the mapping
  is fixed, reviewable and committable -- the randomness is spent, not repeated.
* **Qualifiers are stripped.** Measured on one photograph of lentils, `lentil`
  found 20 seeds while `red lentil` found 3. A translation that adds a colour or a
  size is worse than no translation, so one that arrives with a qualifier has it
  removed before use.
* **Failure is quiet.** No key, no network, a rate limit, an unusable answer: the
  caller falls back to its previous behaviour and the detection still runs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path

import httpx

log = logging.getLogger("phenome.translate")

DEFAULT_MODEL = "glm-4-flash-250414"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
TIMEOUT_SECONDS = 20.0

# Sits beside the dictionaries it extends, so the file can be read, corrected and
# committed like them.
CACHE_PATH = Path(__file__).resolve().parent.parent / "pheno" / "config" / "prompt_translations.json"

# Words that make a prompt worse. The detector matches the phrase it is given, and
# a described object is found far less reliably than a named one.
_QUALIFIERS = {
    "red", "green", "yellow", "brown", "black", "white", "purple", "pink", "orange",
    "dark", "light", "bright", "pale", "deep", "grey", "gray", "violet",
    "small", "large", "big", "little", "tiny", "huge", "long", "short", "thin", "thick",
    "young", "old", "fresh", "dry", "dried", "wet", "mature", "immature", "ripe", "raw",
    "wild", "cultivated", "healthy", "damaged", "spotted", "freshly", "slightly",
}

# The words that make a prompt groundable, because they name something with a shape
# in the frame. A prompt built only from a botanical category has none of these.
_VISIBLE_NOUNS = {
    "pod", "pods", "seed", "seeds", "leaf", "leaves", "leaflet", "flower", "flowers",
    "petal", "calyx", "stem", "branch", "root", "roots", "seedling", "tendril",
    "cotyledon", "hilum", "embryo", "thorn", "spine", "fruit", "grain", "kernel",
    "bean", "beans", "pea", "peas", "lentil", "lentils", "nut", "berry", "silique",
    "tuber", "bulb", "head", "cob", "spike", "panicle", "truss",
}

SYSTEM_PROMPT = """\
You turn one Chinese plant-phenotyping term into the English noun phrase that an
object detector should be given.

Rules:
- Output ONLY the English noun phrase. No quotes, no explanation, no Chinese.
- Use one to three words. Name the thing; do not describe it.
- NEVER add a colour, size, age or condition. 红扁豆 is `lentil`, never `red lentil`.
  A qualifier makes the detector find far fewer objects, so it is worse than useless.
- Name the object that is VISIBLE in the photograph, not the category it belongs to
  and not its scientific classification. Measured on one photograph of lentils,
  `seed` and `lentil` each found 20 while `legume` found 3 and `pulse` and
  `lens culinaris` found 1.
- Keep a common physical noun at the end -- `pod`, `seed`, `leaf`, `flower`, `thorn`,
  `root`, `stem`. A qualified physical noun is fine (`wisteria pod`); a botanical
  category is not (`fabaceae`, `grain legume`, `pulse`).
- Keep the organ when one is named: 紫藤荚 is `wisteria pod`, not `wisteria`.
- If you do not recognise the term, reply with exactly: UNKNOWN

Examples:
豆荚 -> bean pod
种子 -> seed
红扁豆 -> lentil
紫藤荚 -> wisteria pod
皂角刺 -> honey locust thorn
长角豆 -> carob pod
不认识的东西 -> UNKNOWN
"""

_cache: dict[str, str] = {}
_cache_loaded = False
_cache_lock = threading.Lock()


def enabled() -> bool:
    """Whether the fallback may be used at all.

    On by default whenever a key exists: the point of the feature is to rescue terms
    that would otherwise be dropped. Set ``PHENOME_TRANSLATE_FALLBACK=off`` to
    require dictionary-only behaviour, e.g. for a run that must reproduce exactly.
    """
    setting = (os.environ.get("PHENOME_TRANSLATE_FALLBACK") or "on").strip().lower()
    if setting in ("off", "0", "false", "no"):
        return False
    return bool(_credentials()[1])


def _credentials() -> tuple[str, str, str]:
    """Endpoint, key and model for the model-backed translation fallback.

    ``PHENOME_TRANSLATE_*`` variables win outright; whatever is left unset falls
    back to the deployment's provider variables, so the existing key file on the
    host keeps feeding this fallback with no config change.
    """
    base = (os.environ.get("PHENOME_TRANSLATE_BASE_URL") or "").strip()
    key = (os.environ.get("PHENOME_TRANSLATE_API_KEY") or "").strip()
    model = (os.environ.get("PHENOME_TRANSLATE_MODEL") or "").strip()

    base = base or (os.environ.get("PHENOME_AGENT_BASE_URL") or DEFAULT_BASE_URL).strip()
    key = (key or os.environ.get("PHENOME_AGENT_API_KEY")
           or os.environ.get("ZHIPUAI_API_KEY")
           or os.environ.get("DEEPSEEK_API_KEY") or "")
    model = model or (os.environ.get("PHENOME_AGENT_MODEL") or DEFAULT_MODEL)
    return base, key, model


def _load_cache() -> dict[str, str]:
    global _cache, _cache_loaded
    with _cache_lock:
        if not _cache_loaded:
            try:
                if CACHE_PATH.is_file():
                    loaded = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        _cache = {str(k): str(v) for k, v in loaded.items()}
            except (OSError, ValueError) as exc:
                log.warning("could not read %s: %s", CACHE_PATH.name, exc)
            _cache_loaded = True
        return dict(_cache)


def _store(chinese: str, english: str) -> None:
    """Remember a translation, on disk as well as in memory.

    Persisting is the whole point: it is what turns a one-off model guess into a
    fixed mapping that later runs -- and other people -- reproduce exactly.
    """
    global _cache
    with _cache_lock:
        _cache[chinese] = english
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(
                json.dumps(_cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("could not write %s: %s", CACHE_PATH.name, exc)


def cached() -> dict[str, str]:
    """Everything learned so far (a copy)."""
    return _load_cache()


def forget_all() -> None:
    """Test seam: drop the cache in memory and on disk."""
    global _cache, _cache_loaded
    with _cache_lock:
        _cache = {}
        _cache_loaded = True
        try:
            CACHE_PATH.unlink(missing_ok=True)
        except OSError:
            pass


def strip_qualifiers(phrase: str) -> str:
    """Remove colour/size/condition words from a translation.

    If that empties the phrase the original is kept: a phrase made only of
    qualifiers is odd, but it is still better than nothing.
    """
    words = [word for word in phrase.split() if word.strip(",.();").lower() not in _QUALIFIERS]
    return " ".join(words) if words else phrase


def clean_answer(raw: str) -> str | None:
    """Turn a model reply into a usable prompt, or ``None`` if it is not one."""
    text = (raw or "").strip()
    if not text:
        return None
    # Models like to wrap answers in quotes, bullets or a sentence. Only the ends
    # are trimmed, so a hyphen inside a word ("honey-locust") survives.
    text = text.splitlines()[0].strip().strip("\"'`*_ .-•·")
    if not text or text.upper() == "UNKNOWN":
        return None
    if re.search(r"[\u4e00-\u9fff]", text):
        return None  # a Chinese answer is no use to the detector
    if len(text.split()) > 4:
        return None  # a sentence, not a term
    text = strip_qualifiers(text)

    # A botanical category instead of a visible object. `legume` found 3 lentils
    # where `lentil` found 20, so this is a quality warning rather than a rejection:
    # the answer is still used, but it is visible in the log if detections come back
    # thin. Guessing an organ to append would be worse -- the term may be a pod.
    if text and not any(word in _VISIBLE_NOUNS for word in text.lower().split()):
        log.info("translation %r names no visible object; it may ground poorly", text)

    return text.lower() or None


def _ask_model(chinese: str) -> str | None:
    base, key, model = _credentials()
    if not key:
        return None
    url = f"{base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chinese},
        ],
        # Translation has one right answer; sampling only adds variance.
        "temperature": 0,
        "max_tokens": 24,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        response = client.post(url, json=payload, headers=headers)
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:160]}")
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        return None
    return (choices[0].get("message") or {}).get("content") or ""


def translate(text: str) -> str | None:
    """Chinese term -> English prompt, or ``None`` to leave it alone.

    Consulted only for fragments the dictionaries could not fully translate.
    """
    term = (text or "").strip()
    if not term:
        return None

    known = _load_cache().get(term)
    if known:
        return known

    if not enabled():
        return None

    try:
        answer = clean_answer(_ask_model(term))
    except Exception as exc:  # noqa: BLE001 - the detection must not depend on this
        log.warning("fallback translation of %r failed: %s", term, exc)
        return None

    if not answer:
        log.info("fallback translation of %r produced nothing usable", term)
        return None

    log.info("fallback translation: %r -> %r", term, answer)
    _store(term, answer)
    return answer
