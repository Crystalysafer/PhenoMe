"""Chinese prompt support for semantic detection.

Grounding DINO only understands English, so a Chinese prompt used to fail outright
unless it happened to match one of `TextDetector.PROMPT_ALIASES` exactly. That
made the feature unusable for the people it is for: agronomists type 绿色豆荚, not
"green bean pod".

This module translates a Chinese fragment into plain English by greedy
longest-match scanning -- no tokenizer, no network, no model. The output is
deliberately *plain* ("pod", "seed"): `TextDetector.PROMPT_ALIASES` then adds the
qualifiers the model actually grounds well ("bean pod"), so the two stages compose
instead of fighting. That is also why crop names translate to a bare species name:
蚕豆荚 becomes "broad bean pod" and never "broad bean bean pod".

Anything already in English passes through untouched, and a mixed fragment like
"green 豆荚" works too.
"""

from __future__ import annotations

import re

# Fragment separators that mean "and also": the caller splits on these so that
# 豆荚和种子 asks about two concepts instead of one nonsense phrase.
CONJUNCTIONS = ("以及", "还有", "并且", "和", "与", "及", "跟", "、")

# Words that carry no visual meaning and would only confuse the model.
NOISE = (
    "的", "地", "得", "了", "着", "过", "是", "有", "所有", "全部", "每个", "每一",
    "这个", "那个", "这些", "那些", "一些", "若干", "多个", "几个", "各种", "各类",
    "个", "只", "颗", "粒", "条", "根", "片", "朵", "株", "枚", "块", "把", "支",
    "张", "份", "组", "对", "双", "串", "堆", "排", "行", "列", "层",
    "请", "帮我", "检测", "识别", "找出", "找到", "圈出", "分割", "标出", "选出",
    "图中", "图片", "照片", "里的", "上的", "中的", "里面", "上面", "中间",
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "两",
)

# Crop names. Bare species on purpose -- the organ word supplies "pod"/"seed" and
# TextDetector's aliases supply "bean" when no species was named.
CROPS = {
    "蚕豆": "broad bean",
    "胡豆": "broad bean",
    "扁豆": "hyacinth bean",
    "鹊豆": "hyacinth bean",
    "peas": "pea",
    "豌豆": "pea",
    "荷兰豆": "snow pea",
    "大豆": "soybean",
    "黄豆": "soybean",
    "毛豆": "edamame",
    "菜豆": "common bean",
    "四季豆": "common bean",
    "芸豆": "kidney bean",
    "绿豆": "mung bean",
    "红豆": "adzuki bean",
    "赤豆": "adzuki bean",
    "黑豆": "black soybean",
    "豇豆": "cowpea",
    "长豆角": "yardlong bean",
    "鹰嘴豆": "chickpea",
    "小豆": "adzuki bean",
    "花生": "peanut",
    "水稻": "rice",
    "稻": "rice",
    "大米": "rice",
    "小麦": "wheat",
    "麦": "wheat",
    "玉米": "maize",
    "油菜": "rapeseed",
    "棉花": "cotton",
    "番茄": "tomato",
    "西红柿": "tomato",
    "辣椒": "chili pepper",
    "黄瓜": "cucumber",
    "茄子": "eggplant",
    "土豆": "potato",
    "马铃薯": "potato",
    "栀子": "gardenia",
    "枸杞": "goji berry",
    "油茶": "camellia",
}

# Plant parts. These are what the sample usually is.
ORGANS = {
    "豆荚": "pod",
    "荚果": "pod",
    "荚壳": "pod shell",
    "豆角": "pod",
    "荚": "pod",
    "豆粒": "seed",
    "籽粒": "grain",
    "谷粒": "grain",
    "米粒": "rice grain",
    "稻粒": "rice grain",
    "玉米粒": "maize kernel",
    "麦粒": "wheat grain",
    "花生粒": "peanut kernel",
    "花生仁": "peanut kernel",
    "种子": "seed",
    "籽": "seed",
    "豆子": "seed",
    "豆": "seed",
    "种皮": "seed coat",
    "脐": "hilum",
    "胚": "embryo",
    "胚芽": "embryo",
    "子叶": "cotyledon",
    "叶片": "leaf",
    "叶子": "leaf",
    "叶": "leaf",
    "小叶": "leaflet",
    "复叶": "compound leaf",
    "托叶": "stipule",
    "花朵": "flower",
    "花序": "inflorescence",
    "花瓣": "petal",
    "花萼": "calyx",
    "花": "flower",
    "穗": "spike",
    "稻穗": "rice spike",
    "麦穗": "wheat spike",
    "茎": "stem",
    "主茎": "main stem",
    "枝条": "branch",
    "枝": "branch",
    "叶柄": "petiole",
    "花梗": "pedicel",
    "果柄": "fruit stalk",
    "根": "root",
    "根系": "root system",
    "果实": "fruit",
    "果": "fruit",
    "块茎": "tuber",
    "果穗": "ear",
    "幼苗": "seedling",
    "苗": "seedling",
    "芽": "bud",
    "卷须": "tendril",
    "植株": "plant",
    "整株": "whole plant",
}

# Reference and staging objects that share the frame with the sample.
TOOLS = {
    "直尺": "ruler",
    "钢尺": "steel ruler",
    "卷尺": "tape measure",
    "刻度尺": "ruler",
    "标尺": "ruler",
    "尺子": "ruler",
    "尺": "ruler",
    "比例尺": "scale bar",
    "比色卡": "color checker",
    "标准色卡": "color checker",
    "色卡": "color checker",
    "色板": "color chart",
    "灰卡": "gray card",
    "白平衡卡": "white balance card",
    "标签牌": "label",
    "标签": "label",
    "吊牌": "tag",
    "标牌": "label",
    "纸牌": "paper card",
    "卡片": "card",
    "纸片": "paper card",
    "纸": "paper",
    "硬币": "coin",
    "托盘": "tray",
    "盘子": "plate",
    "培养皿": "petri dish",
    "镊子": "tweezers",
    "手": "hand",
    "手指": "finger",
    "背景板": "background board",
    "黑布": "black cloth",
    "绒布": "cloth",
    "网格": "grid",
    "二维码": "qr code",
    "笔": "pen",
}

# Colours, including the compounds an agronomist actually writes.
COLORS = {
    "深绿": "dark green",
    "浅绿": "light green",
    "淡绿": "light green",
    "黄绿": "yellow green",
    "青绿": "green",
    "墨绿": "dark green",
    "绿色": "green",
    "绿": "green",
    "青": "green",
    "深红": "dark red",
    "浅红": "light red",
    "紫红": "purple red",
    "粉红": "pink",
    "粉色": "pink",
    "红色": "red",
    "红": "red",
    "橙色": "orange",
    "橘色": "orange",
    "橙": "orange",
    "金黄": "golden yellow",
    "淡黄": "pale yellow",
    "浅黄": "light yellow",
    "深黄": "deep yellow",
    "黄色": "yellow",
    "黄": "yellow",
    "蓝色": "blue",
    "蓝": "blue",
    "深紫": "dark purple",
    "浅紫": "light purple",
    "紫色": "purple",
    "紫": "purple",
    "白色": "white",
    "乳白": "creamy white",
    "米白": "off white",
    "白": "white",
    "黑色": "black",
    "黑": "black",
    "灰色": "gray",
    "灰": "gray",
    "褐色": "brown",
    "棕色": "brown",
    "咖啡色": "brown",
    "褐": "brown",
    "棕": "brown",
    "米色": "beige",
    "花斑": "speckled",
    "斑点": "spotted",
    "条纹": "striped",
    "双色": "bicolor",
    "彩色": "colorful",
}

# Size, shape and condition modifiers.
MODIFIERS = {
    "大": "large",
    "小": "small",
    "长": "long",
    "短": "short",
    "宽": "wide",
    "窄": "narrow",
    "厚": "thick",
    "薄": "thin",
    "圆形": "round",
    "圆": "round",
    "椭圆": "oval",
    "方形": "square",
    "弯曲": "curved",
    "弯": "curved",
    "笔直": "straight",
    "直": "straight",
    "扁平": "flat",
    "扁": "flat",
    "饱满": "plump",
    "干瘪": "shrivelled",
    "瘪": "shrivelled",
    "成熟": "ripe",
    "未成熟": "unripe",
    "嫩": "tender",
    "老": "mature",
    "新鲜": "fresh",
    "鲜": "fresh",
    "干燥": "dry",
    "干": "dry",
    "破损": "damaged",
    "损伤": "damaged",
    "发霉": "mouldy",
    "完整": "intact",
    "单个": "single",
    "带壳": "in shell",
    "去壳": "shelled",
}

# Longest match wins, so the order of these tables does not matter -- but a longer
# key must never be shadowed by a shorter one, which is why the scan sorts by
# length rather than iterating the dicts.
_VOCABULARY: dict[str, str] = {}
for _table in (CROPS, ORGANS, TOOLS, COLORS, MODIFIERS):
    _VOCABULARY.update(_table)

_TERMS = sorted(_VOCABULARY, key=len, reverse=True)
_NOISE_TERMS = sorted(NOISE, key=len, reverse=True)
_HAN = re.compile(r"[\u4e00-\u9fff]")

_CROP_WORDS = {value.lower() for value in CROPS.values()}
# Organs that belong to a bean when no crop was named. Measured in TextDetector's
# own notes: a bare "pod" grounds far worse than "bean pod". Cereal-only organs are
# left out -- "bean spike" would be nonsense, and PROMPT_ALIASES already turns a
# bare "spike" into "rice spike".
_BEAN_DEFAULT_ORGANS = (
    "pod", "pod shell", "seed", "seed coat", "leaf", "leaflet", "compound leaf",
    "flower", "inflorescence", "petal", "calyx", "stem", "main stem", "branch",
    "root", "root system", "seedling", "tendril", "cotyledon", "hilum", "embryo",
)


def has_chinese(text) -> bool:
    return bool(_HAN.search(str(text or "")))


# A model-backed translator can be plugged in for vocabulary this file does not
# know. It is optional and unset by default: the dictionaries above are instant and
# deterministic, and a measurement that changes between runs is not a measurement.
# Whatever the fallback returns should be cached by the caller so that a word is
# only ever translated once.
_fallback_translator = None


def set_fallback_translator(translator) -> None:
    """Install ``callable(chinese: str) -> english | None`` for unknown vocabulary.

    Pass ``None`` to go back to the dictionaries alone.
    """
    global _fallback_translator
    _fallback_translator = translator


def fallback_translator():
    """The installed fallback, or ``None``."""
    return _fallback_translator


def split_conjunctions(text) -> str:
    """Turn "豆荚和种子" into "豆荚,种子" so the caller can split it as usual."""
    result = str(text or "")
    for word in sorted(CONJUNCTIONS, key=len, reverse=True):
        result = result.replace(word, ",")
    return result


def translate(text) -> str:
    """Translate one Chinese fragment into plain English.

    Returns the input unchanged when it has no Chinese, and returns an empty string
    when the fragment is Chinese but nothing in it is recognised -- the caller then
    knows to reject it with a helpful message rather than sending Chinese to a model
    that cannot read it.
    """
    fragment = str(text or "").strip()
    if not fragment or not has_chinese(fragment):
        return fragment

    words: list[str] = []
    # Characters no dictionary entry claimed. Their presence means the fragment is
    # only partly understood, which is the one case worth asking a model about.
    unknown: list[str] = []
    index = 0
    length = len(fragment)
    while index < length:
        char = fragment[index]

        # Latin runs (and digits) survive as they are: "green 豆荚" must work.
        if not _HAN.match(char):
            start = index
            while index < length and not _HAN.match(fragment[index]):
                index += 1
            chunk = fragment[start:index].strip()
            if chunk:
                words.extend(part for part in chunk.split() if part)
            continue

        for term in _TERMS:
            if fragment.startswith(term, index):
                words.append(_VOCABULARY[term])
                index += len(term)
                break
        else:
            for noise in _NOISE_TERMS:
                if fragment.startswith(noise, index):
                    index += len(noise)
                    break
            else:
                # Unknown character: drop it. Passing Chinese through would only
                # produce a confident-looking but meaningless detection.
                unknown.append(char)
                index += 1

    if unknown and _fallback_translator is not None:
        # Part of this fragment is outside the dictionaries. Translating the rest
        # and dropping the unknown characters would quietly change the meaning --
        # 紫藤荚 would come back as a bare "pod", which is a different plant part --
        # so the whole fragment goes to the fallback, and only if that fails do we
        # fall through to the partial result below.
        try:
            translated = _fallback_translator(fragment)
        except Exception:
            translated = None
        if translated:
            return translated

    # Keep the first occurrence of each word: 绿色的绿豆 should not say "green" twice.
    seen: set[str] = set()
    unique: list[str] = []
    for word in words:
        key = word.lower()
        if key not in seen:
            seen.add(key)
            unique.append(word)

    named_crop = any(word.lower() in _CROP_WORDS for word in unique)
    if not named_crop:
        for position, word in enumerate(unique):
            if word.lower() in _BEAN_DEFAULT_ORGANS:
                unique[position] = "bean " + word
                break
    return " ".join(unique)
