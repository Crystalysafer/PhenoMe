"""Help text shared by the desktop app and the web service.

One source for both front ends: the desktop renders it into a QTextBrowser and the
web service serves it as JSON for the in-page dialog, so the two cannot describe
the same buttons differently.

Every entry is tagged with the platforms it applies to, because the two front ends
do not have the same feature set (no brush editing or batch mode on the web, no
sample images or shared link on the desktop).
"""

from __future__ import annotations

import re

WEB = "web"
DESKTOP = "desktop"
BOTH = (WEB, DESKTOP)

# {{icon:key}} markers in the guide text. The desktop dialog substitutes the
# real toolbar glyph (theme-tinted via QTextDocument resources); the web API
# substitutes the emoji so the JSON stays plain text. ``(png_name | None,
# emoji)`` -- None means the emoji is used on both front ends.
_ICONS = {
    "open": ("open", "📂"),
    "save": ("save", "💾"),
    "export": ("export", "📤"),
    "text_detect": ("text_detect", "🔤"),
    "click": ("click", "👆"),
    "edit_mask": ("edit_mask", "🖌️"),
    "ruler_detection": ("ruler_detection", "📏"),
    "color_checker": ("color_checker", "🎨"),
    "set_scale": ("set_scale", "📐"),
    "batch": ("batch", "📦"),
    "objects": ("objects", "📋"),
    "calc": ("calc", "🧮"),
    "help": ("help", "❓"),
    "camera": (None, "📷"),
    "demo": ("demo", "🖼️"),
    "cut_out": ("cut_out", "✂️"),
    "manageAlgorithm": ("manageAlgorithm", "🧩"),
    "labels": ("labels", "🏷️"),
    "file": ("file", "📄"),
}

_ICON_RE = re.compile(r"\{\{icon:([A-Za-z_]+)\}\}")

# (heading, [(text, platforms), ...]) per language.
_DOCUMENT = {
    "zh": {
        "title": "PhenoMe 使用说明",
        "intro": (
            "PhenoMe 用一张照片测量植物器官的形状与颜色：先把目标从背景里分割出来，"
            "再用照片里的标尺或色卡标定，最后导出每个目标的形状表。"
        ),
        "sections": [
            (
                "{{icon:open}} 基本流程",
                [
                    ("{{icon:open}} 打开图片：点工具栏「打开图片」，或直接点画布中间。手机上会弹出系统选择器，可以拍照、从相册选、或选文件。", (WEB,)),
                    ("{{icon:demo}} 想先试试效果，可以点「示例图片」里的蚕豆荚或扁豆种子。", (WEB,)),
                    ("{{icon:open}} 打开图片：File → Open 打开单张，或 Open Dir 打开整个文件夹。", (DESKTOP,)),
                    ("图片打开后会自动准备分割（计算图像嵌入），进度条走完就可以点选目标了。", (WEB,)),
                    ("分割目标：见下一节。", BOTH),
                    ("{{icon:ruler_detection}} 标定比例：用「尺度检测」自动识别标尺，或「设置比例」手动输入 1 像素等于多少毫米。不标定时结果以像素为单位。", BOTH),
                    ("{{icon:calc}} 计算：点「计算」得到表型数值，切到「表型」标签查看。", (WEB,)),
                    ("{{icon:export}} 导出：「下载形状表」得到 CSV，「导出标注」得到可再次编辑的 .pheno.json。", (WEB,)),
                    ("{{icon:save}} 保存：File → 保存标注（Ctrl+S）把对象和掩膜写进图像旁边的 .pheno.json，下次打开可继续编辑。", (DESKTOP,)),
                    ("{{icon:export}} 导出：工具栏「导出」出性状表 CSV；「导出掩膜」把勾选对象存成 PNG，给不读 .pheno.json 的工具用。整批结果用批处理。", (DESKTOP,)),
                ],
            ),
            (
                "{{icon:text_detect}} 分割目标",
                [
                    ("{{icon:text_detect}} 语义检测：输入对目标的直接描述（豆荚、绿色豆荚、种子、叶片、标签、尺子、色卡…），中文英文都可以。最省事，建议先试这个。", BOTH),
                    ("如果提示无法识别，换个更常见的说法（豆荚、种子、叶片、花）。", BOTH),
                    ("语义检测需要等待一会儿：文本检测在 CPU 上运行，把显卡留给分割。", (WEB,)),
                    ("如果描述和照片内容明显不符，会先弹出提示，确认后才继续。", BOTH),
                    ("{{icon:click}} 点选目标：按下「点选目标」后在图上点一下目标，会出现黄色预览；不准就再点几下细化，满意后按「保存目标」。保存后仍在点选模式，可以接着点下一个。", (WEB,)),
                    ("{{icon:click}} 点击分割：按下按钮后左键点目标内部，右键点背景来排除区域，确认后成为一个对象。", (DESKTOP,)),
                    ("{{icon:edit_mask}} 编辑掩膜：用笔刷修补分割结果，左键加、右键减；滚轮调笔刷大小，Ctrl+滚轮缩放画布。", (DESKTOP,)),
                    ("{{icon:objects}} 目标列表里可以改名和删除。改名会影响形状表里的 Name 列。", BOTH),
                ],
            ),
            (
                "{{icon:color_checker}} 标定与颜色",
                [
                    ("{{icon:ruler_detection}} 尺度检测：自动找照片里的直尺刻度并换算比例。需要刻度清晰可辨——1 厘米在照片上至少占十几个像素。", BOTH),
                    ("{{icon:color_checker}} 色卡矫正：识别 24 色标准色卡（MCC24）并校正整张图的颜色，之后的颜色类性状用校正后的像素计算。", BOTH),
                    ("色卡矫正会报告校正前后的色差（ΔE）。数字越小越好；拟合太差时会拒绝校正，而不是悄悄改坏颜色。", BOTH),
                    ("同一张图重复点色卡矫正不会叠加，始终以原图为输入。", (WEB,)),
                ],
            ),
            (
                "{{icon:camera}} 拍摄建议",
                [
                    ("相机尽量正对桌面，避免倾斜——倾斜会让照片不同位置的比例不一致。", BOTH),
                    ("把直尺放在样品旁边、和样品同一横向区域，不要贴在画幅最边缘——标定的比例就能代表样品所在位置，减少透视误差。", BOTH),
                    ("样品之间留出间隙，避免相互接触，分割会更干净。", BOTH),
                    ("背景与样品的明度差别越大越好；避免强反光和阴影。", BOTH),
                    ("色卡和直尺尽量与样品放在同一平面（同一高度）。", BOTH),
                ],
            ),
            (
                "{{icon:objects}} 形状表的列",
                [
                    ("Name：目标名称。Area / Perimeter：面积与周长。", BOTH),
                    ("Major Axis / Minor Axis：拟合椭圆的长轴与短轴；Aspect Ratio 为二者之比。", BOTH),
                    ("Solidity：面积与凸包面积之比，越小说明轮廓越凹。Convexity：凸包周长与轮廓周长之比。", BOTH),
                    ("Circularity：圆度，正圆为 1。Convex Area / Convex Perimeter：凸包的面积与周长。", BOTH),
                    ("MeanB / MeanG / MeanR：目标内部的平均蓝、绿、红通道值（0–255）。", BOTH),
                    ("单位直接写在表头里：未标定时是 (px) / (px²)，标定后变成 (mm) / (mm²)，跟着比例自动变；比值类列没有单位。", BOTH),
                    ("改变比例会清空已算出的数值：旧数字是按旧比例算的，配上新单位会得到一张看起来合理却错误的表。重新点一次「计算」即可。", BOTH),
                ],
            ),
            (
                "{{icon:help}} 常见问题",
                [
                    ("尺度检测失败：说明照片里没有可辨的刻度，或刻度太小。此时宁可返回失败也不给错误比例——请用「设置比例」手动输入。", BOTH),
                    ("色卡矫正失败：照片里没有标准 24 色卡，或色卡太小、反光。纸质自制色板通常识别不了。", BOTH),
                    ("点选后提示没找到目标：换一个更靠目标中心的位置再点。", BOTH),
                    ("上传较慢：移动网络下大照片上传需要一些时间，进度条显示的是真实上传进度。", (WEB,)),
                    ("切换语言后所有界面文字和本说明都会跟着变，选择会被记住。", (WEB,)),
                ],
            ),
            (
                "{{icon:save}} 关于数据",
                [
                    ("网页端不存盘：图片和分析结果只在服务器内存里，闲置 30 分钟或服务重启后清除，没有账号也没有历史记录。", (WEB,)),
                    ("这是一个公开服务，任何人都可以直接使用，不需要注册或口令。", (WEB,)),
                    ("背后只有一块显卡在排队处理，所以高峰时可能提示「服务器正忙」——等几秒重试即可，不是出错。", (WEB,)),
                    ("请不要用脚本批量刷接口：入口有限流，超过会返回 429/503。需要批量处理请用桌面版。", (WEB,)),
                    ("需要保留结果请及时下载形状表或导出标注。", (WEB,)),
                    ("{{icon:save}} 桌面端的标注保存为 .pheno.json，和图片放在一起即可再次打开继续编辑。", (DESKTOP,)),
                ],
            ),
        ],
    },
    "en": {
        "title": "PhenoMe user guide",
        "intro": (
            "PhenoMe measures the shape and colour of plant organs from a single photograph: "
            "segment the objects from the background, calibrate with the ruler or colour "
            "chart in the frame, then export one row of traits per object."
        ),
        "sections": [
            (
                "{{icon:open}} Basic workflow",
                [
                    ("{{icon:open}} Open an image with the toolbar button or by tapping the middle of the canvas. On a phone the system picker offers camera, photo library and files.", (WEB,)),
                    ("{{icon:demo}} To try the tool straight away, pick one of the bundled samples (bean pods or bean seeds).", (WEB,)),
                    ("{{icon:open}} Open an image with File -> Open, or a whole folder with Open Dir.", (DESKTOP,)),
                    ("Segmentation is prepared automatically once the image is open; when the progress bar finishes you can start picking objects.", (WEB,)),
                    ("Segment the objects -- see the next section.", BOTH),
                    ("{{icon:ruler_detection}} Calibrate: use ruler detection, or Set scale to type how many millimetres one pixel is. Without calibration the results are in pixels.", BOTH),
                    ("{{icon:calc}} Press Calculate for the trait values and switch to the Traits tab to read them.", (WEB,)),
                    ("{{icon:export}} Export: Download table gives a CSV, Export annotation gives a re-editable .pheno.json.", (WEB,)),
                    ("{{icon:save}} Save: File -> Save Annotation (Ctrl+S) writes the objects and their masks to a .pheno.json beside the image, ready to reopen later.", (DESKTOP,)),
                    ("{{icon:export}} Export: the toolbar's Export writes the trait table as CSV; Export Masks writes the ticked objects as PNG files, for tools that do not read .pheno.json. For a whole folder, use batch mode.", (DESKTOP,)),
                ],
            ),
            (
                "{{icon:text_detect}} Segmenting objects",
                [
                    ("{{icon:text_detect}} Semantic detection: type a direct description of the object (pod, green pod, seed, leaf, label, ruler, colour chart) -- English and Chinese both work. It is the least work, so try it first.", BOTH),
                    ("If the wording is not recognised, try a more common one (pod, seed, leaf, flower).", BOTH),
                    ("Semantic detection takes a while: the text detector runs on the CPU so the GPU stays free for segmentation.", (WEB,)),
                    ("If the description clearly disagrees with the picture you are warned first and nothing is created until you confirm.", BOTH),
                    ("{{icon:click}} Pick object: press the button, tap the object, and a yellow preview appears. Tap again to refine, then press Save object. Picking stays on, so the next object is one tap away.", (WEB,)),
                    ("{{icon:click}} Click segment: left-click inside the object, right-click on background to exclude it, then confirm to create the object.", (DESKTOP,)),
                    ("{{icon:edit_mask}} Edit mask: brush the result -- left button adds, right button removes; the wheel changes the brush size and Ctrl+wheel zooms the canvas.", (DESKTOP,)),
                    ("{{icon:objects}} Objects can be renamed and deleted in the list. The name becomes the Name column of the trait table.", BOTH),
                ],
            ),
            (
                "{{icon:color_checker}} Calibration and colour",
                [
                    ("{{icon:ruler_detection}} Ruler detection finds the tick marks and converts them to a scale. The ticks must be resolvable -- one centimetre needs at least a dozen pixels.", BOTH),
                    ("{{icon:color_checker}} Colour calibration detects a 24-patch MCC24 chart and corrects the whole image; colour traits are then computed on the corrected pixels.", BOTH),
                    ("Calibration reports the colour error (ΔE) before and after. Lower is better, and a fit that is too poor is refused rather than quietly wrecking the picture.", BOTH),
                    ("Pressing colour calibration twice does not stack: the fit always starts from the original pixels.", (WEB,)),
                ],
            ),
            (
                "{{icon:camera}} Photography tips",
                [
                    ("Keep the camera square to the table -- tilt makes the scale inconsistent across the frame.", BOTH),
                    ("Put the ruler next to the samples, in the same part of the frame -- not along the very edge. The calibration then represents the samples' position and reduces perspective error.", BOTH),
                    ("Leave gaps between samples so they do not touch; segmentation is much cleaner.", BOTH),
                    ("The bigger the brightness difference between background and sample the better; avoid strong reflections and shadows.", BOTH),
                    ("Keep the chart and the ruler in the same plane as the samples.", BOTH),
                ],
            ),
            (
                "{{icon:objects}} Trait table columns",
                [
                    ("Name: object label. Area / Perimeter: area and outline length.", BOTH),
                    ("Major Axis / Minor Axis: axes of the fitted ellipse; Aspect Ratio is their quotient.", BOTH),
                    ("Solidity: area over convex-hull area, lower means a more concave outline. Convexity: hull perimeter over outline perimeter.", BOTH),
                    ("Circularity: 1 for a perfect circle. Convex Area / Convex Perimeter: the hull's area and perimeter.", BOTH),
                    ("MeanB / MeanG / MeanR: mean blue, green and red value inside the object (0-255).", BOTH),
                    ("The unit is printed in the header: (px)/(px²) before calibration, (mm)/(mm²) after, following the scale automatically. Ratio columns carry no unit.", BOTH),
                    ("Changing the scale clears the computed values: the old numbers were produced under the old scale, and pairing them with a new unit would look right and be wrong. Press Calculate again.", BOTH),
                ],
            ),
            (
                "{{icon:help}} Troubleshooting",
                [
                    ("Ruler detection failed: there is no resolvable tick pattern in the frame. A failure is deliberate -- a wrong scale would corrupt every measurement -- so type the scale by hand instead.", BOTH),
                    ("Colour calibration failed: no standard 24-patch chart, or it is too small or too reflective. Home-made paper charts are usually not recognised.", BOTH),
                    ("\u201cNo object found there\u201d: tap closer to the middle of the object.", BOTH),
                    ("Slow upload: a large photo on mobile data takes a while; the progress bar shows real bytes.", (WEB,)),
                    ("Switching the language updates the whole interface and this guide, and the choice is remembered.", (WEB,)),
                ],
            ),
            (
                "{{icon:save}} About your data",
                [
                    ("Nothing is stored on the web service: images and results live in memory only and are dropped after 30 minutes idle or on restart. There are no accounts and no history.", (WEB,)),
                    ("This is an open service -- anyone can use it, with no sign-up and no access code.", (WEB,)),
                    ("A single GPU serves everyone in turn, so at busy moments you may be told the server is busy. Wait a few seconds and retry; nothing went wrong.", (WEB,)),
                    ("Please do not script bulk requests against it: the entry is rate limited and will answer 429/503. Use the desktop application for batch work.", (WEB,)),
                    ("Download the trait table or export the annotation if you need to keep the results.", (WEB,)),
                    ("{{icon:save}} On the desktop the annotation is saved as .pheno.json next to the image and can be reopened to continue editing.", (DESKTOP,)),
                ],
            ),
        ],
    },
}


def normalise_language(language: str) -> str:
    """Map anything the callers use ('zh', 'zh_CN', 'en_GB') onto our two keys."""
    return "zh" if str(language or "").lower().startswith("zh") else "en"


def _icon_text(text: str) -> str:
    """Markers -> emoji, for the plain-text JSON the web dialog renders."""
    def repl(match):
        entry = _ICONS.get(match.group(1))
        return entry[1] if entry else ""
    return _ICON_RE.sub(repl, text)


def _icon_html(text: str) -> str:
    """Markers -> <img> tags (desktop) or emoji when the key has no PNG."""
    def repl(match):
        entry = _ICONS.get(match.group(1))
        if entry is None:
            return ""
        png, emoji = entry
        if png is None:
            return emoji
        return f"<img src='icon:{png}' width='16' height='16'>"
    return _ICON_RE.sub(repl, text)


def help_icon_names(language: str = "zh", platform: str = DESKTOP) -> list:
    """PNG names the rendered guide needs, in document order.

    The desktop dialog registers one QTextDocument image resource per name
    (``icon:<png>``), tinted like the toolbar glyphs.
    """
    document = _DOCUMENT[normalise_language(language)]
    names = []
    for heading, entries in document["sections"]:
        texts = [heading] + [text for text, platforms in entries
                             if platform in platforms]
        for text in texts:
            for key in _ICON_RE.findall(text):
                png = _ICONS[key][0]
                if png and png not in names:
                    names.append(png)
    return names


def _raw_sections(language: str, platform: str) -> list:
    """(heading, [items]) with the icon markers still in place."""
    document = _DOCUMENT[normalise_language(language)]
    sections = []
    for heading, entries in document["sections"]:
        items = [text for text, platforms in entries if platform in platforms]
        if items:
            sections.append((heading, items))
    return document, sections


def help_document(language: str = "zh", platform: str = WEB) -> dict:
    """The guide for one language and front end, as plain data."""
    document, sections = _raw_sections(language, platform)
    return {
        "language": normalise_language(language),
        "platform": platform,
        "title": document["title"],
        "intro": document["intro"],
        "sections": [
            {"heading": _icon_text(heading),
             "items": [_icon_text(item) for item in items]}
            for heading, items in sections
        ],
    }


def help_html(language: str = "zh", platform: str = DESKTOP) -> str:
    """The same guide as a small HTML document (used by the desktop dialog).

    The colours come from the active theme: this document is rendered inside a
    QTextBrowser, which takes no notice of the application stylesheet's palette, so a
    hardcoded dark-on-light body made the whole guide unreadable in the dark theme.
    """
    from pheno import theme

    c = theme.palette()
    document, sections = _raw_sections(language, platform)
    parts = [
        "<html><head><meta charset='utf-8'><style>"
        f"body{{font-family:'Microsoft YaHei','Segoe UI',sans-serif;font-size:13px;"
        f"color:{c['text']};}}"
        "h2{font-size:16px;margin:0 0 8px;}"
        f"h3{{font-size:13px;margin:16px 0 6px;color:{c['accent']};}}"
        f"p{{margin:0 0 10px;line-height:1.6;color:{c['muted']};}}"
        "ul{margin:0 0 4px 18px;padding:0;}"
        "li{margin:0 0 6px;line-height:1.6;}"
        "img{vertical-align:-3px;}"
        "</style></head><body>",
        f"<h2>{document['title']}</h2>",
        f"<p>{document['intro']}</p>",
    ]
    for heading, items in sections:
        parts.append(f"<h3>{_icon_html(heading)}</h3><ul>")
        parts.extend(f"<li>{_icon_html(item)}</li>" for item in items)
        parts.append("</ul>")
    parts.append("</body></html>")
    return "".join(parts)
