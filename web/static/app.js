/*
 * PhenoMe Web front-end.
 *
 * The image stays at its natural resolution on the server; the browser only
 * scales it for display. Every coordinate sent to the API is therefore
 * converted back to original image pixels, while the overlay canvas is drawn in
 * display pixels so outlines and handles keep a constant on-screen size.
 *
 * All user-visible text goes through t(); static markup carries data-i18n keys.
 * Trait table headers are deliberately left in English: they are the CSV column
 * names and must match the desktop export.
 */

const STRINGS = {
  zh: {
    "help.open": "使用说明",
    "help.close": "知道了",
    "help.failed": "帮助文档加载失败",
    "language.switch": "切换到英文",
    "language.other": "EN",
    "status.connecting": "连接中",
    "status.online": "服务在线",
    "status.offline": "服务离线",
    "model.load": "加载模型",
    "model.ready": "模型就绪",
    "model.loading": "模型加载中…",
    "model.notLoaded": "模型未加载",
    "model.unknown": "模型状态未知",
    "model.error": "模型错误：{message}",
    "model.missing": "缺少权重 {checkpoint}",
    "model.loadFailed": "模型加载失败",
    "toolbar.open": "打开图片",
    "toolbar.clear": "清除",
    "toolbar.resetPoints": "清除提示点",
    "pick.start": "点选目标",
    "pick.exit": "退出点选",
    "pick.save": "保存目标",
    "group.sample": "示例图片",
    "sample.pods": "豆荚",
    "sample.seeds": "种子",
    "sample.leaves": "叶片",
    "sample.flowers": "花",
    "sample.loading": "正在载入示例图片…",
    "sample.empty": "示例图片读取失败（0 字节），请重试",
    "progress.model": "加载模型",
    "progress.upload": "上传图片",
    "progress.embedding": "准备图像嵌入",
    "progress.segment": "分割中",
    "progress.semantic": "语义检测中",
    "progress.color": "色卡矫正中",
    "progress.ruler": "标尺检测中",
    "progress.traits": "计算表型中",
    "progress.sample": "载入示例图片",
    "error.network": "网络中断，请重试",
    "toolbar.semantic": "语义检测",
    "toolbar.color": "色卡矫正",
    "toolbar.ruler": "尺度检测",
    "toolbar.scale": "设置比例",
    "group.analyze": "分析",
    "group.calibrate": "标定与比例",
    "panel.image": "图像",
    "panel.emptyImage": "尚未打开图片",
    "panel.objects": "目标",
    "panel.traits": "表型",
    "panel.noObjects": "暂无目标",
    "panel.noTraits": "暂无表型结果",
    "actions.calc": "计算",
    "actions.download": "下载形状表",
    "actions.annotation": "导出标注",
    "actions.clearAll": "删除全部",
    "object.rename": "改名",
    "object.delete": "删除",
    "canvas.empty": "点这里打开图片",
    "canvas.emptyHint": "也可以选择上方的示例图片",
    "canvas.hint.noImage": "打开图片后自动准备分割",
    "canvas.hint.idle": "尚未准备图像嵌入",
    "canvas.hint.pending": "正在计算图像嵌入…",
    "canvas.hint.ready": "已就绪，点「点选目标」后在图上点击",
    "canvas.hint.picking": "在图像上点击目标，可连点多次细化，然后点「保存目标」",
    "canvas.hint.error": "嵌入失败：{message}",
    "busy.segment": "正在分割…",
    "busy.commit": "正在创建目标…",
    "busy.semantic": "正在进行语义检测…",
    "busy.color": "正在检测色卡并矫正…",
    "busy.ruler": "正在检测标尺…",
    "busy.traits": "正在计算表型…",
    "toast.notImage": "请选择图片文件",
    "toast.uploaded": "图片已上传，正在准备嵌入",
    "toast.embeddingReady": "图像嵌入已就绪",
    "toast.embeddingNotReady": "图像嵌入尚未就绪",
    "toast.noObjectHere": "该位置没有找到目标，请换一个点",
    "toast.created": "已创建 {label}",
    "toast.commitFailed": "分割失败，请重新选点",
    "toast.modelReady": "模型已就绪",
    "toast.scaleUpdated": "比例已更新",
    "toast.positiveNumber": "请输入正数",
    "toast.semanticNew": "语义检测新增 {count} 个目标{filtered}",
    "toast.semanticNone": "未找到匹配目标{filtered}",
    "toast.translatedSuffix": "（按「{translated}」检索）",
    "toast.filteredSuffix": "，过滤 {count} 个异常",
    "toast.colorDone": "色卡矫正完成，显示 {count} 个色卡点",
    "toast.colorDoneQuality": "色卡矫正完成（色差 {before} → {after}），{count} 个色块",
    "toast.colorAlready": "该图像已完成色卡矫正",
    "toast.rulerDone": "尺度检测完成，识别 {count} 个刻度",
    "toast.rulerUneven": "尺度检测完成（{count} 个刻度）；尺子两端比例差 {drift}%，相机可能未正对",
    "toast.traitsDone": "已计算 {count} 个目标",
    "toast.downloadStarted": "形状表已开始下载",
    "toast.noSession": "无法创建会话",
    "prompt.rename": "目标名称",
    "prompt.scaleValue": "1 像素对应的真实长度（例如 0.0523）",
    "prompt.scaleUnit": "长度单位",
    "prompt.semantic": "描述要检测的目标：如 豆荚、绿色豆荚、种子、叶片、花等",
    "confirm.clearAll": "删除全部 {count} 个目标？",
    "confirm.semanticMismatch":
      "图像看起来不像“{requested}”，检测模型更倾向“{suggested}”（{count} 个区域，最高分 {score}）。\n\n仍然继续检测“{prompt}”吗？",
    "error.request": "请求失败 ({status})",
    "error.busy": "服务器正忙，稍等几秒再试（一块显卡在为所有人排队）",
    "error.tooFast": "操作太频繁，请稍等一下再继续",
  },
  en: {
    "help.open": "User guide",
    "help.close": "Got it",
    "help.failed": "The user guide could not be loaded",
    "language.switch": "Switch to Chinese",
    "language.other": "中文",
    "status.connecting": "Connecting",
    "status.online": "Service online",
    "status.offline": "Service offline",
    "model.load": "Load model",
    "model.ready": "Model ready",
    "model.loading": "Loading model…",
    "model.notLoaded": "Model not loaded",
    "model.unknown": "Model state unknown",
    "model.error": "Model error: {message}",
    "model.missing": "Missing checkpoint {checkpoint}",
    "model.loadFailed": "The model failed to load",
    "toolbar.open": "Open image",
    "toolbar.clear": "Clear",
    "toolbar.resetPoints": "Clear points",
    "pick.start": "Pick object",
    "pick.exit": "Stop picking",
    "pick.save": "Save object",
    "group.sample": "Samples",
    "sample.pods": "Pods",
    "sample.seeds": "Seeds",
    "sample.leaves": "Leaves",
    "sample.flowers": "Flowers",
    "sample.loading": "Loading the sample image…",
    "sample.empty": "The sample image came back empty (0 bytes) -- try again",
    "progress.model": "Loading the model",
    "progress.upload": "Uploading",
    "progress.embedding": "Preparing the embedding",
    "progress.segment": "Segmenting",
    "progress.semantic": "Semantic detection",
    "progress.color": "Color calibration",
    "progress.ruler": "Ruler detection",
    "progress.traits": "Calculating traits",
    "progress.sample": "Loading the sample",
    "error.network": "The connection dropped -- try again",
    "toolbar.semantic": "Semantic detect",
    "toolbar.color": "Color calibration",
    "toolbar.ruler": "Ruler detection",
    "toolbar.scale": "Set scale",
    "group.analyze": "Analyse",
    "group.calibrate": "Calibration & scale",
    "panel.image": "Image",
    "panel.emptyImage": "No image yet",
    "panel.objects": "Objects",
    "panel.traits": "Traits",
    "panel.noObjects": "No objects yet",
    "panel.noTraits": "No trait results yet",
    "actions.calc": "Calculate",
    "actions.download": "Download table",
    "actions.annotation": "Export annotation",
    "actions.clearAll": "Delete all",
    "object.rename": "Rename",
    "object.delete": "Delete",
    "canvas.empty": "Tap here to open an image",
    "canvas.emptyHint": "or pick one of the samples above",
    "canvas.hint.noImage": "Segmentation is prepared automatically after opening an image",
    "canvas.hint.idle": "The image embedding is not prepared yet",
    "canvas.hint.pending": "Preparing the image embedding…",
    "canvas.hint.ready": "Ready -- press \u201cPick object\u201d, then tap the image",
    "canvas.hint.picking": "Tap the object, tap again to refine, then press \u201cSave object\u201d",
    "canvas.hint.error": "Embedding failed: {message}",
    "busy.segment": "Segmenting…",
    "busy.commit": "Creating the object…",
    "busy.semantic": "Running semantic detection…",
    "busy.color": "Detecting the color chart…",
    "busy.ruler": "Detecting the ruler…",
    "busy.traits": "Calculating traits…",
    "toast.notImage": "Please choose an image file",
    "toast.uploaded": "Image uploaded, preparing the embedding",
    "toast.embeddingReady": "The image embedding is ready",
    "toast.embeddingNotReady": "The image embedding is not ready yet",
    "toast.noObjectHere": "No object was found there -- try another point",
    "toast.created": "Created {label}",
    "toast.commitFailed": "Segmentation failed -- pick the points again",
    "toast.modelReady": "The model is ready",
    "toast.scaleUpdated": "Scale updated",
    "toast.positiveNumber": "Please enter a positive number",
    "toast.semanticNew": "Semantic detection added {count} object(s){filtered}",
    "toast.semanticNone": "No matching object found{filtered}",
    "toast.translatedSuffix": " (searched for \"{translated}\")",
    "toast.filteredSuffix": ", {count} outlier(s) filtered",
    "toast.colorDone": "Color calibration done, {count} patch centers shown",
    "toast.colorDoneQuality": "Calibrated: colour error {before} -> {after} dE, {count} patches",
    "toast.colorAlready": "This image is already calibrated",
    "toast.rulerDone": "Ruler detected, {count} ticks",
    "toast.rulerUneven": "Ruler detected ({count} ticks); scale differs by {drift}% end to end -- check the camera angle",
    "toast.traitsDone": "Calculated {count} object(s)",
    "toast.downloadStarted": "The trait table is downloading",
    "toast.noSession": "Could not create a session",
    "prompt.rename": "Object name",
    "prompt.scaleValue": "Real length of one pixel (for example 0.0523)",
    "prompt.scaleUnit": "Unit",
    "prompt.semantic": "Describe the target to detect: e.g. pod, green pod, seed, leaf, flower, etc.",
    "confirm.clearAll": "Delete all {count} objects?",
    "confirm.semanticMismatch":
      "This image does not look like \u201c{requested}\u201d; the detector prefers \u201c{suggested}\u201d ({count} regions, best score {score}).\n\nRun \u201c{prompt}\u201d anyway?",
    "error.request": "Request failed ({status})",
    "error.busy": "The server is busy -- try again in a few seconds (one GPU serves everyone)",
    "error.tooFast": "Too many requests in a row; please slow down a little",
  },
};

// Bump on every static-asset release: appended to sample URLs so browsers (and
// the edge cache in front of the service) cannot serve an older sample image
// alongside a newer page. index.html carries the same stamp for app.js/css.
const STATIC_VERSION = "20261106a";

const state = {
  lang: "zh",
  sessionId: null,
  localUrl: null,
  selectedFile: null,
  selectedName: null,
  selectedType: null,
  image: null,
  model: null,
  embedding: { state: "idle" },
  clickMode: false,
  points: [],
  preview: null,
  objects: [],
  selectedUid: null,
  traits: null,
  scale: { per_pixel: 1, unit: "pixel" },
  colorCenters: [],
  colorCorrection: false,
  rulerTicks: [],
  correctedImageUrl: null,
  busy: false,
};

function t(key, vars) {
  const table = STRINGS[state.lang] || STRINGS.zh;
  let text = table[key];
  if (text === undefined) text = STRINGS.zh[key];
  if (text === undefined) return key;
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (match, name) => (name in vars ? String(vars[name]) : match));
}

const el = (id) => document.querySelector(id);
const elements = {
  detailTabs: [...document.querySelectorAll("[data-detail-target]")],
  detailPanes: [...document.querySelectorAll("[data-detail]")],
  languageToggle: el("#language-toggle"),
  statusDot: el("#status-dot"),
  serviceStatus: el("#service-status"),
  workspaceFile: el("#workspace-file"),
  workspaceStage: el("#workspace-stage"),
  workspacePreview: el("#workspace-preview"),
  canvasFrame: el("#canvas-frame"),
  overlay: el("#canvas-overlay"),
  workspaceFileList: el("#workspace-file-list"),
  workspaceDimensions: el("#workspace-dimensions"),
  canvasHint: el("#canvas-hint"),
  scaleReadout: el("#scale-readout"),
  fileCount: el("#file-count"),
  modelNote: el("#model-note"),
  loadModel: el("#load-model"),
  pickObject: el("#pick-object"),
  // Sample buttons are wired in one pass further down, by their data-sample
  // name, so adding a sample image needs no entry here.
  resetPoints: el("#reset-points"),
  setScale: el("#set-scale"),
  semanticDetect: el("#semantic-detect"),
  colorChecker: el("#color-checker"),
  rulerDetection: el("#ruler-detection"),
  calibrateGroup: el("#calibrate-group"),
  objectList: el("#object-list"),
  traitWrap: el("#trait-wrap"),
  calcTraits: el("#calc-traits"),
  exportAnnotation: el("#export-annotation"),
  exportResults: el("#export-results"),
  clearObjects: el("#clear-objects"),
  clearImage: el("#clear-image"),
  helpOverlay: el("#help-overlay"),
  helpPanelTitle: el("#help-title"),
  helpBody: el("#help-body"),
  helpOpen: el("#help-open"),
  helpClose: el("#help-close"),
  progress: el("#progress"),
  progressFill: el("#progress-fill"),
  progressLabel: el("#progress-label"),
  progressPercent: el("#progress-percent"),
  toast: el("#toast"),
};

// ---------------------------------------------------------------------------
// Language
// ---------------------------------------------------------------------------

function applyLanguage() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-title]").forEach((node) => {
    node.title = t(node.dataset.i18nTitle);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
  elements.languageToggle.textContent = t("language.other");
  elements.languageToggle.title = t("language.switch");

  // Everything below is rendered from state, so it has to be redrawn by hand.
  renderModel();
  renderScale();
  renderObjects();
  renderTraits();
  renderEmbedding();
  renderImageMeta();
  if (!elements.helpOverlay.hidden) {
    loadHelp(state.lang).then(renderHelp).catch(() => {});
  }
}

function setLanguage(lang) {
  state.lang = STRINGS[lang] ? lang : "zh";
  try {
    window.localStorage.setItem("phenome.lang", state.lang);
  } catch (_) {
    // Private browsing: the choice simply does not persist.
  }
  applyLanguage();
}

function initialLanguage() {
  try {
    const saved = window.localStorage.getItem("phenome.lang");
    if (saved && STRINGS[saved]) return saved;
  } catch (_) {
    // ignore
  }
  const preferred = (navigator.language || "zh").toLowerCase();
  return preferred.startsWith("zh") ? "zh" : "en";
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

// Fetched rather than bundled: /api/help renders the same source the desktop
// dialog uses, so the two guides cannot drift apart. Cached per language because
// switching back and forth should not re-request.
const helpCache = {};

async function loadHelp(lang) {
  if (helpCache[lang]) return helpCache[lang];
  const document_ = await request(`/api/help?lang=${encodeURIComponent(lang)}`);
  helpCache[lang] = document_;
  return document_;
}

function renderHelp(document_) {
  elements.helpPanelTitle.textContent = document_.title;
  elements.helpBody.replaceChildren();

  const intro = document.createElement("p");
  intro.className = "help-intro";
  intro.textContent = document_.intro;
  elements.helpBody.append(intro);

  document_.sections.forEach((section) => {
    const heading = document.createElement("h3");
    heading.textContent = section.heading;
    const list = document.createElement("ul");
    section.items.forEach((item) => {
      const entry = document.createElement("li");
      entry.textContent = item;
      list.append(entry);
    });
    elements.helpBody.append(heading, list);
  });
}

async function openHelp() {
  try {
    renderHelp(await loadHelp(state.lang));
  } catch (error) {
    showToast(t("help.failed"));
    return;
  }
  elements.helpOverlay.hidden = false;
  elements.helpBody.scrollTop = 0;
  elements.helpClose.focus();
}

function closeHelp() {
  elements.helpOverlay.hidden = true;
  try {
    window.localStorage.setItem("phenome.helpSeen", "1");
  } catch (_) {
    // Private browsing: the guide opens again next time, which is harmless.
  }
}

function helpAlreadySeen() {
  try {
    return window.localStorage.getItem("phenome.helpSeen") === "1";
  } catch (_) {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Plumbing
// ---------------------------------------------------------------------------

function setDetail(name) {
  elements.detailTabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.detailTarget === name));
  elements.detailPanes.forEach((pane) => pane.classList.toggle("is-active", pane.dataset.detail === name));
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

// One strip for every slow operation. Uploads report real bytes; the GPU and CPU
// steps have no server-side progress, so they ease towards 92 % over the time
// they usually take and jump to 100 % when the answer arrives. That is honest
// enough to show something is running and never claims to be finished early.
const progress = { timer: null, key: null, value: 0 };

function paintProgress() {
  const percent = Math.max(0, Math.min(1, progress.value));
  elements.progressFill.style.width = `${(percent * 100).toFixed(1)}%`;
  elements.progressPercent.textContent = `${Math.round(percent * 100)}%`;
}

function startProgress(key, expectedSeconds = 0) {
  progress.key = key;
  progress.value = 0.02;
  elements.progressLabel.textContent = t(key);
  elements.progressLabel.dataset.i18n = key;
  elements.progress.hidden = false;
  elements.progress.classList.add("is-running");
  paintProgress();

  window.clearInterval(progress.timer);
  if (expectedSeconds > 0) {
    const step = 250;
    const started = Date.now();
    progress.timer = window.setInterval(() => {
      const elapsed = (Date.now() - started) / 1000;
      // Asymptotic: fast at first, then crawling, capped below 100 %.
      progress.value = 0.92 * (1 - Math.exp(-2.2 * (elapsed / expectedSeconds)));
      paintProgress();
    }, step);
  }
}

function setProgress(fraction) {
  window.clearInterval(progress.timer);
  progress.timer = null;
  progress.value = fraction;
  paintProgress();
}

function endProgress(ok = true) {
  window.clearInterval(progress.timer);
  progress.timer = null;
  if (ok) {
    progress.value = 1;
    paintProgress();
  }
  elements.progress.classList.remove("is-running");
  const key = progress.key;
  window.setTimeout(() => {
    // Only hide if nothing else claimed the strip in the meantime.
    if (progress.key === key) {
      elements.progress.hidden = true;
      progress.value = 0;
      paintProgress();
    }
  }, ok ? 450 : 0);
}

// XHR rather than fetch: it is the only way to see upload progress, and the
// upload is the slowest step on a phone (a 5 MB photo takes ~20 s on the tunnel).
function uploadWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
    });
    xhr.addEventListener("load", () => {
      let payload = null;
      if (xhr.responseText) {
        try {
          payload = JSON.parse(xhr.responseText);
        } catch (_) {
          payload = { detail: xhr.responseText };
        }
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
      else {
        if (xhr.status === 503) return reject(new Error(t("error.busy")));
        if (xhr.status === 429) return reject(new Error(t("error.tooFast")));
        const detail = payload && payload.detail ? payload.detail : t("error.request", { status: xhr.status });
        reject(new Error(typeof detail === "string" ? detail : JSON.stringify(detail)));
      }
    });
    xhr.addEventListener("error", () => reject(new Error(t("error.network"))));
    xhr.addEventListener("abort", () => reject(new Error(t("error.network"))));
    xhr.send(formData);
  });
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("is-visible"), 2600);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (_) {
      payload = { detail: text };
    }
  }
  if (!response.ok) {
    // The service is public and has one GPU, so "everyone is using it right now"
    // is a normal outcome rather than a fault. 429 comes from nginx's rate limit,
    // 503 from the analysis queue being full; both mean "wait a moment", and the
    // raw English detail from the server would only confuse the visitor.
    if (response.status === 503) throw new Error(t("error.busy"));
    if (response.status === 429) throw new Error(t("error.tooFast"));
    const detail = payload && payload.detail ? payload.detail : t("error.request", { status: response.status });
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

const postJson = (url, body) =>
  request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

function imageRef() {
  return { session_id: state.sessionId, image_id: state.image.id };
}

// ---------------------------------------------------------------------------
// Model and embedding
// ---------------------------------------------------------------------------

function describeModel(model) {
  if (!model) return t("model.unknown");
  const device = model.device ? ` · ${model.device}` : "";
  if (model.state === "ready") return t("model.ready") + device;
  if (model.state === "loading") return t("model.loading");
  if (model.state === "error") return t("model.error", { message: model.error || "?" });
  if (!model.checkpoint_available) return t("model.missing", { checkpoint: model.checkpoint });
  return t("model.notLoaded");
}

function renderModel() {
  elements.modelNote.textContent = describeModel(state.model);
  const ready = state.model && state.model.state === "ready";
  elements.loadModel.disabled = ready || (state.model && state.model.state === "loading");
  elements.loadModel.textContent = ready ? t("model.ready") : t("model.load");
  if (state.model) {
    elements.serviceStatus.textContent = `${t("status.online")} · ${describeModel(state.model)}`;
  }
}

async function refreshModel() {
  try {
    state.model = await request("/api/model");
  } catch (_) {
    state.model = null;
  }
  renderModel();
  return state.model;
}

async function loadModel() {
  elements.loadModel.disabled = true;
  elements.modelNote.textContent = t("model.loading");
  // ~12 s on the deployment GPU, so the bar is worth more here than anywhere else.
  startProgress("progress.model", 13);
  try {
    state.model = await postJson("/api/model/load", {});
  } catch (error) {
    endProgress(false);
    showToast(error.message);
    await refreshModel();
    return;
  }
  renderModel();
  await waitFor(async () => {
    const model = await refreshModel();
    return model && (model.state === "ready" || model.state === "error");
  }, 240);
  endProgress(state.model ? state.model.state === "ready" : false);
  if (state.model && state.model.state === "error") showToast(state.model.error || t("model.loadFailed"));
  else showToast(t("toast.modelReady"));
}

async function waitFor(check, attempts = 60, delay = 1000) {
  for (let i = 0; i < attempts; i += 1) {
    if (await check()) return true;
    await new Promise((resolve) => window.setTimeout(resolve, delay));
  }
  return false;
}

function renderEmbedding() {
  let hint;
  if (!state.image) {
    hint = t("canvas.hint.noImage");
  } else {
    const stateName = state.embedding ? state.embedding.state : "idle";
    if (stateName === "ready") hint = state.clickMode ? t("canvas.hint.picking") : t("canvas.hint.ready");
    else if (stateName === "pending") hint = t("canvas.hint.pending");
    else if (stateName === "error") {
      hint = t("canvas.hint.error", { message: (state.embedding && state.embedding.error) || "?" });
    } else hint = t("canvas.hint.idle");
  }
  elements.canvasHint.textContent = hint;
  updateControls();
}

async function prepareEmbedding() {
  if (!state.image) return;
  // Model load (first image only) plus the encode itself.
  startProgress("progress.embedding", state.model && state.model.state === "ready" ? 5 : 16);
  try {
    const payload = await postJson("/api/analysis/embedding", imageRef());
    state.embedding = payload.embedding;
    state.model = payload.model;
  } catch (error) {
    endProgress(false);
    showToast(error.message);
    return;
  }
  renderModel();
  renderEmbedding();
  await waitFor(async () => {
    const current = await refreshAnalysis();
    return current && current.embedding.state !== "pending";
  }, 240);
  endProgress(state.embedding.state === "ready");
  if (state.embedding.state === "ready") showToast(t("toast.embeddingReady"));
  else if (state.embedding.state === "error") showToast(elements.canvasHint.textContent);
}

async function refreshAnalysis() {
  if (!state.image) return null;
  try {
    const payload = await request(
      `/api/analysis?session_id=${encodeURIComponent(state.sessionId)}&image_id=${encodeURIComponent(state.image.id)}`
    );
    state.embedding = payload.embedding;
    state.model = payload.model;
    state.objects = payload.objects;
    state.scale = payload.scale;
    state.colorCenters = payload.color_centers || [];
    state.colorCorrection = Boolean(payload.color_correction);
    state.rulerTicks = payload.ruler_ticks || [];
    renderModel();
    renderObjects();
    renderScale();
    renderEmbedding();
    return payload;
  } catch (_) {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Canvas overlay
// ---------------------------------------------------------------------------

// The overlay is a 2D canvas stacked on the image rather than an SVG. On the
// browsers this service is expected to meet — domestic Linux builds running
// software rasterisation without GPU compositing — a few hundred SVG nodes
// carrying translucent fills cost far more to redraw than one bitmap, and the
// SVG scaling is also what previously collapsed outline widths to sub-pixel so
// that borders vanished on small screens. A canvas states widths directly in
// screen pixels, so 3.5 px stays 3.5 px at any zoom or window size.
const OUTLINE_PX = 3.5; // outline width on screen, in CSS pixels
const OUTLINE_SELECTED_PX = 4.5;
const PROMPT_RADIUS_PX = 6;
const RULER_TICK_HALF_PX = 4; // half-length of a ruler tick, in screen pixels
const MAX_POLYGON_POINTS = 320; // thin longer contours out before drawing
const MAX_DPR = 2; // cap the backing store; a 4K panel buys nothing here

// Colours live here rather than in the stylesheet: a canvas cannot be styled
// from CSS, and one place is better than two.
const COLOURS = {
  shapeFill: "rgba(255, 0, 0, 0.26)",
  shapeStroke: "rgba(34, 197, 94, 0.95)",
  selectedFill: "rgba(0, 128, 255, 0.42)",
  selectedStroke: "#ffffff",
  previewFill: "rgba(255, 196, 0, 0.30)",
  previewStroke: "#ffd23f",
  promptFill: "#22c55e",
  promptStroke: "#ffffff",
  colorCenter: "#f59e0b",
  ruler: "#38bdf8",
};

function outlinePoints(polygon) {
  // Long contours cost the most to draw and add nothing visible at display
  // size, so thin them out before they reach the canvas.
  if (!polygon || polygon.length <= MAX_POLYGON_POINTS) return polygon;
  const step = polygon.length / MAX_POLYGON_POINTS;
  const thinned = [];
  for (let index = 0; index < polygon.length; index += step) {
    thinned.push(polygon[Math.floor(index)]);
  }
  return thinned;
}

function fitCanvasFrame() {
  // The picture is drawn with object-fit: contain, so whenever the frame's
  // aspect ratio differs from the image's the browser letterboxes it -- and the
  // overlay, which spans the whole frame, would then be stretched and shifted
  // against the pixels below it. Sizing the frame to the fitted picture keeps
  // image, overlay and click coordinates on one rectangle.
  const frame = elements.canvasFrame;
  const stage = elements.workspaceStage;
  if (!frame || !stage || !state.image || !state.image.width || !state.image.height) return;
  const rect = stage.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const scale = Math.min(rect.width / state.image.width, rect.height / state.image.height);
  const width = Math.max(1, Math.round(state.image.width * scale));
  const height = Math.max(1, Math.round(state.image.height * scale));
  if (frame.style.width !== `${width}px`) frame.style.width = `${width}px`;
  if (frame.style.height !== `${height}px`) frame.style.height = `${height}px`;
}

function overlaySurface() {
  const canvas = elements.overlay;
  if (!canvas || typeof canvas.getContext !== "function") return null;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const ratio = Math.min(MAX_DPR, Math.max(1, window.devicePixelRatio || 1));
  const backingWidth = Math.round(width * ratio);
  const backingHeight = Math.round(height * ratio);
  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    // Assigning either dimension clears the canvas, so only do it on a change.
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.lineJoin = "round";
  context.lineCap = "round";
  return { context, width, height };
}

function polygonPath(context, polygon, scaleX, scaleY) {
  const points = outlinePoints(polygon);
  context.beginPath();
  points.forEach((point, index) => {
    const x = point[0] * scaleX;
    const y = point[1] * scaleY;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.closePath();
}

function drawShape(context, polygon, scaleX, scaleY, fill, stroke, width) {
  polygonPath(context, polygon, scaleX, scaleY);
  context.fillStyle = fill;
  context.fill();
  context.strokeStyle = stroke;
  context.lineWidth = width;
  context.stroke();
}

function drawCircle(context, x, y, radius, fill, stroke, width) {
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  if (fill) {
    context.fillStyle = fill;
    context.fill();
  }
  if (stroke && width) {
    context.strokeStyle = stroke;
    context.lineWidth = width;
    context.stroke();
  }
}

function drawOverlay() {
  fitCanvasFrame();
  const surface = overlaySurface();
  if (!surface) return;
  const { context, width, height } = surface;
  if (!state.image || !state.image.width || !state.image.height) return;
  const scaleX = width / state.image.width;
  const scaleY = height / state.image.height;

  state.objects.forEach((object) => {
    if (!object.polygon || object.polygon.length < 3) return;
    const selected = object.uid === state.selectedUid;
    drawShape(
      context,
      object.polygon,
      scaleX,
      scaleY,
      selected ? COLOURS.selectedFill : COLOURS.shapeFill,
      selected ? COLOURS.selectedStroke : COLOURS.shapeStroke,
      selected ? OUTLINE_SELECTED_PX : OUTLINE_PX,
    );
  });

  if (state.preview && state.preview.polygon && state.preview.polygon.length >= 3) {
    drawShape(
      context,
      state.preview.polygon,
      scaleX,
      scaleY,
      COLOURS.previewFill,
      COLOURS.previewStroke,
      OUTLINE_PX,
    );
  }

  state.points.forEach((point) => {
    drawCircle(
      context,
      point.x * scaleX,
      point.y * scaleY,
      PROMPT_RADIUS_PX,
      COLOURS.promptFill,
      COLOURS.promptStroke,
      1.5,
    );
  });

  state.colorCenters.forEach(([x, y]) => {
    drawCircle(context, x * scaleX, y * scaleY, PROMPT_RADIUS_PX * 0.5,
      COLOURS.colorCenter, null, 0);
  });

  if (state.rulerTicks.length >= 2) {
    const [startX, startY] = state.rulerTicks[0];
    const [endX, endY] = state.rulerTicks[state.rulerTicks.length - 1];
    const length = Math.hypot(endX - startX, endY - startY) || 1;
    const normalX = (startY - endY) / length;
    const normalY = (endX - startX) / length;

    // Ruler line and every tick go into a single path with one stroke call: a
    // 40 cm ruler carries well over a hundred ticks.
    context.strokeStyle = COLOURS.ruler;
    context.lineWidth = OUTLINE_PX * 0.7;
    context.beginPath();
    state.rulerTicks.forEach(([x, y], index) => {
      const px = x * scaleX;
      const py = y * scaleY;
      if (index === 0) context.moveTo(px, py);
      else context.lineTo(px, py);
    });
    state.rulerTicks.forEach(([x, y]) => {
      const px = x * scaleX;
      const py = y * scaleY;
      context.moveTo(px - normalX * RULER_TICK_HALF_PX, py - normalY * RULER_TICK_HALF_PX);
      context.lineTo(px + normalX * RULER_TICK_HALF_PX, py + normalY * RULER_TICK_HALF_PX);
    });
    context.stroke();
  }
}

// Several state changes can land inside one interaction, so redraws are
// coalesced into a single frame.
let overlayFrame = 0;

function renderOverlay() {
  if (typeof window.requestAnimationFrame !== "function") {
    drawOverlay();
    return;
  }
  if (overlayFrame) return;
  overlayFrame = window.requestAnimationFrame(() => {
    overlayFrame = 0;
    drawOverlay();
  });
}

function imageCoordinates(event) {
  const rect = elements.workspacePreview.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const x = ((event.clientX - rect.left) / rect.width) * state.image.width;
  const y = ((event.clientY - rect.top) / rect.height) * state.image.height;
  if (x < 0 || y < 0 || x > state.image.width || y > state.image.height) return null;
  return { x: Math.round(x), y: Math.round(y) };
}

async function addPoint(event) {
  if (!state.clickMode || !state.image || state.busy) return;
  if (state.embedding.state !== "ready") {
    showToast(t("toast.embeddingNotReady"));
    return;
  }
  const position = imageCoordinates(event);
  if (!position) return;
  state.points.push({ x: position.x, y: position.y, label: 1 });
  renderOverlay();
  setBusy(true, t("busy.segment"));
  startProgress("progress.segment", 3);
  try {
    const payload = await postJson("/api/analysis/click", {
      ...imageRef(),
      points: state.points,
      commit: false,
    });
    if (!payload.ok) {
      state.points.pop();
      state.preview = null;
      showToast(t("toast.noObjectHere"));
    } else {
      state.preview = payload.preview;
    }
  } catch (error) {
    state.points.pop();
    showToast(error.message);
  } finally {
    endProgress(true);
    setBusy(false);
    renderOverlay();
    updateControls();
  }
}

async function commitObject() {
  if (!state.points.length || state.busy) return;
  setBusy(true, t("busy.commit"));
  startProgress("progress.segment", 2);
  try {
    const payload = await postJson("/api/analysis/click", {
      ...imageRef(),
      points: state.points,
      commit: true,
    });
    if (!payload.ok) {
      showToast(t("toast.commitFailed"));
      return;
    }
    state.objects = payload.objects;
    state.selectedUid = payload.object.uid;
    resetPoints();
    renderObjects();
    showToast(t("toast.created", { label: payload.object.label }));
  } catch (error) {
    showToast(error.message);
  } finally {
    endProgress(true);
    setBusy(false);
    renderOverlay();
    updateControls();
  }
}

function resetPoints() {
  state.points = [];
  state.preview = null;
  renderOverlay();
  updateControls();
}

function setClickMode(enabled) {
  state.clickMode = enabled;
  elements.pickObject.classList.toggle("is-active", enabled);
  elements.pickObject.setAttribute("aria-pressed", String(enabled));
  elements.canvasFrame.classList.toggle("is-picking", enabled);
  if (!enabled) resetPoints();
  renderEmbedding();
}

function setBusy(busy, message) {
  state.busy = busy;
  if (busy && message) elements.canvasHint.textContent = message;
  if (!busy) renderEmbedding();
  updateControls();
}

// ---------------------------------------------------------------------------
// Objects, scale, traits
// ---------------------------------------------------------------------------

function renderObjects() {
  const list = elements.objectList;
  list.replaceChildren();
  if (!state.objects.length) {
    const empty = document.createElement("div");
    empty.className = "object-empty";
    empty.textContent = t("panel.noObjects");
    list.append(empty);
    updateControls();
    renderOverlay();
    return;
  }

  state.objects.forEach((object) => {
    const row = document.createElement("div");
    row.className = object.uid === state.selectedUid ? "object-row is-selected" : "object-row";

    const name = document.createElement("button");
    name.type = "button";
    name.className = "object-name";
    name.textContent = object.label;
    name.title = `${object.area_pixels} px`;
    name.addEventListener("click", () => {
      state.selectedUid = object.uid;
      renderObjects();
    });

    const area = document.createElement("span");
    area.className = "object-area";
    area.textContent = `${(object.area_pixels / 1000).toFixed(1)} kpx`;

    const rename = document.createElement("button");
    rename.type = "button";
    rename.className = "object-action";
    rename.textContent = t("object.rename");
    rename.addEventListener("click", () => renameObject(object));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "object-action is-danger";
    remove.textContent = t("object.delete");
    remove.addEventListener("click", () => deleteObject(object));

    row.append(name, area, rename, remove);
    list.append(row);
  });
  updateControls();
  renderOverlay();
}

async function renameObject(object) {
  const label = window.prompt(t("prompt.rename"), object.label);
  if (label === null) return;
  try {
    await request(`/api/analysis/objects/${object.uid}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...imageRef(), label }),
    });
    await refreshAnalysis();
  } catch (error) {
    showToast(error.message);
  }
}

async function deleteObject(object) {
  try {
    const payload = await request(
      `/api/analysis/objects/${object.uid}?session_id=${encodeURIComponent(state.sessionId)}&image_id=${encodeURIComponent(
        state.image.id
      )}`,
      { method: "DELETE" }
    );
    state.objects = payload.objects;
    if (state.selectedUid === object.uid) state.selectedUid = null;
    state.traits = null;
    renderTraits();
    renderObjects();
  } catch (error) {
    showToast(error.message);
  }
}

async function clearObjects() {
  if (!state.objects.length) return;
  if (!window.confirm(t("confirm.clearAll", { count: state.objects.length }))) return;
  for (const object of [...state.objects]) {
    await deleteObject(object);
  }
}

function renderScale() {
  const scale = state.scale || { per_pixel: 1, unit: "pixel" };
  const value = Number(scale.per_pixel);
  const text = value === 1 && scale.unit === "pixel" ? "1 pixel = 1 pixel" : `1 pixel = ${value} ${scale.unit}`;
  elements.scaleReadout.textContent = text;
}

async function editScale() {
  const current = state.scale || { per_pixel: 1, unit: "pixel" };
  const raw = window.prompt(t("prompt.scaleValue"), String(current.per_pixel));
  if (raw === null) return;
  const perPixel = Number(raw);
  if (!Number.isFinite(perPixel) || perPixel <= 0) {
    showToast(t("toast.positiveNumber"));
    return;
  }
  const unit = window.prompt(t("prompt.scaleUnit"), current.unit === "pixel" ? "mm" : current.unit);
  if (unit === null) return;
  try {
    const payload = await postJson("/api/analysis/scale", {
      ...imageRef(),
      per_pixel: perPixel,
      unit: unit.trim() || "pixel",
    });
    state.scale = payload.scale;
    state.traits = null;
    renderScale();
    renderTraits();
    showToast(t("toast.scaleUpdated"));
  } catch (error) {
    showToast(error.message);
  }
}

async function runSemanticDetection() {
  if (!state.image || state.busy) return;
  if (state.embedding.state !== "ready") {
    showToast(t("toast.embeddingNotReady"));
    return;
  }
  // Default in the current language: the point of the Chinese vocabulary is that
  // nobody has to think in English to use this.
  const prompt = window.prompt(t("prompt.semantic"), state.lang === "zh" ? "豆荚" : "pod");
  if (prompt === null || !prompt.trim()) return;
  await requestSemanticDetection(prompt.trim(), false);
}

async function requestSemanticDetection(prompt, skipCheck) {
  setBusy(true, t("busy.semantic"));
  // Grounding DINO runs on the CPU: the cross-check probe plus the prompt pass
  // take 13-17 s on a 16 MPx photo.
  startProgress("progress.semantic", 17);
  try {
    const payload = await postJson("/api/analysis/semantic", {
      ...imageRef(),
      prompt,
      skip_semantic_check: skipCheck,
    });

    // The detector cross-checks the prompt against the image first and creates
    // nothing when it disagrees, the same as the desktop warning dialog.
    if (payload.ok === false && payload.reason === "semantic_mismatch") {
      const info = payload.mismatch;
      const message = t("confirm.semanticMismatch", {
        requested: info.requested.join(", "),
        suggested: info.suggested,
        count: info.suggested_count,
        score: info.suggested_score.toFixed(2),
        prompt,
      });
      endProgress(true);
      setBusy(false);
      if (window.confirm(message)) await requestSemanticDetection(prompt, true);
      return;
    }

    state.objects = payload.objects;
    state.traits = null;
    state.selectedUid = payload.created.length ? payload.created[payload.created.length - 1].uid : state.selectedUid;
    renderObjects();
    renderTraits();
    const filtered = payload.filtered ? t("toast.filteredSuffix", { count: payload.filtered }) : "";
    // Show the rewrite when a Chinese prompt was translated: without it a result
    // that does not match the wording is inexplicable.
    const translated = (payload.translated || [])
      .map((entry) => `${entry.typed} → ${entry.used}`)
      .join("; ");
    const suffix = translated ? t("toast.translatedSuffix", { translated }) : "";
    showToast(
      (payload.created.length
        ? t("toast.semanticNew", { count: payload.created.length, filtered })
        : t("toast.semanticNone", { filtered })) + suffix
    );
  } catch (error) {
    showToast(error.message);
  } finally {
    endProgress(true);
    setBusy(false);
  }
}

async function runColorCorrection() {
  if (!state.image || state.busy) return;
  if (state.colorCorrection) {
    showToast(t("toast.colorAlready"));
    return;
  }
  setBusy(true, t("busy.color"));
  startProgress("progress.color", 6);
  try {
    const payload = await postJson("/api/analysis/color", imageRef());
    state.colorCorrection = true;
    state.colorCenters = payload.centers || [];
    state.correctedImageUrl = payload.image_url;
    state.traits = null;
    elements.workspacePreview.src = payload.image_url;
    renderTraits();
    renderOverlay();
    updateControls();
    const quality = payload.quality;
    showToast(
      quality
        ? t("toast.colorDoneQuality", {
            count: state.colorCenters.length,
            before: quality.before_delta_e,
            after: quality.mean_delta_e,
          })
        : t("toast.colorDone", { count: state.colorCenters.length })
    );
  } catch (error) {
    showToast(error.message);
  } finally {
    endProgress(true);
    setBusy(false);
  }
}

async function runRulerDetection() {
  if (!state.image || state.busy) return;
  setBusy(true, t("busy.ruler"));
  startProgress("progress.ruler", 6);
  try {
    const payload = await postJson("/api/analysis/ruler", imageRef());
    state.rulerTicks = payload.tickpoints || [];
    state.scale = payload.scale;
    state.traits = null;
    renderScale();
    renderTraits();
    renderOverlay();
    const uneven = payload.uniformity && payload.uniformity.uneven;
    showToast(
      uneven
        ? t("toast.rulerUneven", {
            count: state.rulerTicks.length,
            drift: payload.uniformity.drift_percent,
          })
        : t("toast.rulerDone", { count: state.rulerTicks.length })
    );
  } catch (error) {
    showToast(error.message);
  } finally {
    endProgress(true);
    setBusy(false);
  }
}

function renderTraits() {
  const wrap = elements.traitWrap;
  wrap.replaceChildren();
  if (!state.traits || !state.traits.rows.length) {
    const empty = document.createElement("div");
    empty.className = "object-empty";
    empty.textContent = t("panel.noTraits");
    wrap.append(empty);
    return;
  }

  const table = document.createElement("table");
  table.className = "trait-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  // Header names stay as the API returns them: they are the CSV columns and the
  // desktop uses the same English identifiers.
  state.traits.headers.forEach((header) => {
    const cell = document.createElement("th");
    cell.textContent = header;
    headRow.append(cell);
  });
  head.append(headRow);

  const body = document.createElement("tbody");
  state.traits.rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent =
        index === 0 || value === null || typeof value !== "number" ? value ?? "-" : value.toFixed(3);
      tr.append(cell);
    });
    body.append(tr);
  });

  table.append(head, body);
  wrap.append(table);
}

async function calculateTraits() {
  if (!state.objects.length) return;
  setBusy(true, t("busy.traits"));
  startProgress("progress.traits", 3);
  try {
    state.traits = await postJson("/api/analysis/traits", imageRef());
    renderTraits();
    setDetail("traits");
    showToast(t("toast.traitsDone", { count: state.traits.rows.length }));
  } catch (error) {
    showToast(error.message);
  } finally {
    endProgress(true);
    setBusy(false);
  }
}

function exportAnnotation() {
  if (!state.image) return;
  const url = `/api/analysis/annotation?session_id=${encodeURIComponent(state.sessionId)}&image_id=${encodeURIComponent(
    state.image.id
  )}`;
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.image.filename.replace(/\.[^.]+$/, "")}.pheno.json`;
  link.click();
}

// One CSV path only, and it is the server's: it recomputes from the current
// masks and scale, so the file cannot disagree with the table on screen the way
// a separately built client-side export could.
async function exportTraitsTable() {
  if (!state.image || !state.sessionId) return;
  const url = `/api/analysis/results?session_id=${encodeURIComponent(state.sessionId)}&image_id=${encodeURIComponent(
    state.image.id
  )}`;
  const name = `${state.image.filename.replace(/\.[^.]+$/, "")}_traits.csv`;
  // Some mobile browsers (Quark, older Safari) ignore Content-Disposition and
  // render text/csv inline. Fetching it and handing over an object URL with a
  // download attribute makes them save the file instead.
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(t("error.request", { status: response.status }));
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 10000);
    showToast(t("toast.downloadStarted"));
  } catch (error) {
    showToast(error.message);
  }
}

function updateControls() {
  const hasImage = Boolean(state.image);
  const ready = hasImage && state.embedding && state.embedding.state === "ready";
  // The pick button carries the whole click flow: start picking, save the
  // object it is previewing, or leave the mode again.
  const canSave = state.clickMode && Boolean(state.preview) && state.points.length > 0;
  const pickKey = canSave ? "pick.save" : state.clickMode ? "pick.exit" : "pick.start";
  elements.pickObject.textContent = t(pickKey);
  elements.pickObject.dataset.i18n = pickKey;
  elements.pickObject.classList.toggle("command-primary", canSave);
  elements.pickObject.disabled = !ready || state.busy;
  elements.resetPoints.disabled = !state.points.length || state.busy;
  elements.setScale.disabled = !hasImage || state.busy;
  elements.semanticDetect.disabled = !ready || state.busy;
  elements.calcTraits.disabled = !state.objects.length || state.busy;
  elements.exportAnnotation.disabled = !hasImage;
  elements.exportResults.disabled = !state.objects.length || state.busy;
  elements.clearObjects.disabled = !state.objects.length || state.busy;
  elements.clearImage.disabled = !hasImage || state.busy;
  elements.colorChecker.disabled = !hasImage || state.busy || state.colorCorrection;
  elements.rulerDetection.disabled = !hasImage || state.busy;
}

// ---------------------------------------------------------------------------
// Image input
// ---------------------------------------------------------------------------

function renderImageMeta() {
  elements.workspaceFileList.replaceChildren();
  if (!state.image) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = t("panel.emptyImage");
    elements.workspaceFileList.append(empty);
    elements.fileCount.textContent = "0";
    elements.workspaceDimensions.textContent = "0 x 0";
    return;
  }
  elements.workspaceDimensions.textContent = `${state.image.width} x ${state.image.height}`;
  const item = document.createElement("div");
  item.className = "file-item";
  item.textContent = state.image.filename;
  item.title = state.image.filename;
  elements.workspaceFileList.append(item);
  elements.fileCount.textContent = "1";
}

// Everything derived from the previous image is dropped the moment a new one is
// chosen, not after the upload finishes: on a phone the upload takes seconds to
// a minute, and until now the old outlines sat on top of the new photo the whole
// time. The old image is also released on the server so it stops holding its
// embedding and its slot in the session.
function resetAnalysisState() {
  const previous = state.image;
  state.image = null;
  state.objects = [];
  state.traits = null;
  state.selectedUid = null;
  state.embedding = { state: "idle" };
  state.scale = { per_pixel: 1, unit: "pixel" };
  state.colorCenters = [];
  state.colorCorrection = false;
  state.rulerTicks = [];
  state.correctedImageUrl = null;
  resetPoints();
  setClickMode(false);
  // Drop the old picture too: if the new one fails to decode, an <img> keeps
  // showing the last image that loaded, which reads as "it worked".
  elements.workspacePreview.removeAttribute("src");
  elements.workspaceStage.classList.remove("has-image");
  renderImageMeta();
  renderObjects();
  renderTraits();
  renderScale();
  renderEmbedding();
  if (previous) {
    fetch(`/api/images/${previous.id}`, { method: "DELETE" }).catch(() => {});
  }
}

// `blob` is whatever the user chose (a File from the picker is already a Blob) and
// `name` is the file name sent to the server. They are kept apart because the
// multipart part is built with FormData.append(field, blob, name): wrapping a
// fetched Blob in `new File(...)` and appending that produced a zero-byte part on
// the phone browser ("The uploaded image is empty"), which is what broke the
// sample images.
async function selectImage(blob, name, mimeType) {
  if (!blob) return;
  const filename = name || blob.name || "capture.jpg";
  // A blob fetched from our own static files can arrive with an empty type on
  // some browsers, so the caller may state it instead.
  const type = blob.type || mimeType || "";
  if (!type.startsWith("image/")) {
    showToast(t("toast.notImage"));
    return;
  }
  if (!blob.size) {
    showToast(t("sample.empty"));
    return;
  }
  resetAnalysisState();
  if (state.localUrl) URL.revokeObjectURL(state.localUrl);
  state.localUrl = URL.createObjectURL(blob);
  state.selectedFile = blob;
  state.selectedName = filename;
  state.selectedType = type;

  elements.workspacePreview.src = state.localUrl;
  elements.workspaceStage.classList.add("has-image");

  await uploadSelectedImage();
}

// The two bundled photographs go through exactly the same upload and analysis
// path as a camera shot; only the source of the bytes differs.
// XHR with responseType "blob" as a second opinion: the phone browser handed back
// a 200 with a zero-byte body for the fetch() of the sample, which is what made
// the upload fail with "The uploaded image is empty". The fetch no longer passes
// a force-cache option either, since that was the only unusual thing about it.
function fetchBlobViaXhr(url) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", url);
    xhr.responseType = "blob";
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300 && xhr.response) resolve(xhr.response);
      else reject(new Error(t("error.request", { status: xhr.status })));
    });
    xhr.addEventListener("error", () => reject(new Error(t("error.network"))));
    xhr.send();
  });
}

async function loadSample(name) {
  if (state.busy) return;
  const url = `/static/samples/${name}.jpg?v=${STATIC_VERSION}`;
  setBusy(true, t("sample.loading"));
  startProgress("progress.sample", 3);
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(t("error.request", { status: response.status }));
    let blob = await response.blob();
    if (!blob.size) blob = await fetchBlobViaXhr(url);
    if (!blob.size) throw new Error(t("sample.empty"));
    endProgress(true);
    setBusy(false);
    await selectImage(blob, `sample_${name}.jpg`, "image/jpeg");
  } catch (error) {
    endProgress(false);
    setBusy(false);
    showToast(error.message);
  }
}

// Last-resort repair path. If the server says the multipart part was empty or the
// type is unsupported, re-draw the picture into a canvas and send a fresh JPEG.
// That works around browsers whose FormData serialises certain Blobs as zero
// bytes, and it also converts formats the API rejects (HEIC from an iPhone).
function reencodeToJpeg(blob, maxSide = 4096, quality = 0.92) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      canvas.toBlob(
        (result) => (result && result.size ? resolve(result) : reject(new Error(t("sample.empty")))),
        "image/jpeg",
        quality
      );
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(t("toast.notImage")));
    };
    image.src = url;
  });
}

function looksLikeBadUpload(message) {
  return /empty|not a valid image|Supported image types/i.test(message || "");
}

async function uploadSelectedImage(retried = false) {
  if (!state.selectedFile) return;
  if (!state.sessionId) await createSession();
  const formData = new FormData();
  const blob =
    state.selectedFile.type || !state.selectedType
      ? state.selectedFile
      : state.selectedFile.slice(0, state.selectedFile.size, state.selectedType);
  formData.append("file", blob, state.selectedName || "capture.jpg");
  startProgress("progress.upload");
  setProgress(0.02);
  try {
    const payload = await uploadWithProgress(
      `/api/images?session_id=${encodeURIComponent(state.sessionId)}`,
      formData,
      (fraction) => setProgress(Math.max(0.02, fraction * 0.98))
    );
    endProgress(true);
    // selectImage() already cleared the previous analysis; only the new image
    // metadata is missing at this point.
    state.image = payload;
    renderImageMeta();
    renderObjects();
    renderTraits();
    renderScale();
    // Keep showing the local blob rather than re-fetching `payload.url`: over the
    // public tunnel that would pull the whole original photo back down again,
    // doubling the traffic of every upload for a pixel-identical picture.
    if (!state.localUrl) elements.workspacePreview.src = payload.url;
    showToast(t("toast.uploaded"));
    await prepareEmbedding();
  } catch (error) {
    endProgress(false);
    if (!retried && looksLikeBadUpload(error.message)) {
      try {
        const repaired = await reencodeToJpeg(state.selectedFile);
        state.selectedFile = repaired;
        state.selectedName = (state.selectedName || "capture").replace(/\.[^.]+$/, "") + ".jpg";
        await uploadSelectedImage(true);
        return;
      } catch (_) {
        // Fall through to the original error, which is the more useful one.
      }
    }
    showToast(error.message);
  }
}

async function clearImage() {
  if (state.image) {
    try {
      await fetch(`/api/images/${state.image.id}`, { method: "DELETE" });
    } catch (_) {
      // The local view can always be cleared, even if the session is gone.
    }
  }
  if (state.localUrl) URL.revokeObjectURL(state.localUrl);
  state.localUrl = null;
  state.selectedFile = null;
  state.image = null;
  state.objects = [];
  state.traits = null;
  state.embedding = { state: "idle" };
  state.scale = { per_pixel: 1, unit: "pixel" };
  state.colorCenters = [];
  state.colorCorrection = false;
  state.rulerTicks = [];
  state.correctedImageUrl = null;
  resetPoints();
  setClickMode(false);
  elements.workspacePreview.removeAttribute("src");
  elements.workspaceStage.classList.remove("has-image");
  elements.workspaceFile.value = "";
  renderImageMeta();
  renderObjects();
  renderTraits();
  renderScale();
}

async function createSession() {
  try {
    const payload = await postJson("/api/sessions", { source: "workspace" });
    state.sessionId = payload.id;
  } catch (_) {
    state.sessionId = null;
    showToast(t("toast.noSession"));
  }
}

async function checkService() {
  try {
    const payload = await request("/health");
    elements.statusDot.classList.add("is-online");
    elements.serviceStatus.textContent = `${t("status.online")} · ${
      payload.model_state === "ready" ? t("model.ready") : t("model.notLoaded")
    }`;
  } catch (_) {
    elements.statusDot.classList.remove("is-online");
    elements.serviceStatus.textContent = t("status.offline");
  }
  await refreshModel();
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

elements.detailTabs.forEach((tab) => tab.addEventListener("click", () => setDetail(tab.dataset.detailTarget)));
elements.languageToggle.addEventListener("click", () => setLanguage(state.lang === "zh" ? "en" : "zh"));
elements.helpOpen.addEventListener("click", openHelp);
elements.helpClose.addEventListener("click", closeHelp);
elements.helpOverlay.addEventListener("click", (event) => {
  if (event.target === elements.helpOverlay) closeHelp();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.helpOverlay.hidden) closeHelp();
});
elements.workspaceFile.addEventListener("change", (event) => selectImage(event.target.files[0]));
elements.clearImage.addEventListener("click", clearImage);
elements.loadModel.addEventListener("click", loadModel);
elements.pickObject.addEventListener("click", () => {
  if (state.clickMode && state.preview && state.points.length) commitObject();
  else setClickMode(!state.clickMode);
});
// Every sample button carries its own name, so adding one stays a markup-only
// change: binding them by id meant a new button silently did nothing.
document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => loadSample(button.dataset.sample));
});
elements.resetPoints.addEventListener("click", resetPoints);
elements.setScale.addEventListener("click", editScale);
elements.semanticDetect.addEventListener("click", runSemanticDetection);
elements.colorChecker.addEventListener("click", runColorCorrection);
elements.rulerDetection.addEventListener("click", runRulerDetection);
elements.calcTraits.addEventListener("click", calculateTraits);
elements.exportAnnotation.addEventListener("click", exportAnnotation);
elements.exportResults.addEventListener("click", exportTraitsTable);
elements.clearObjects.addEventListener("click", clearObjects);

elements.canvasFrame.addEventListener("click", addPoint);

window.addEventListener("resize", renderOverlay);
// The canvas is sized from the displayed image, so it has to be redrawn once
// the picture (and therefore the layout) is known.
elements.workspacePreview.addEventListener("load", renderOverlay);

// Calibration is a once-per-image step, so it starts folded on a phone and open
// where there is room for it.
if (window.matchMedia("(min-width: 901px)").matches) elements.calibrateGroup.open = true;

setLanguage(initialLanguage());
createSession();
checkService();

// First visit in this browser opens the guide; afterwards it is behind the "?".
if (!helpAlreadySeen()) openHelp();
