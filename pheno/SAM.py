import time
import cv2
import hashlib
import numpy as np
import os
from qtpy.QtCore import QRect, QSettings

from .outliers import (
    MIN_GROUP_FOR_OUTLIERS,
    area_is_outlier,
    axis_is_outlier,
    mask_major_axis,
)

# Import CUDA memory management
try:
    from .cuda import check_memory, cleanup_gpu_memory, disable_cuda_for_session, get_optimal_device, print_memory_status
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False
    print("[WARN] SAM: CUDA memory manager unavailable")


# ---------------------------------------------------------------------------
# Segmenter backends. The rest of the app only talks to the SAM class below, so
# a new model family registers here instead of touching canvas/app code.
# ---------------------------------------------------------------------------

SEGMENTER_SAM1 = "sam_vit_h"
SEGMENTER_SAM2_TINY = "sam2_hiera_t"
SEGMENTER_SAM2_LARGE = "sam2_hiera_l"

_SEGMENTERS = {
    SEGMENTER_SAM1: {
        "label": "SAM ViT-H",
        "kind": "sam1",
        "weights": os.path.join("models", "sam_vit_h_4b8939.pth"),
    },
    SEGMENTER_SAM2_TINY: {
        "label": "SAM2.1 Hiera-Tiny",
        "kind": "sam2",
        "weights": os.path.join("models", "sam2", "sam2.1_hiera_tiny.pt"),
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
    },
    SEGMENTER_SAM2_LARGE: {
        "label": "SAM2.1 Hiera-Large",
        "kind": "sam2",
        "weights": os.path.join("models", "sam2", "sam2.1_hiera_large.pt"),
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
}

SEGMENTER_SETTING = "model/segmenter"


def _segmenter_usable(spec):
    if not os.path.isfile(spec["weights"]):
        return False
    try:
        if spec["kind"] == "sam2":
            import sam2  # noqa: F401
        else:
            import segment_anything  # noqa: F401
    except ImportError:
        return False
    return True


def available_segmenters():
    """Backends whose weights file and python package are both present."""
    return [key for key, spec in _SEGMENTERS.items() if _segmenter_usable(spec)]


def segmenter_label(key):
    return _SEGMENTERS.get(key, {}).get("label", key)


def default_segmenter():
    """Backend used when the user has not picked one at the startup screen.

    On a working GPU the heavyweight ViT-H costs nothing extra, so the
    accuracy-first model (and the one the paper's numbers were produced with)
    wins by default. On a CPU-only box SAM2-Tiny embeds roughly 6x faster with
    negligible quality loss on solid organs, so it becomes the default there.
    """
    keys = available_segmenters()
    if not keys:
        return SEGMENTER_SAM1
    if str(QSettings('pheno', 'pheno').value('model/device', 'auto')) == 'cpu':
        device = "cpu"
    elif MEMORY_MANAGER_AVAILABLE:
        device = get_optimal_device()
    else:
        device = "cuda" if os.getenv('USE_GPU', 'false').lower() == 'true' else "cpu"
    if device.startswith('cuda'):
        # GPU removes the speed argument, so the strongest mask model wins.
        # Officially SAM2-Large is both faster and slightly more accurate than
        # SAM1 ViT-H; SAM1 stays as the fallback when its weights are missing.
        if SEGMENTER_SAM2_LARGE in keys:
            return SEGMENTER_SAM2_LARGE
        return SEGMENTER_SAM1 if SEGMENTER_SAM1 in keys else keys[0]
    return SEGMENTER_SAM2_TINY if SEGMENTER_SAM2_TINY in keys else keys[0]


def current_segmenter_key():
    """The model the SAM class will load, honouring the user's setting."""
    wanted = str(QSettings('pheno', 'pheno').value(SEGMENTER_SETTING, ''))
    usable = available_segmenters()
    if wanted in usable:
        return wanted
    return default_segmenter()


def cropAndResize(image, pos1, pos2, max_side=1024):
    """
    裁剪图像，并将最长边缩放到 `max_side`。

    :param image: np.ndarray - 输入的原始图像 (H, W, C)
    :param pos1: QPoint - 裁剪区域左上角
    :param pos2: QPoint - 裁剪区域右下角
    :param max_side: int - 目标最长边（默认 1024）
    :return: (resized_image, crop_rect) - 缩放后的图像 & QRect（原始裁剪区域）
    """
    # 获取裁剪区域坐标
    crop_x1, crop_y1 = pos1.x(), pos1.y()
    crop_x2, crop_y2 = pos2.x(), pos2.y()

    # 计算裁剪区域 (x, y, width, height)
    crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
    crop_rect = QRect(crop_x1, crop_y1, crop_w, crop_h)  # 记录裁剪区域

    # 裁剪图像
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # 计算缩放比例
    scale = max_side / max(crop_h, crop_w)
    new_h, new_w = int(crop_h * scale), int(crop_w * scale)

    # 缩放图像
    resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return resized_image, crop_rect
        
def restore(mask, crop_rect, original_shape):
    """
    还原 `mask` 到 `crop_rect` 在 `original_shape` 里的原始位置。

    :param mask: np.ndarray - 预测出的 `mask` (new_h, new_w)
    :param crop_rect: QRect - `cropAndResize` 返回的 `crop_rect`
    :param original_shape: tuple(int, int) - 原始图像尺寸 (H, W)
    :return: np.ndarray - 还原回原始大小的 `mask` (H, W)
    """
    H, W = original_shape[:2]  # 原始图像尺寸
    crop_x, crop_y, crop_w, crop_h = crop_rect.x(), crop_rect.y(), crop_rect.width(), crop_rect.height()

    # 计算 `mask` 逆变换回 `crop_rect` 的大小
    restored_crop_mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    # 创建一个全黑 `mask`，并将 `restored_crop_mask` 贴回 `crop_rect` 位置
    full_size_mask = np.zeros((H, W), dtype=np.uint8)
    full_size_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = restored_crop_mask

    return full_size_mask

def calculate_hash(image):
    """计算图像的感知哈希（pHash）。"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 调整大小为8x8（哈希计算的标准尺寸）
    resized = cv2.resize(gray, (8, 8))
    # 计算离散余弦变换（DCT）
    dct = cv2.dct(np.float32(resized))
    # 取DCT左上角的8x8（低频部分，代表图像主要特征）
    dct_roi = dct[0:8, 0:8]
    # 计算DCT系数的均值
    avg = np.mean(dct_roi)
    # 生成哈希：大于均值的为1，否则为0
    hash_value = (dct_roi > avg).flatten().astype(int)
    return hash_value

def _dedupe_view(mask):
    """Boolean mask, subsampled while it is big.

    Dedupe ratios (IoU/overlap against 0.3/0.5 bands) tolerate the few-percent
    sampling error; a full-resolution pairwise pass over 12-megapixel masks
    does not -- it cost tens of seconds on a 40-mask batch. Slicing is a view,
    so the boolean conversion only ever runs on the small result.
    """
    view = np.asarray(mask)
    while view.size > 1_000_000:
        view = view[::2, ::2]
    return view > 0


def compute_iou(mask1, mask2):
    """计算两个 mask 之间的 IoU（交并比）"""
    a = _dedupe_view(mask1)
    b = _dedupe_view(mask2)
    if a.shape != b.shape:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return intersection / union if union > 0 else 0


def mask_overlap_ratio(mask_a, mask_b):
    """Intersection divided by the *smaller* mask's area.

    IoU misses containment: a cluster-level blob that swallows twenty seed
    masks still scores ~0.05 IoU against each of them, so a plain IoU dedupe
    keeps the blob *and* the seeds it contains. Measuring the overlap against
    the smaller mask scores 1.0 there, which is what dedupe actually needs.
    """
    a = _dedupe_view(mask_a)
    b = _dedupe_view(mask_b)
    if a.shape != b.shape:
        return 0.0
    smaller = min(int(a.sum()), int(b.sum()))
    if smaller == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / smaller


class SAM:
    def __init__(self, model_key=None):
        print("Initializing SAM model...")
        self.model_key = model_key or current_segmenter_key()
        print(f"Segmenter backend: {segmenter_label(self.model_key)}")

        # Determine device
        if MEMORY_MANAGER_AVAILABLE:
            device = get_optimal_device()
            print(f"SAM will use device: {device}")
        else:
            device = "cuda" if os.getenv('USE_GPU', 'false').lower() == 'true' else "cpu"

        self.device = device

        # If using GPU, check memory
        if device.startswith('cuda') and MEMORY_MANAGER_AVAILABLE:
            if not check_memory(required_gb=2.0):  # SAM needs approximately 2GB
                print("[WARN] Insufficient GPU memory, switching to CPU mode")
                device = "cpu"
                self.device = device

        # Load model. Absent weights must not kill the app: a failed load
        # just leaves predictor=None and every segmentation action reports
        # a clear "weights not installed" error instead (see _infer).
        self.sam = None
        self.predictor = None
        self._mask_generator_cls = None
        try:
            self._load_backend(device)

            if MEMORY_MANAGER_AVAILABLE:
                print_memory_status()

            print(f"[OK] SAM model initialized successfully, using device: {device}")

        except Exception as e:
            print(f"[FAIL] SAM model initialization failed: {e}")
            if device.startswith('cuda'):
                print("Trying CPU mode...")
                device = "cpu"
                self.device = device
                try:
                    self._load_backend(device)
                    print("[OK] CPU mode initialization successful")
                except Exception as e2:
                    print(f"[FAIL] CPU fallback also failed: {e2}")
        
        self.input_point = []
        self.input_label = []
        self.current_image_hash = None
        self.scale = 1.0
        self.scaled_image = None
        self.raw_shape = (None, None)

    def _switch_to_cpu(self, cause):
        """Rebuild the model on CPU after an unrecoverable CUDA failure.

        A poisoned CUDA context can take the whole process down at the next
        kernel launch (observed on GPUs whose compute capability the bundled
        PyTorch does not cover, e.g. sm_120 cards on a cu124 build), so once a
        non-OOM device error escapes we reload on CPU and never touch CUDA
        again in this session.
        """
        print(f"[WARN] CUDA inference failed ({cause}); switching to CPU for the rest of this session")
        if MEMORY_MANAGER_AVAILABLE:
            disable_cuda_for_session(cause)
        self.device = "cpu"
        self._load_backend("cpu")
        try:
            import torch
            torch.cuda.empty_cache()  # release the stale GPU context's pool
        except Exception:
            pass
        if self.scaled_image is not None:
            self.predictor.set_image(self.scaled_image)
        print("[OK] SAM is now running on CPU")

    def _load_backend(self, device):
        """Instantiate model + predictor for the selected segmenter on `device`.

        Fills ``self.sam`` (the torch model), ``self.predictor`` (set_image /
        predict) and ``self._mask_generator_cls`` so the rest of the class does
        not care which SAM family is underneath.
        """
        spec = _SEGMENTERS[self.model_key]
        self._kind = spec["kind"]
        if spec["kind"] == "sam2":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            self.sam = build_sam2(spec["config"], spec["weights"], device=device)
            self.predictor = SAM2ImagePredictor(self.sam)
            self._mask_generator_cls = SAM2AutomaticMaskGenerator
        else:
            from segment_anything import (
                sam_model_registry, SamPredictor, SamAutomaticMaskGenerator)
            self.sam = sam_model_registry["vit_h"](checkpoint=spec["weights"])
            self.sam.to(device=device)
            self.predictor = SamPredictor(self.sam)
            self._mask_generator_cls = SamAutomaticMaskGenerator

    # Substrings that mark a genuine CUDA/device-level failure, as opposed to
    # usage bugs (shape mismatches, dtype errors) which would fail identically
    # on CPU and must not poison the session's device.
    _CUDA_ERROR_HINTS = (
        'cuda', 'cudnn', 'cublas', 'cufft', 'nccl', 'illegal memory',
        'device-side assert', 'misaligned address', 'kernel image',
        'no kernel', 'device error', 'gpu',
    )

    def _infer(self, op):
        """Run one model call; on a genuine CUDA failure, drop to CPU and retry."""
        if self.predictor is None:
            raise RuntimeError(
                "No segmentation model is installed. Download the model "
                "archive (models.zip), unzip it next to the application so "
                "that models/ exists, then restart PhenoMe."
            )
        try:
            return op()
        except Exception as e:
            if not self.device.startswith('cuda'):
                raise
            msg = str(e).lower()
            if isinstance(e, RuntimeError) and 'out of memory' in msg:
                raise
            if not any(h in msg for h in self._CUDA_ERROR_HINTS):
                raise
            self._switch_to_cpu(e)
            return op()

    def auto_predict(self, image, pos1, pos2, samples=3):
        
        def keep_largest_connected_component(mask):
            mask = mask.astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels <= 1:
                return mask.astype(bool)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # ignore background (label 0)
            return (labels == largest_label)

        resized_image, crop_rect = cropAndResize(image, pos1, pos2, min(max(image.shape[:2]), 1024))

        
        def _generate_masks():
            self.auto_predictor = self._mask_generator_cls(self.sam, points_per_side=samples//2+8, box_nms_thresh=0.9)
            return self.auto_predictor.generate(resized_image)

        masks = self._infer(_generate_masks)

        for mask in masks:
            mask["segmentation"] = keep_largest_connected_component(mask["segmentation"])
            mask["area"] = mask["segmentation"].sum()

        masks.sort(key=lambda x: x['area'], reverse=True)
        # print("Raw Mask Nums : ", len(masks))

        img_area = resized_image.shape[0] * resized_image.shape[1] - masks[0]['area']
        maxi_thresh = (img_area / samples) * 2
        mini_thresh = (img_area / samples) / 4

        # Check area
        masks = [mask for mask in masks if mask['area'] > mini_thresh and mask['area'] < maxi_thresh]
        
        unique_masks = []
        # 升序遍历：小的先入库，后到的大 blob 因 overlap≈1.0（吞掉已入库的
        # 小对象）被丢弃 —— 单靠 IoU<0.3 认不出这种包含关系。
        for mask in sorted(masks, key=lambda m: m['area']):
            mask_binary = mask['segmentation']

            ys, xs = np.where(mask_binary > 0)

            width = xs.max() - xs.min() + 1
            height = ys.max() - ys.min() + 1
            
            # Check Background
            if height > resized_image.shape[0] * 0.98 or width > resized_image.shape[1] * 0.98:
                continue
            
            # Check IoU + containment
            if all(compute_iou(mask_binary, kept['segmentation']) < 0.3
                   and mask_overlap_ratio(mask_binary, kept['segmentation']) < 0.5
                   for kept in unique_masks):
                unique_masks.append(mask)


        masks = sorted(unique_masks, key=lambda m: m['area'], reverse=True)
        # print("Filtered Mask Nums : ", len(masks))

        if len(masks) <= samples:
            return [restore(m["segmentation"].astype(np.uint8), crop_rect, image.shape[:2]) for m in masks]


        min_variance = float("inf")
        best_group = []

        for i in range(len(masks)-samples+1):
            group = masks[i : i + samples]
            areas = [m["area"] for m in group]
            variance = np.var(areas)

            if variance < min_variance:
                min_variance = variance
                best_group = group
        best_group = [restore(m["segmentation"].astype(np.uint8), crop_rect, image.shape[:2]) for m in best_group]
        return best_group

    def auto_predict_with_centers(self, image, pos1, pos2, centers):
        print("Auto Predict with centers...")
        mask_list = []
        for center in centers:
            self.add_points(center, 1)
            mask = self.predict(oneDomain=True)
            self.reset_points()
            if mask is None:
                continue
            foreground_indices = np.argwhere(mask == 1)
            if foreground_indices.size == 0:
                continue  # 如果没有前景区域，跳过
            
            # 获取前景区域的最小外接矩形
            x1, y1 = pos1.x(), pos1.y()
            x2, y2 = pos2.x(), pos2.y()
            y_min, x_min = foreground_indices.min(axis=0)
            y_max, x_max = foreground_indices.max(axis=0)
            if (x_min < x1 or x_max > x2 or y_min < y1 or y_max > y2):
                # 前景区域存在区域外，跳过
                continue
            mask_list.append(mask)

        # 去重放在收集完之后、按面积升序：单靠 IoU<0.3 认不出"大 blob 吞了
        # 若干小对象"（IoU≈小/大）。小对象先入库，后到的大 mask 因 overlap≈1.0
        # 被丢弃 —— 否则一个整簇 blob 会和它包着的籽粒一起被当成对象。
        deduped = []
        for mask in sorted(mask_list, key=lambda m: int(np.count_nonzero(m))):
            if all(compute_iou(mask, kept) < 0.3
                   and mask_overlap_ratio(mask, kept) < 0.5
                   for kept in deduped):
                deduped.append(mask)
        mask_list = deduped
        print("Auto detected masks: ", len(mask_list))
        # One shared rule with the other filters (pheno/outliers.py), including
        # its minimum group size: with only a few masks a median says nothing
        # about the sample, and filtering there used to drop the largest organ
        # of a small plate as an "outlier".
        if len(mask_list) < MIN_GROUP_FOR_OUTLIERS:
            return list(mask_list)
        median_area = float(np.median([np.sum(m > 0) for m in mask_list]))
        new_mask_list = [
            mask for mask in mask_list
            if not area_is_outlier(float(np.sum(mask > 0)), median_area)
        ]
        print("After area filtering, masks: ", len(new_mask_list))
        if len(new_mask_list) >= MIN_GROUP_FOR_OUTLIERS:
            axes = {id(mask): mask_major_axis(mask) for mask in new_mask_list}
            median_axis = float(np.median(list(axes.values())))
            new_mask_list = [
                mask for mask in new_mask_list
                if not axis_is_outlier(axes[id(mask)], median_axis)
            ]
            print("After axis filtering, masks: ", len(new_mask_list))
        return new_mask_list

    def set_image(self, image):
        """Set image with memory management"""
        if self.predictor is None:
            raise RuntimeError(
                "No segmentation model is installed. Download the model "
                "archive (models.zip), unzip it next to the application so "
                "that models/ exists, then restart PhenoMe."
            )
        image_hash = hashlib.md5(image).hexdigest()
        if self.current_image_hash == image_hash:
            print("Same image, skipping...")
            return
        self.current_image_hash = image_hash

        print("Setting SAM image...")
        
        # Image preprocessing
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        self.raw_shape = image.shape[:2]
        
        if max(image.shape[0], image.shape[1]) > 1024:
            self.scale = 1024.0 / max(image.shape[0], image.shape[1])
            self.scaled_image = cv2.resize(image, (int(image.shape[1]*self.scale), int(image.shape[0]*self.scale)), interpolation=cv2.INTER_LINEAR)
        else:
            self.scale = 1.0
            self.scaled_image = image


        try:
            # Set image
            self.predictor.set_image(self.scaled_image)

            if MEMORY_MANAGER_AVAILABLE and self.device.startswith('cuda'):
                print_memory_status()

            print("[OK] SAM image setting completed")


        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[FAIL] GPU out of memory: {e}")
                print("Suggestions:")
                print("1. Use smaller images")
                print("2. Restart the application")
                print("3. Close other GPU-using programs")
                raise RuntimeError("GPU out of memory, please try using smaller images or restart the application")
            elif self.device.startswith('cuda'):
                # Unsupported-GPU-class failure (e.g. no kernel image): the
                # switch already re-embeds scaled_image on the CPU predictor.
                self._switch_to_cpu(e)
            else:
                raise e
        except Exception as e:
            if self.device.startswith('cuda'):
                self._switch_to_cpu(e)
            else:
                raise


    def add_point(self, posx, posy, label):
        posx = int(posx * self.scale)
        posy = int(posy * self.scale)
        self.input_point.append([posx, posy])
        self.input_label.append(label)

    def add_points(self, points, label):
        if isinstance(points, tuple):
            self.add_point(points[0], points[1], label)
            return
        for i in range(len(points)):
            # print("Adding point: ", points[i])
            posx = int(points[i][0] * self.scale)
            posy = int(points[i][1] * self.scale)
            self.input_point.append([posx, posy])
            self.input_label.append(label)

    def pop_point(self):
        self.input_point.pop(-1)
        self.input_label.pop(-1)

    def reset_points(self):
        self.input_point = []
        self.input_label = []

    def predict_box(self, box):
        """Predict one full-resolution mask from an original-image XYXY box."""
        scaled_box = np.asarray(box, dtype=np.float32) * self.scale
        masks, scores, _ = self._infer(lambda: self.predictor.predict(
            box=scaled_box,
            multimask_output=False,
        ))
        if len(masks) == 0:
            return None

        mask = cv2.resize(
            masks[0].astype(np.uint8),
            (self.raw_shape[1], self.raw_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return mask

    def predict_boxes(self, boxes):
        """One full-resolution mask per original-image XYXY box, one model call.

        Both predictors accept an (N, 4) box array; batching N boxes into a
        single decoder pass roughly halves segmentation time on a full plate.
        Any predictor that refuses the batched shape falls back to the per-box
        loop, so behaviour is unchanged for unexpected mask shapes too.
        """
        boxes = [np.asarray(box, dtype=np.float32) for box in boxes]
        if not boxes:
            return []
        scaled = np.stack(boxes) * self.scale
        # Batched box prompts are only valid on the SAM2 predictor; SAM1's
        # SamPredictor.predict accepts a single box, and feeding it an (N,4)
        # array dies inside prompt concatenation with a shape error.
        try:
            if getattr(self, "_kind", "sam2") != "sam2":
                raise ValueError("sam1 predictor does not support batched boxes")
            masks, _scores, _ = self._infer(lambda: self.predictor.predict(
                box=scaled,
                multimask_output=False,
            ))
            masks = np.asarray(masks)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.ndim != 3 or masks.shape[0] != len(boxes):
                raise ValueError(f"unexpected batched mask shape {masks.shape}")
        except Exception:
            return [self.predict_box(box) for box in boxes]
        return [
            cv2.resize(
                mask.astype(np.uint8),
                (self.raw_shape[1], self.raw_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            for mask in masks
        ]

        mask = cv2.resize(
            masks[0].astype(np.uint8),
            (self.raw_shape[1], self.raw_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return mask

    def predict(self, oneDomain=False):
        """Predict masks with memory management"""
        
        try:        
            # Predict
            print("Point Coords: ", np.array(self.input_point) / self.scale)
            masks, _, _ = self._infer(lambda: self.predictor.predict(
                        point_coords=np.array(self.input_point),
                        point_labels=np.array(self.input_label),
                        multimask_output=False,
                    ))
            
            if oneDomain:
                mask = self.get_oneDomain(masks[0])
                if mask is None: return None
                mask = cv2.resize(mask.astype(np.uint8), 
                        (self.raw_shape[1], self.raw_shape[0]), 
                        interpolation=cv2.INTER_NEAREST)
                return mask
            else:
                mask = cv2.resize(masks[0].astype(np.uint8), 
                        (self.raw_shape[1], self.raw_shape[0]), 
                        interpolation=cv2.INTER_NEAREST)
                return mask
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[FAIL] Prediction failed due to GPU memory shortage: {e}")
                if MEMORY_MANAGER_AVAILABLE:
                    cleanup_gpu_memory()  # Emergency cleanup
                raise RuntimeError("GPU out of memory during prediction, please try using smaller images")
            else:
                raise e
    
    def get_oneDomain(self, mask):
        # Convert mask to binary image
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Use cv2.connectedComponents to find connected domains
        num_labels, labeled_mask = cv2.connectedComponents(binary_mask)

        # Get coordinates of positive points
        positive_points = np.array(self.input_point)[np.array(self.input_label) == 1]

        # Initialize a variable to save the connected domain containing all positive points
        valid_domain = None
        # Iterate through all connected domains
        for label in range(1, num_labels):  # Note: label 0 is background
            # Get current connected domain mask
            current_domain = (labeled_mask == label).astype(np.uint8)
            
            # Check if this connected domain contains all positive points
            contains_any_positive_points = np.any([current_domain[int(point[1]), int(point[0])] == 1 for point in positive_points])
            
            if contains_any_positive_points:
                valid_domain = current_domain
                break  # Exit after finding the connected domain containing all positive points
        # If no connected domain containing all positive points is found, return original mask
        if valid_domain is None:
            print("Error: No valid domain found.")
            return None

        return valid_domain
