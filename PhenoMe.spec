# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

block_cipher = None

# 获取项目根目录
project_root = Path(os.getcwd()).absolute()

# 分析主要的Python文件
# 检查SAM模型文件是否存在
sam_model_files = []
rulernet_model_files = []
text_detector_files = []
# SAM2 backend: weights + hydra config yamls + hydra _target_ submodule loading.
# TODO(verify in frozen app): hydra's compose()/instantiate() resolves classes by
# string, so sam2 needs every submodule and its configs/ data bundled.
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
sam2_hidden = collect_submodules('sam2')
sam2_datas = collect_data_files('sam2', includes=['configs/**/*.yaml'])
sam2_model_files = []
for _w in ('sam2.1_hiera_tiny.pt', 'sam2.1_hiera_large.pt'):
    if os.path.exists(f'models/sam2/{_w}'):
        sam2_model_files.append((f'models/sam2/{_w}', 'models/sam2'))
if os.path.exists('models/sam_vit_h_4b8939.pth'):
    sam_model_files.append(('models/sam_vit_h_4b8939.pth', 'models'))
if os.path.exists('models/rulernet.onnx'):
    rulernet_model_files.append(('models/rulernet.onnx', 'models'))
text_detector_dir = Path('models/grounding-dino-tiny')
if (text_detector_dir / 'config.json').exists():
    text_detector_names = [
        'config.json',
        'preprocessor_config.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'added_tokens.json',
        'tokenizer.json',
        'vocab.txt',
    ]
    if (text_detector_dir / 'model.safetensors').exists():
        text_detector_names.append('model.safetensors')
    elif (text_detector_dir / 'pytorch_model.bin').exists():
        text_detector_names.append('pytorch_model.bin')

    text_detector_files.extend([
        (str(text_detector_dir / name), 'models/grounding-dino-tiny')
        for name in text_detector_names
        if (text_detector_dir / name).exists()
    ])


a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=sam_model_files + rulernet_model_files,
    datas=[
        # 包含pheno包的所有文件
        ('pheno', 'pheno'),
        # 自定义表型算法随包分发
        ('algorithms', 'algorithms'),
    ] + text_detector_files + sam2_model_files + sam2_datas,
    hiddenimports=[
        # === PyTorch 深度学习框架 ===
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torch.nn.modules',
        'torch.utils',
        'torch.utils.data',
        'torch.utils.data.dataloader',
        'torch.distributed',
        'torch.functional',
        'torch._VF',
        'torchvision',
        
        # === 计算机视觉 ===
        'cv2',
        'segment_anything',
        'transformers',
        'transformers.models.grounding_dino',
        'transformers.models.grounding_dino.modeling_grounding_dino',
        'transformers.models.grounding_dino.processing_grounding_dino',
        'huggingface_hub',
        'safetensors',
        'tokenizers',
        'onnxruntime',
        'onnxruntime.capi._pybind_state',
        
        # === 基础数值计算 - NumPy ===
        'numpy',
        'numpy.testing',
        'numpy.linalg',
        'numpy.random',
        'numpy.fft',
        'numpy.ma',
        'numpy.core',
        
        # === 科学计算 - SciPy ===
        'scipy',
        'scipy.sparse',
        'scipy.sparse._base',
        'scipy.sparse._sputils',
        'scipy._lib',
        'scipy._lib._util',
        'scipy._lib._array_api',
        'scipy.spatial',
        'scipy.spatial.distance',
        'scipy.ndimage',
        'scipy.ndimage.filters',
        'scipy.optimize',
        'scipy.stats',
        'scipy.signal',
        'scipy.interpolate',
        'scipy.linalg',
        
        # === 机器学习 - scikit-learn ===
        'sklearn',
        'sklearn.cluster',
        'sklearn.cluster.KMeans',
        'sklearn.cluster._kmeans',
        'sklearn.base',
        'sklearn.utils',
        'sklearn.utils._chunking',
        'sklearn.utils._param_validation',
        'sklearn.preprocessing',
        'sklearn.decomposition',
        'sklearn.neighbors',
        'sklearn.metrics',
        
        # === 图像处理 - scikit-image ===
        'skimage',
        'skimage.measure',
        'skimage.morphology',
        'skimage.graph',  # 🔥 关键添加：图论算法
        'skimage.graph.route_through_array',
        'skimage.graph._graph',
        'skimage.segmentation',
        'skimage.filters',
        'skimage.feature',
        'skimage.transform',
        'skimage.util',
        'skimage.restoration',
        'skimage.color',
        'skimage.exposure',
        'skimage.io',
        'skimage.draw',
        'skimage._shared',
        'skimage._shared.utils',
        
        # === 几何计算 ===
        'shapely',
        'shapely.geometry',
        'shapely.ops',
        'shapely.affinity',
        
        # === 绘图和可视化 ===
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.patches',
        'matplotlib.colors',
        'matplotlib.backends',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_agg',
        
        # === 数据处理 ===
        'pandas',
        'pandas.core',
        'pandas.io',
        
        # === 配置和文件处理 ===
        'yaml',
        'json',
        'csv',
        'pickle',
        'requests',
        'urllib3',
        'certifi',
        'charset_normalizer',
        'idna',
        
        # === 图像格式支持 ===
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFilter',
        'PIL.ImageEnhance',
        
        # === PyQt5 界面框架 ===
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtWidgets',
        'PyQt5.QtGui',
        'PyQt5.sip',
        'sip',
        
        # === 必要的标准库模块 ===
        'unittest',
        'unittest.mock',
        'pdb',           # PyTorch distributed 需要
        'json',
        'csv',
        'datetime',
        'pathlib',
        'logging',
        'threading',
        'multiprocessing',
        'queue',
        'time',
        'os',
        'sys',
        'io',
        'contextlib',
        'functools',
        'itertools',
        'collections',
        'math',
        'statistics',
        'random',
        'copy',
        'warnings',
        'traceback',
        'glob',
        'shutil',
        'tempfile',
        're',
        'string',
        'platform',
        'subprocess',
        'hashlib',
        
        # === 应用程序模块 ===
        'pheno',
        'pheno.cuda',
        'pheno.SAM',
        'pheno.app',
        'pheno.splash',
        'pheno.canvas',
        'pheno.lib',
        'pheno.batch',
        'pheno.calcPheno',
        'pheno.shape',
        'pheno.toolBar',
        'pheno.widgets',
        'pheno.zoomWidget',
        'pheno.TextDetector',

        # === 自定义算法模块（随 algorithms/ 目录分发） ===
        'algorithms.beanSeedPheno',
        'algorithms.beanPodPheno',
    ] + sam2_hidden + [
        
        # === 其他工具库 ===
        'pkg_resources.py2_warn',
        'importlib',
        'importlib.util',
        'types',
        'inspect',
        'gc',
        'weakref',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的模块以减小体积
        'tkinter.test',
        'test',
        # 'unittest',  # 不能排除，numpy/scipy 需要它
        # 'pdb',      # 不能排除，PyTorch distributed 需要它
        'doctest',
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'pytest',
        'setuptools',
        'wheel',
        'pip',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 过滤掉一些不需要的文件
a.binaries = [x for x in a.binaries if not x[0].startswith(('api-ms-win-', 'ucrtbase'))]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # 使用动态链接库，不把所有内容打包到一个exe
    name='PhenoMe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 不显示控制台窗口
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # The app resolves bundled resources via CWD-relative paths (models/, pheno/
    # icons, algorithms/) with no _MEIPASS handling, so keep the pre-6.x layout
    # where everything sits next to PhenoMe.exe instead of under _internal/.
    contents_directory='.',
    icon='pheno/icons/icon.png',  # PhenoMe图标
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PhenoMe',
)
