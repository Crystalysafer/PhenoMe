import functools
import warnings
import webbrowser
import sys
import time
import os
import json
import zlib
from PIL import Image
import importlib
import cv2
import numpy as np
import datetime


# 尝试导入 mcc 模块，如果失败则使用备用方案
try:
    from cv2 import mcc
    HAS_MCC = True
except ImportError:
    HAS_MCC = False
    mcc = None

from qtpy import QT_VERSION
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtWidgets import QMessageBox, QFileDialog


QT5 = QT_VERSION[0] == '5'  # NOQA

from pheno import __appname__
from pheno import __version__
from pheno.canvas import Canvas
from pheno.shape import Shape
from pheno.config import get_config
from pheno.lib import addActions, fmtShortcut, newAction, newIcon, struct, qimage_to_cv2, Worker, WorkerSignals, write_image_file
from pheno.toolBar import ToolBar
from pheno.zoomWidget import ZoomWidget
from pheno.widgets import *
from pheno.calcPheno import PhenomicsCalculator
from pheno.TextDetector import TextDetector
from pheno import annotation
from pheno.reviewDialog import BatchReviewDialog
from pheno.gpuMonitor import GpuMonitor
from pheno.modelSelect import ModelSelectDialog
from pheno.i18n import get_language, set_language, tr
from pheno import batch



# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - [high] Add polygon movement with arrow keys
# - [high] Deselect shape when clicking and already selected(?)
# - [low,maybe] Open images with drag & drop.
# - [low,maybe] Preview images on file dialogs.
# - Zoom is too "steppy".


class WindowMixin(object):
    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu


    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName('%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        # Icon-only like labelme: captions live in the tooltip, and the whole bar
        # fits on one row even at modest window widths.
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        toolbar.setIconSize(QtCore.QSize(36, 36))
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar





class MainWindow(QtWidgets.QMainWindow, WindowMixin):
    
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2
    MAX_UNDO_STEPS = 20
    loadFileInMainThread = Signal(str)
    forceCanvasRepaint = Signal()
    loadFinished = Signal(str)

    def __init__(self, config=None, filename=None, output=None):
        # see pheno/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        super(MainWindow, self).__init__()
        self.settings = QtCore.QSettings('pheno', 'pheno')
        set_language(self.settings.value('ui/language', 'en'))
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False
        self.dont_show_again = False
        self.lastOpenDir = '.'
        self.previous = None

        self.batch_mode = False
        self.loadFileInMainThread.connect(self.loadFile)
        self.forceCanvasRepaint.connect(self.paintCanvas)

        # Traits table state. ``result_keys`` holds the Shape.uid each row came
        # from so the table can follow renames and deletions.
        self.results = []
        self.result_keys = []
        self.headers = []

        # Object-level undo history (see _snapshotShapes).
        self._undoStack = []

        self.custom_algorithms = {'DEFAULT': PhenomicsCalculator.defaultPhenomics}
        self.custom_headers = {'DEFAULT': PhenomicsCalculator.defaultHeaders()}
        self.imported_names = ['DEFAULT']
        self.loaded_names = ['DEFAULT']

        # File Dock
        self.fileListWidget = FileQListWidget()
        # self.fileListWidget.item
        self.fileListWidget.currentItemChanged.connect(self.fileSelectionAboutChanged)
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)

        self.batchBtn = QtWidgets.QToolButton()
        self.batchBtn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        batchBtnLayout = QtWidgets.QHBoxLayout()
        batchBtnLayout.addWidget(self.fileListWidget.selectAllCheckbox)
        batchBtnLayout.addWidget(self.batchBtn)

        # Recorded batch steps live next to the Batch button and stay hidden
        # until batch recording actually starts.
        self.batchStepsWidget = BatchStepsWidget()
        self.batchStepsWidget.removeRequested.connect(self.removeBatchSteps)
        self.batchStepsWidget.clearRequested.connect(self.clearBatchSteps)
        self.batchStepsWidget.loadRequested.connect(lambda: batch.loadRecipe(self))
        self.batchStepsWidget.saveRequested.connect(lambda: batch.saveRecipe(self))
        self.batchStepsWidget.exitRequested.connect(lambda: batch.cancelBatchSetup(self))
        self.batchStepsWidget.setVisible(False)

        filelistLayout = QtWidgets.QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addLayout(batchBtnLayout)
        filelistLayout.addWidget(self.batchStepsWidget)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QtWidgets.QWidget()
        fileListContainer.setLayout(filelistLayout)


        self.file_dock = QtWidgets.QDockWidget(tr('File List'), self)
        self.file_dock.setObjectName(u'Files')
        self.file_dock.setWidget(fileListContainer)
     

        # Shape Dock
        self.shapeList = ShapeQListWidget()

        self.shapeList.itemActivated.connect(self.labelSelectionChanged)
        self.shapeList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.shapeList.itemClicked.connect(self.labelSelectionChanged)
        self.shapeList.itemDoubleClicked.connect(self.editShapeLabel)
        self.shapeList.itemChanged.connect(self.labelChanged)
        # The highlight is intentionally kept when the list loses focus: it is
        # what the toolbar Delete button acts on.
        self.shapeList.deleteRequested.connect(self.deleteSelectedShape)
        self.shapeList.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.shapeList.customContextMenuRequested.connect(self._shapeContextMenu)
        

        self.masksSaveBtn = QtWidgets.QToolButton()
        self.masksSaveBtn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.calcBtn = QtWidgets.QToolButton()
        self.calcBtn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Text detection is driven from the toolbar action only; the Objects
        # panel keeps just the object list and its own buttons.
        self.lastTextPrompt = None

        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(self.shapeList.selectAllCheckbox)
        buttonLayout.addWidget(self.calcBtn)
        buttonLayout.addWidget(self.masksSaveBtn)

        listLayout = QtWidgets.QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)
        listLayout.addLayout(buttonLayout)
        listLayout.addWidget(self.shapeList)
        self.labelListContainer = QtWidgets.QWidget()
        self.labelListContainer.setLayout(listLayout)


        self.shape_dock = QtWidgets.QDockWidget(tr('Objects'), self)
        self.shape_dock.setObjectName('Objects')
        self.shape_dock.setWidget(self.labelListContainer)

        # Phenomics Table
        # 尚未打开图片，此时比例就是像素；打开或标定后各处会用当前单位重建表头。
        self.phenomicsTableModel = PhenomicTableModel(header=PhenomicsCalculator.getHeaders())
        self.table_view = QtWidgets.QTableView()
        self.table_view.setModel(self.phenomicsTableModel)


        self.manageAlgorithmBtn = QtWidgets.QToolButton()
        self.manageAlgorithmBtn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.exportBtn = QtWidgets.QToolButton()
        self.exportBtn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon) 
        self.setScaleBtn = QtWidgets.QToolButton()
        self.setScaleBtn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        # 添加显示当前比例尺的标签
        # Styled by restyleChrome(), so it follows the theme like everything else.
        self.scaleLabel = QtWidgets.QLabel("1 pixel = 1 pixel")
        self.restyleChrome()
        
        # self.phenoPatternCB = PhenomicComboBox() # phenomics calculation pattern
        phenoLayout = QtWidgets.QHBoxLayout()
        phenoLayout.setAlignment(Qt.AlignLeft)
        phenoLayout.addWidget(self.manageAlgorithmBtn)
        phenoLayout.addWidget(self.setScaleBtn)
        phenoLayout.addWidget(self.scaleLabel)
        
        phenoLayout.addStretch()
        phenoLayout.addWidget(self.exportBtn)
        # phenoLayout.addWidget(self.phenoPatternCB)

        tableLayout = QtWidgets.QVBoxLayout()
        tableLayout.addLayout(phenoLayout)
        tableLayout.addWidget(self.table_view)
        tableWidget = QtWidgets.QWidget()
        tableWidget.setLayout(tableLayout)

        self.table_view.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.table_view.setSortingEnabled(True)
        # Clicking a traits row highlights (and selects) the matching object.
        self.table_view.clicked.connect(self._onTraitsRowClicked)

        # 使用模型的智能列宽设置
        self.phenomicsTableModel.setupTableView(self.table_view)

        self.table_dock = QtWidgets.QDockWidget(tr("Traits Table"), self)
        self.table_dock.setObjectName('Table')
        self.table_dock.setWidget(tableWidget)


        # set DockWidgets
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.table_dock)

        # self.addDockWidget(Qt.BottomDockWidgetArea, self.file_dock)

        # VRAM is the one resource whose number changes a decision here -- whether the
        # next detection or batch will fit. The status bar's right side is built for a
        # permanent indicator, costs no layout space, and showMessage() writes to the
        # left without ever overwriting it.
        self.gpuMonitor = GpuMonitor(self)
        self.statusBar().addPermanentWidget(self.gpuMonitor)

        # Model/device switcher anchored at the status bar's left edge. The
        # real gate lives in openModelSelect(); the enabled state only mirrors
        # batch/switching so the button visibly greys out while those run.
        self._model_switching = False
        self.modelSelectBtn = QtWidgets.QToolButton(self)
        self.modelSelectBtn.setIcon(newIcon('modelSelect'))
        self.modelSelectBtn.setToolTip(tr("Segmentation model & compute device"))
        self.modelSelectBtn.setAutoRaise(True)
        self.modelSelectBtn.clicked.connect(self.openModelSelect)
        self.statusBar().addWidget(self.modelSelectBtn)

        self.zoomWidget = ZoomWidget()
        self.canvas = self.shapeList.canvas= Canvas(parent=self)
        # Named, and marked as drawing its own styled background, so the theme can colour
        # the well behind the picture. Qt paints a stylesheet background only when
        # WA_StyledBackground is set; without it the canvas fills from the palette and
        # the generic QWidget ground shows through, which is why the left dock used to
        # look like part of the picture area.
        self.canvas.setObjectName('CanvasView')
        self.canvas.setAttribute(Qt.WA_StyledBackground, True)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        
        # 初始化比例尺属性
        self.canvas.scale_per_pixel = 1.0
        self.canvas.scale_unit = "pixel"

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)

        # 创建坐标显示控件
        self.coordLabel = QtWidgets.QLabel()
        self.coordLabel.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 150);
                border: none;
                padding: 6px 8px;
                font-size: 12px;
                font-weight: bold;
                font-family: monospace;
            }
        """)
        self.coordLabel.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.coordLabel.setFixedSize(180, 60)  # 缩小框的尺寸
        
        # 将坐标标签设置为scrollArea的子控件，并固定在右下角
        self.coordLabel.setParent(scrollArea)
        self.coordLabel.raise_()  # 确保在最上层

        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.canvas.newShape.connect(self.newShape)
        self.canvas.maskPolygon.connect(self._onRegionDrawn)
        self.canvas.mousePositionChanged.connect(self.updateMousePosition)
        self.canvas.shapePicked.connect(self._onCanvasShapePicked)
        self.canvas.brushStrokeStarted.connect(self._onBrushStrokeStarted)
        self.canvas.brushStrokeFinished.connect(self._onBrushStrokeFinished)

        self.setCentralWidget(scrollArea)

        # 画布右上角的小浮条：审查会话的返回按钮 + 画笔模式控件，
        # 和 coordLabel 一样叠在 scrollArea 上随窗口尺寸重排。
        self._buildEditHud(scrollArea)
        self.canvas.brushRadiusChanged.connect(self._onBrushRadiusChanged)
        self.canvas.measureStageChanged.connect(self._onMeasureStageChanged)
        # Esc 退出画笔模式 —— 比回到右键菜单再点一次顺手。
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Escape), self,
                            activated=self._exitBrushMode)

        # 连接滚动区域的resize事件以更新坐标标签位置
        scrollArea.resizeEvent = self.scrollAreaResizeEvent

        # Actions
        action = functools.partial(newAction, self)
        shortcuts = self._config['shortcuts']
        quit = action(tr('&Quit'), self.close, shortcuts['quit'], 'quit',
                      tr('Quit application'))
        open_ = action(tr('&Open'), self.openFile, shortcuts['open'], 'open',
                       tr('Open image or label file'))
        opendir = action(tr('&Open Dir'), self.openDirDialog,
                         shortcuts['open_dir'], 'open', tr('Open Dir'))
        openNextImg = action(tr('&Next Image'), self.openNextImg,
                             shortcuts['open_next'], 'next', tr('Open Next'))

        openPrevImg = action(tr('&Prev Image'), self.openPrevImg,
                             shortcuts['open_prev'], 'prev', tr('Open Prev'))
        demo = action('Demo', self.demo, None, 'demo', 'Demo', enabled=True)

        click = action(tr('&Click'), self.click, shortcuts['click'], 'click',
                        tr('Click'), enabled=False)
        cutOut = action(tr('&Extract Object'), self.cutOut, shortcuts['cut_out'], 'cut_out',
                        tr('Extract the selected object'), enabled=False)
        
        delete = action(tr('&Delete'), self.deleteSelectedShape, shortcuts['delete'], 'delete',
                        tr('Delete the Object'), enabled=False)

        undo = action(tr('&Undo'), self.undoLastChange, shortcuts['undo'], 'redo',
                      tr('Undo the last object change'), enabled=False)

        editMask = action(tr('&Edit Mask'), self.toggleMaskBrush,
                          shortcuts.get('edit_mask', 'B'), 'edit_mask',
                          tr('Brush: left button adds, right button removes, '
                             'wheel resizes, Ctrl+wheel zooms'),
                          enabled=False, checkable=True)
        
        auto = action(tr('&Region Detect'), self.auto, shortcuts['auto'], 'area_detect',
                        tr('Region Detect'), enabled=False)

        textDetect = action(tr('&Semantic Detect'), self.textPromptDetect,
                            shortcuts.get('text_detect', 'T'), 'text_detect',
                            tr('Detect objects by describing them'), enabled=False)

        # Green-leaf segmentation had its own toolbar button between Semantic Detect and
        # Region Detect, and read as a third kind of detection sitting next to two others
        # -- visually incongruous, and it is not a peer of them: it is a colour threshold
        # that works on one specific kind of photograph. The method (greenLeafDetect), its
        # status messages and the `green` batch step type all remain, so recipes that
        # contain one still replay; only the UI entry is gone.
        
        export = action('&Export', self.export, shortcuts['export'], 'export',
                        'Export Spreadsheet', enabled=True)

        # Ctrl+S writes the annotation, because that is the only thing here that can be
        # reopened. It is also what the unsaved-changes prompt already calls when its
        # Save button is pressed, so the shortcut and the dialog now agree.
        saveAnnotation = action(
            tr('&Save Annotation'), self.saveAnnotation,
            shortcuts.get('save', 'Ctrl+S'), 'save',
            tr('Save objects of this image'), enabled=False)

        deleteAnnotation = action(
            tr('&Delete Annotation'), self.deleteAnnotation,
            None, 'deleteAnn',
            tr('Delete the saved annotation of this image'), enabled=True)

        
        manageAlgorithm = action('&Manage Algorithms', self.manageAlgorithm, shortcuts['manageAlgorithm'], 'manageAlgorithm',
                              'Manage Custom Algorithms', enabled=True)

        resetView = action('&Reset View', self.resetView, 'None', 'reset_View',
                        'reset View', enabled=True)
        
        colorChecker = action('&Color Calibration', self.colorChecker, shortcuts['color_checker'], 'color_checker',
                              'Color Checker Detection And Calibration', enabled=False, checkable=True)
        
        rulerDetection = action('&Ruler Detection', self.rulerDetection, shortcuts['ruler_detection'], 'ruler_detection',
                        'Ruler Detection', enabled=False, checkable=True)

        setScale = action('&Set Scale', self.setScale, shortcuts['set_scale'], 'set_scale',
                        'Set Real World Scale', enabled=True)

        # Batch review lives in its own window so the main UI stays uncluttered.
        reviewBatch = action(tr('&Review Batch Results'), self.reviewBatchResults,
                             shortcuts.get('review_batch', 'Ctrl+R'), 'blue_check',
                             tr('Review and summarise a batch output folder'),
                             enabled=True)


        self.exportBtn.setDefaultAction(export)
        self.manageAlgorithmBtn.setDefaultAction(manageAlgorithm)
        self.setScaleBtn.setDefaultAction(setScale)
        
        # A mask PNG is an export, not a save: nothing reads it back, the sidecar carries
        # the same masks as RLE, and it needs objects to be ticked first. It used to be
        # labelled "Save" and hold Ctrl+S, which is how a user could press Ctrl+S, cancel
        # the directory picker, and carry on believing their work was stored.
        exportMasks = action(
            tr('&Export Masks'), self.saveSelectedMasks,
            shortcuts.get('export_masks', 'Ctrl+Shift+M'), 'export_mask',
            tr("Export the selected objects' masks as PNG files"), enabled=True)
        calcSelectedMasks = action('&Calculate', self.calcSelectedMasks, 'None', 'calc', 'Calculate Selected Objects', enabled=True)
        batchProcessing = action('&Batch', lambda: batch.batchPre(self), 'None', 'batch', 'Batch Processing', checkable=True, enabled=True)
        # Fills the step list from a file instead of from the mouse: the same recipe
        # replays identically for anyone who has the file.
        # Batch recipes load/save through the Batch Steps panel buttons, not the
        # menu: the panel is where the step list lives, so the controls sit next
        # to the thing they fill.


        self.masksSaveBtn.setDefaultAction(exportMasks)
        self.calcBtn.setDefaultAction(calcSelectedMasks)
        self.batchBtn.setDefaultAction(batchProcessing)


        help = action('&Homepage', self.openHomepage, icon='github',
                      tip='Open the project homepage')
        userGuide = action('&User Guide', self.showUserGuide, icon='help',
                           tip='Show the built-in user guide')

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            "Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." %
            (fmtShortcut('%s,%s' % (shortcuts['zoom_in'],
                                    shortcuts['zoom_out'])),
             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action('Zoom &In', functools.partial(self.addZoom, 10),
                        shortcuts['zoom_in'], 'zoom-in',
                        'Increase zoom level', enabled=False)
        zoomOut = action('&Zoom Out', functools.partial(self.addZoom, -10),
                         shortcuts['zoom_out'], 'zoom-out',
                         'Decrease zoom level', enabled=False)
        zoomOrg = action('&Original size',
                         functools.partial(self.setZoom, 100),
                         shortcuts['zoom_to_original'], 'zoom',
                         'Zoom to original size', enabled=False)
        fitWindow = action('&Fit Window', self.setFitWindow,
                           shortcuts['fit_window'], 'fit-window',
                           'Zoom follows window size', checkable=True,
                           enabled=False)
        fitWidth = action('Fit &Width', self.setFitWidth,
                          shortcuts['fit_width'], 'fit-width',
                          'Zoom follows window width',
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut, zoomOrg,
                       fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        # Store actions for further handling.
        self.actions = struct(
            open=open_, export=export, 
            zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
            fitWindow=fitWindow, fitWidth=fitWidth,
            zoomActions=zoomActions, resetView=resetView,
            click=click, cutOut=cutOut, delete=delete,
            auto=auto, textDetect=textDetect,
            undo=undo, editMask=editMask,
            saveAnnotation=saveAnnotation, deleteAnnotation=deleteAnnotation,
            reviewBatch=reviewBatch,
            fileMenuActions=(open_, opendir, quit),
            tool=(), # see below self.actions.tool =
            editMenu=(undo, None, click, cutOut, delete, editMask, None, textDetect, auto, None, colorChecker, rulerDetection),
            exportMasks=exportMasks, calcSelectedMasks=calcSelectedMasks,
            colorChecker=colorChecker, rulerDetection=rulerDetection, setScale=setScale,
            batchProcessing=batchProcessing,
        )

        englishLanguage = action(
            'English', lambda _checked=False: self.setLanguage('en'), checkable=True
        )
        chineseLanguage = action(
            '中文', lambda _checked=False: self.setLanguage('zh_CN'), checkable=True
        )
        languageGroup = QtWidgets.QActionGroup(self)
        languageGroup.setExclusive(True)
        languageGroup.addAction(englishLanguage)
        languageGroup.addAction(chineseLanguage)

        # Appearance, stored like the language and repainted on change. The palette is
        # shared with the web front end -- see pheno/theme.py.
        classicTheme = action(tr('&Classic'), lambda _checked=False: self.setTheme('classic'), checkable=True)
        lightTheme = action(tr('&Light'), lambda _checked=False: self.setTheme('light'), checkable=True)
        darkTheme = action(tr('&Dark'), lambda _checked=False: self.setTheme('dark'), checkable=True)
        themeGroup = QtWidgets.QActionGroup(self)
        themeGroup.setExclusive(True)
        themeGroup.addAction(classicTheme)
        themeGroup.addAction(lightTheme)
        themeGroup.addAction(darkTheme)

        self.menus = struct(
            file=self.menu(tr('&File')),
            edit=self.menu(tr('&Edit')),
            view=self.menu(tr('&View')),
            language=self.menu(tr('&Language')),
            help=self.menu(tr('&Help')),
            recentFiles=QtWidgets.QMenu(tr('Open &Recent')),
        )

        addActions(self.menus.file, (open_, opendir, self.menus.recentFiles,
                                     None, reviewBatch, None,
                                     saveAnnotation, deleteAnnotation, None, quit))
        addActions(self.menus.help, (userGuide, help))
        addActions(self.menus.language, (englishLanguage, chineseLanguage))
        addActions(self.menus.view, (
            resetView, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth, None,
            classicTheme, lightTheme, darkTheme))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        self.tools = self.toolbar(tr('Tools'))
        self.actions.tool = (
            open_, opendir, openNextImg, openPrevImg,
            None, click, cutOut, None,
            textDetect, auto, None,
            colorChecker, rulerDetection, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self._translated_actions = {
            quit: ('&Quit', 'Quit application'),
            open_: ('&Open', 'Open image or label file'),
            opendir: ('&Open Dir', 'Open Dir'),
            openNextImg: ('&Next Image', 'Open Next'),
            openPrevImg: ('&Prev Image', 'Open Prev'),
            click: ('&Click', 'Click'),
            cutOut: ('&Extract Object', 'Extract the selected object'),
            delete: ('&Delete', 'Delete the Object'),
            undo: ('&Undo', 'Undo the last object change'),
            editMask: ('&Edit Mask', 'Brush-edit the selected object\'s mask (one object at a time)'),
            auto: ('&Region Detect', 'Region Detect'),
            textDetect: ('&Semantic Detect', 'Detect objects by describing them'),
            export: ('&Export', 'Export Spreadsheet'),
            saveAnnotation: ('&Save Annotation', 'Save objects of this image'),
            deleteAnnotation: ('&Delete Annotation', 'Delete the saved annotation of this image'),
            reviewBatch: ('&Review Batch Results', 'Review and summarise a batch output folder'),
            manageAlgorithm: ('&Manage Algorithms', 'Manage Custom Algorithms'),
            resetView: ('&Reset View', 'reset View'),
            colorChecker: ('&Color Calibration', 'Color Checker Detection And Calibration'),
            rulerDetection: ('&Ruler Detection', 'Ruler Detection'),
            setScale: ('&Set Scale', 'Set Real World Scale'),
            exportMasks: ('&Export Masks', "Export the selected objects' masks as PNG files"),
            calcSelectedMasks: ('&Calculate', 'Calculate Selected Objects'),
            batchProcessing: ('&Batch', 'Batch Processing'),
            help: ('&Homepage', 'Open the project homepage'),
            userGuide: ('&User Guide', 'Show the built-in user guide'),
            zoomIn: ('Zoom &In', 'Increase zoom level'),
            zoomOut: ('&Zoom Out', 'Decrease zoom level'),
            zoomOrg: ('&Original size', 'Zoom to original size'),
            fitWindow: ('&Fit Window', 'Zoom follows window size'),
            fitWidth: ('Fit &Width', 'Zoom follows window width'),
        }
        self.actions.languageEnglish = englishLanguage
        self.actions.languageChinese = chineseLanguage
        self.actions.themeClassic = classicTheme
        self.actions.themeLight = lightTheme
        self.actions.themeDark = darkTheme
        from pheno.theme import mode as _theme_mode
        for _name, _wanted in (('themeClassic', 'classic'), ('themeLight', 'light'),
                               ('themeDark', 'dark')):
            getattr(self.actions, _name).setChecked(_theme_mode() == _wanted)
        self.setLanguage(get_language(), persist=False)

        self.status('%s started.' % __appname__)

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.output = output
        self.recentFiles = []
        self.maxRecent = 7
        self.zoom_level = 100
        self.fit_window = False

        if filename is not None and os.path.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        # XXX: Could be completely declarative.
        # Restore application settings.
        # self.settings.clear()
        # FIXME: QSettings.value can return None on PyQt4
        self.recentFiles = self.settings.value('recentFiles', []) or []
        size = self.settings.value('window/size', QtCore.QSize(600, 500))
        position = self.settings.value('window/position', QtCore.QPoint(0, 0))
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        # Snapshot the as-built dock/toolbar arrangement first: resetView() puts
        # it back without a restart.
        self._default_state = self.saveState(1)
        # v1 state format: layouts saved before this version are rejected.
        self.restoreState(
            self.settings.value('window/state', QtCore.QByteArray()), 1)


        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        self.threadpool = QtCore.QThreadPool()  # Create a threadpool for Phenomics calculation
        self.sam_threadpool = QtCore.QThreadPool(self)
        self.sam_threadpool.setMaxThreadCount(1)
        self._sam_ready = False
        self._sam_request_id = 0
        self.text_detector = None
        
        # 校正相关状态
        self.color_correction_data = None  # 存储颜色校正结果
        self.ruler_detection_data = None # 存储尺度检测结果
        self.original_image = None  # 存储原始图像
        # 审查编辑会话：当前文件是批处理校色成品图，按钮态/写回标记靠它维持。
        self._loadedColorCorrected = False
        self._colorRevertImage = None   # 可退回的原图（QImage），缺失则不可撤销
        self._colorOriginalPath = None  # sidecar 应登记的原图全路径
        self._loadedRulerDetection = False
        self._reviewReturnDialog = None
        self._reviewPriorFilename = None



    def populateModeActions(self):
        self.tools.clear()
        addActions(self.tools, self.actions.tool)
        self.menus.edit.clear()
        addActions(self.menus.edit, self.actions.editMenu)

    def setTheme(self, mode, persist=True):
        """Switch the look, and remember it.

        The stylesheet is re-applied to the whole application rather than to the
        window, because dialogs and docked panels are separate top-level widgets and
        styling only the window would leave them in the old palette.
        """
        from pheno.theme import apply_theme

        app = QtWidgets.QApplication.instance()
        if app is not None:
            apply_theme(app, mode)
        # The stylesheet reaches every widget, but a few carry a colour of their own;
        # they redraw themselves here.
        self.restyleChrome()
        if persist:
            self.settings.setValue('ui/theme', mode)
        for name, wanted in (('themeClassic', 'classic'), ('themeLight', 'light'),
                             ('themeDark', 'dark')):
            item = getattr(self.actions, name, None)
            if item is not None:
                item.setChecked(mode == wanted)

    def restyleChrome(self):
        """Re-apply the handful of colours the stylesheet cannot reach.

        Qt has no CSS variables, so anything painted from a per-widget stylesheet has to
        be rebuilt when the mode changes. Those places are listed here rather than
        scattered, so adding one is a single edit.
        """
        from pheno import iconTint
        from pheno.theme import palette, themed

        if themed():
            c = palette()
            self.scaleLabel.setStyleSheet(
                f"QLabel {{ color: {c['text']};"
                f" background-color: {c['soft']};"
                f" border: 1px solid {c['line']};"
                " border-radius: 6px;"
                " padding: 4px 9px; margin-left: 12px; }"
            )
            # Icons are pixmaps, so a stylesheet cannot recolour them; they are repainted in
            # the new ink here.
            iconTint.refresh(self, c["icon"])
        else:
            # Classic: native chrome, the label unadorned, the original coloured icons.
            self.scaleLabel.setStyleSheet("")
            iconTint.restore(self)
        monitor = getattr(self, 'gpuMonitor', None)
        if monitor is not None:
            monitor.applyTheme()

    def setLanguage(self, language, persist=True):
        set_language(language)
        if persist:
            self.settings.setValue('ui/language', get_language())

        if not hasattr(self, 'actions'):
            return

        self.actions.languageEnglish.setChecked(get_language() == 'en')
        self.actions.languageChinese.setChecked(get_language() == 'zh_CN')
        self.retranslateUi()

    def retranslateUi(self):
        self.file_dock.setWindowTitle(tr('File List'))
        self.shape_dock.setWindowTitle(tr('Objects'))
        self.table_dock.setWindowTitle(tr('Traits Table'))
        self.fileListWidget.selectAllCheckbox.setText(tr('Select All'))
        self.shapeList.selectAllCheckbox.setText(tr('Select All'))

        self.menus.file.setTitle(tr('&File'))
        self.menus.edit.setTitle(tr('&Edit'))
        self.menus.view.setTitle(tr('&View'))
        self.menus.language.setTitle(tr('&Language'))
        self.menus.help.setTitle(tr('&Help'))
        self.menus.recentFiles.setTitle(tr('Open &Recent'))
        self.tools.setWindowTitle(tr('Tools'))

        for action, (text, tip) in self._translated_actions.items():
            action.setText(tr(text))
            action.setToolTip(tr(tip))
            action.setStatusTip(tr(tip))

        self.batchStepsWidget.retranslateUi(self.batchSteps())

        self.updateMousePosition(-1, -1)
        self.updateScaleLabel()

    # region Batch step recording
    def batchSteps(self):
        """Ordered list of operations recorded for batch replay."""
        return self.canvas.BATCH_PARAM.setdefault("steps", [])

    def recordBatchStep(self, step):
        """Append one recorded operation and refresh the panel."""
        self.batchSteps().append(step)
        self.refreshBatchSteps()

    def _toggleBatchFlagStep(self, kind, enabled):
        """Add or drop a checkable step (color calibration / ruler detection)."""
        steps = self.batchSteps()
        existing = [
            index for index, step in enumerate(steps) if step.get("type") == kind
        ]
        if enabled:
            if not existing:
                self.recordBatchStep({"type": kind})
                self.status(
                    tr(
                        "Recorded batch step: {step}",
                        step=describe_batch_step({"type": kind}),
                    )
                )
            return

        for index in reversed(existing):
            steps.pop(index)
        self.refreshBatchSteps()
        if existing:
            self.status(tr("Removed {count} batch step(s).", count=len(existing)))

    def refreshBatchSteps(self):
        self.batchStepsWidget.refresh(self.batchSteps())

    def showBatchStepsPanel(self, visible):
        self.batchStepsWidget.setVisible(visible)
        if visible:
            self.refreshBatchSteps()

    def removeBatchSteps(self, indices):
        steps = self.batchSteps()
        for index in sorted(indices, reverse=True):
            if 0 <= index < len(steps):
                steps.pop(index)
        self.refreshBatchSteps()
        self._syncBatchFlagActions()
        self.status(tr("Removed {count} batch step(s).", count=len(indices)))

    def clearBatchSteps(self):
        steps = self.batchSteps()
        if not steps:
            return
        count = len(steps)
        del steps[:]
        self.refreshBatchSteps()
        self._syncBatchFlagActions()
        self.status(tr("Removed {count} batch step(s).", count=count))

    def _syncBatchFlagActions(self):
        """Keep the checkable toolbar buttons in sync with recorded steps."""
        kinds = {step.get("type") for step in self.batchSteps()}
        self.actions.colorChecker.setChecked("color_correction" in kinds)
        self.actions.rulerDetection.setChecked("ruler_detection" in kinds)
    # endregion

    def traitHeaders(self):
        """当前比例单位下的性状表头。

        单位不是固定文本：未标定时是像素，标定后是 mm/cm，所以表头必须跟着
        比例一起变——否则一列裸数字无法判断量纲，这正是导出表被诟病的地方。
        """
        canvas = getattr(self, 'canvas', None)
        unit = getattr(canvas, 'scale_unit', 'pixel') if canvas is not None else 'pixel'
        return PhenomicsCalculator.getHeaders(unit)

    def refreshTraitHeaders(self):
        """比例改变后重建表头，并清掉已算出的数值。

        只换表头是不够的：已有数字是在旧比例下算的，配上新单位的表头会得到
        一张看起来合理、实际错误的表——那比没有单位更危险。所以宁可清空，
        让用户重新点一次计算。
        """
        self.results = []
        self.result_keys = []
        self.phenomicsTableModel.updateData([], self.traitHeaders())
        self.updateTraitsTableTitle()

    def updateScaleLabel(self):
        scale = self.canvas.scale_per_pixel
        unit = self.canvas.scale_unit
        if unit == 'pixel' and scale == 1.0:
            self.scaleLabel.setText('1 pixel = 1 pixel')
        else:
            self.scaleLabel.setText(f'1 pixel = {scale:.4f} {unit.upper()}')

    def setDirty(self):        self.dirty = True

    def setClean(self):
        self.dirty = False

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        sam_enabled = value and self._sam_ready
        # Every detection entry point stays available in batch mode, where it
        # records a step instead of processing the current image.
        self.actions.click.setEnabled(sam_enabled)
        self.actions.auto.setEnabled(sam_enabled)
        self.actions.textDetect.setEnabled(sam_enabled)
        self.actions.colorChecker.setEnabled(value)
        self.actions.rulerDetection.setEnabled(value)
        # Saving only needs an image, not a ready SAM embedding.
        self.actions.saveAnnotation.setEnabled(value and not self.batch_mode)


        for z in self.actions.zoomActions:
            z.setEnabled(value)

        if hasattr(self, 'modelSelectBtn'):
            self.modelSelectBtn.setEnabled(
                not self.batch_mode
                and not getattr(self, '_model_switching', False))

    def _prepareSamImage(self, image, request_id):
        self.canvas.sam.set_image(qimage_to_cv2(image))
        return request_id

    def _startSamImageLoad(self, image, filename):
        self._sam_request_id += 1
        request_id = self._sam_request_id
        self._sam_ready = False
        self.toggleActions(True)

        worker = Worker(
            self._prepareSamImage,
            "Prepare SAM Image",
            image,
            request_id,
        )
        worker.signals.result.connect(
            lambda result, path=filename: self._finishSamImageLoad(result, path)
        )
        worker.signals.error.connect(
            lambda message, rid=request_id, path=filename: self._failSamImageLoad(
                rid, path, message
            )
        )
        self.sam_threadpool.start(worker)

    def _finishSamImageLoad(self, request_id, filename):
        if request_id != self._sam_request_id:
            return

        self._sam_ready = True
        self.toggleActions(True)
        self.status("Loaded %s" % os.path.basename(filename))
        self.loadFinished.emit(filename)

    def _failSamImageLoad(self, request_id, filename, message):
        if request_id != self._sam_request_id:
            return

        self._sam_ready = False
        self.toggleActions(True)
        self.status("Segmentation preparation failed: %s" % message)
        self.loadFinished.emit(filename)

    # region Runtime model/device switcher

    def openModelSelect(self):
        """Status-bar button: switch segmenter + device without restarting.

        Only meaningful in NORMAL mode: during a batch the recorded steps
        belong to the segmenter they were replayed with, and mid-draw the
        canvas holds prompt state a rebuilt backend would silently lose.
        """
        if (self.batch_mode or self._model_switching
                or (self.canvas and self.canvas.mode != Canvas.NORMAL)):
            self.status(tr(
                "Model and device can only be changed outside "
                "batch and drawing modes."))
            return

        settings = QtCore.QSettings('pheno', 'pheno')
        dialog = ModelSelectDialog(
            self,
            model_key=self.canvas.sam.model_key,
            device_pref=str(settings.value('model/device', 'auto')),
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        if (dialog.model_key == self.canvas.sam.model_key
                and dialog.device_pref
                == str(settings.value('model/device', 'auto'))):
            return  # nothing changed -- keep the warm backend
        self._applyModelSelect(dialog.model_key, dialog.device_pref)

    def _applyModelSelect(self, model_key, device_pref):
        """Persist the choice, then rebuild SAM in the serial sam thread."""
        from pheno.SAM import SAM, SEGMENTER_SETTING
        settings = QtCore.QSettings('pheno', 'pheno')
        settings.setValue(SEGMENTER_SETTING, model_key)
        settings.setValue('model/device', device_pref)
        if device_pref == 'cpu':
            os.environ['PHENOME_FORCE_CPU'] = '1'
        else:
            os.environ.pop('PHENOME_FORCE_CPU', None)

        self._pre_switch_ready = self._sam_ready
        self._sam_ready = False
        self._model_switching = True
        self.toggleActions(True)
        waiting_dialog = WaitingDialog(
            self, tr("Switching segmentation model..."))
        waiting_dialog.show()

        worker = Worker(functools.partial(SAM, model_key=model_key),
                        "Switch Segmenter")
        worker.signals.result.connect(
            lambda sam: self._finishModelSwitch(sam, waiting_dialog))
        worker.signals.error.connect(
            lambda message: self._failModelSwitch(message, waiting_dialog))
        self.sam_threadpool.start(worker)

    def _finishModelSwitch(self, new_sam, waiting_dialog):
        waiting_dialog.close_dialog()
        old_sam = self.canvas.sam
        self.canvas.sam = new_sam
        try:
            # Confirmed working, so now the old backend's GPU pool can go.
            del old_sam
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # The cached detector was built for the old device; let it rebuild.
        self.text_detector = None
        self._model_switching = False

        from pheno.SAM import segmenter_label
        self.status(tr("Segmenter: {label} on {device}",
                       label=segmenter_label(new_sam.model_key),
                       device=new_sam.device))

        # Re-embed the open image on the new backend through the same
        # worker/request-id path a normal image load uses.
        if not self.image.isNull():
            self._startSamImageLoad(self.image, self.filename or "")
        else:
            self._sam_ready = True
            self.toggleActions(True)

    def _failModelSwitch(self, message, waiting_dialog):
        waiting_dialog.close_dialog()
        self._model_switching = False
        self._sam_ready = getattr(self, '_pre_switch_ready', False)
        self.toggleActions(True)
        self.status(tr("Model switch failed: {message}", message=message))

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        # One funnel for status text, so a raw literal written anywhere still reaches
        # the translation table instead of showing English in a Chinese window. A
        # caller that already translated passes its result through unchanged, because
        # the table is keyed by the English source.
        self.statusBar().showMessage(tr(message), delay)

    def resetState(self):
        self._sam_request_id = getattr(self, '_sam_request_id', 0) + 1
        self._sam_ready = False
        self.shapeList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None

        # Undo history and traits rows belong to one image only.
        self.clearUndoStack()
        self.results = []
        self.result_keys = []
        self.phenomicsTableModel.updateData([], self.traitHeaders())
        self.updateTraitsTableTitle()

        # 重置颜色校正状态
        self.color_correction_data = None
        self.original_image = None
        self._loadedColorCorrected = False
        self._colorRevertImage = None
        self._colorOriginalPath = None
        self._loadedRulerDetection = False

        self.actions.colorChecker.setChecked(False)
        self.actions.rulerDetection.setChecked(False)

        self.setMode(Canvas.NORMAL)
        self.actions.click.setEnabled(False)
        self.actions.auto.setEnabled(False)
        self.canvas.setEnabled(False)
        self.canvas.resetState()


    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks
    def setMode(self, mode=Canvas.NORMAL):
        self.canvas.setMode(mode)
        if mode == Canvas.DRAW:
            self.actions.cutOut.setEnabled(True)
            self.actions.click.setEnabled(False)
            self.actions.auto.setEnabled(False)
            self.actions.textDetect.setEnabled(False)
            self.actions.colorChecker.setEnabled(False)
            self.actions.rulerDetection.setEnabled(False)
            self.actions.editMask.setEnabled(False)

        elif mode == Canvas.MASK:
            self.actions.cutOut.setEnabled(True)
            self.actions.click.setEnabled(False)
            # Region Detect stays enabled so the same toolbar entry toggles
            # the mode off -- no separate Done control needed.
            self.actions.auto.setEnabled(True)
            self.actions.textDetect.setEnabled(False)
            self.actions.colorChecker.setEnabled(False)
            self.actions.rulerDetection.setEnabled(False)
            self.actions.editMask.setEnabled(False)

        elif mode == Canvas.BRUSH:
            # While brushing, every other segmentation entry point is disabled so
            # a stray click cannot start a different operation mid-edit.
            self.actions.cutOut.setEnabled(False)
            self.actions.click.setEnabled(False)
            self.actions.auto.setEnabled(False)
            self.actions.textDetect.setEnabled(False)
            self.actions.colorChecker.setEnabled(False)
            self.actions.rulerDetection.setEnabled(False)
            self.actions.editMask.setEnabled(True)

        else:
            self.actions.cutOut.setEnabled(False)
            self.actions.click.setEnabled(self._sam_ready)
            self.actions.auto.setEnabled(self._sam_ready)
            self.actions.textDetect.setEnabled(self._sam_ready)
            self.actions.colorChecker.setEnabled(True)
            self.actions.rulerDetection.setEnabled(True)
            self.actions.editMask.setChecked(False)
            _sel = self.shapeList.get_selected_shapes()
            self.actions.editMask.setEnabled(
                len(_sel) == 1 and _sel[0].mask is not None)

        self._updateEditHud()



    def demo(self):
        QMessageBox.information(self, 'Demo', 'This is a demo.')

    def resetView(self):
        reply = QMessageBox.warning(
        self, tr("Reset View"),
        tr("Reset the window layout to its default state?"),
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No  # 默认选项
    )
        if reply == QMessageBox.Yes:
            # Put the as-built layout back live, and drop the saved window keys
            # so a restart does not resurrect what the user just reset. Other
            # settings (theme, language, recent files) are not "view" and stay.
            self.restoreState(self._default_state, 1)
            for key in ('window/size', 'window/position', 'window/state'):
                self.settings.remove(key)
        else:
            return

    def click(self):
        self.labelDeSelect()
        self.setMode(Canvas.DRAW)
        # QMessageBox.information(self, 'CLICK', 'This is a click.')

    def cutOut(self):
        if self.batch_mode:
            # Record the prompt points instead of committing an object.
            current = self.canvas.current
            points = []
            if current is not None:
                points = [
                    {
                        "x": int(point.x()),
                        "y": int(point.y()),
                        "label": int(label),
                    }
                    for point, label in zip(
                        current.prompt_points, current.prompt_labels
                    )
                ]
            self.setMode(Canvas.NORMAL)
            if points:
                self.recordBatchStep({"type": "click", "points": points})
                self.status(
                    tr(
                        "Recorded batch step: {step}",
                        step=describe_batch_step(
                            {"type": "click", "points": points}
                        ),
                    )
                )
            return

        self.setMode(Canvas.NORMAL)

    def deleteSelectedShape(self):
        """Delete every highlighted object (Ctrl/Shift multi-selection)."""
        shapes = self.shapeList.get_selected_shapes()
        if not shapes:
            shapes = [self.canvas.selectedShape] if self.canvas.selectedShape else []
        if not shapes:
            return

        if len(shapes) == 1:
            msg = tr(
                'You are about to permanently delete "{name}", proceed anyway?',
                name=shapes[0].label,
            )
        else:
            msg = tr(
                'You are about to permanently delete {count} objects, proceed anyway?',
                count=len(shapes),
            )

        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        self.shapeList.ignore_focus_event = True
        if yes == QtWidgets.QMessageBox.warning(self, tr('Attention'), msg,
                                                yes | no):
            self.pushUndoSnapshot(tr('Delete objects'))
            removed = self.canvas.deleteShapes(shapes)
            for shape in removed:
                self.shapeList.remove_shape(shape)
            # Their traits rows must go too, otherwise the table keeps showing
            # numbers for objects that no longer exist.
            self.dropTraitRows(removed)
            self.canvas.update()
            self.setDirty()
            self.status(tr('Deleted {count} object(s).', count=len(removed)))
            self.actions.delete.setEnabled(bool(self.shapeList.get_selected_shapes()))
        self.shapeList.ignore_focus_event = False


    def auto(self):
        if self.canvas.isMasking():
            self._exitMaskMode()
            return
        self.setMode(Canvas.MASK)
        self._updateEditHud()
        self.status(tr(
            'Region Detect: drag a region on the image; '
            'right-click clears it; press Region Detect again to exit.'))
        # QMessageBox.information(self, 'AUTO', 'This is a auto.')

    def openHomepage(self):
        """打开项目主页。

        使用说明已经内置在 Help → User Guide（pheno/help_content.py，与网页端
        /api/help 同一份来源），所以这里不再单独提供教程入口，只留主页。
        """
        url = 'https://github.com/Crystalysafer/PhenoMe'  # NOQA
        webbrowser.open(url)

    def showUserGuide(self):
        """内置使用说明。

        内容来自 pheno/help_content.py——与网页端 /api/help 同一份来源，
        避免两端对同一个按钮给出不同说明。语言跟随 Language 菜单。
        """
        from pheno.help_content import DESKTOP, help_html, help_icon_names
        from pheno.i18n import get_language

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(tr('User Guide'))
        dialog.resize(720, 640)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)

        language = get_language()
        html = help_html(language, DESKTOP)

        browser = QtWidgets.QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        for name in help_icon_names(language, DESKTOP):
            icon = newIcon(name)
            if not icon.isNull():
                browser.document().addResource(
                    QtGui.QTextDocument.ImageResource,
                    QtCore.QUrl(f"icon:{name}"),
                    icon.pixmap(QtCore.QSize(48, 48)))
        browser.setHtml(html)
        layout.addWidget(browser)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close, parent=dialog)
        buttons.rejected.connect(dialog.close)
        layout.addWidget(buttons)
        dialog.exec_()


    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()


    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue(
            'filename', self.filename if self.filename else '')
        self.settings.setValue('window/size', self.size())
        self.settings.setValue('window/position', self.pos())
        self.settings.setValue('window/state', self.saveState(1))
        self.settings.setValue('recentFiles', self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    # region Zoom and Scroll
    def scrollRequest(self, delta, orientation):
        units = - delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        bar.setValue(int(bar.value() + bar.singleStep() * units))

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(int(value))

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()

        units = delta * 0.1
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.scrollBars[Qt.Horizontal].setValue(
                self.scrollBars[Qt.Horizontal].value() + x_shift)
            self.scrollBars[Qt.Vertical].setValue(
                self.scrollBars[Qt.Vertical].value() + y_shift)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def adjustScale(self, initial=False):
            value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
            self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)
        
        # 更新坐标标签位置
        if hasattr(self, 'coordLabel'):
            QtCore.QTimer.singleShot(10, self.updateCoordLabelPosition)

    def updateMousePosition(self, x, y):
        """更新鼠标位置显示"""
        if not self.canvas.pixmap.isNull():
            img_name_info = f"{tr('File Name')}: {os.path.basename(self.filename)}"
            img_size_info = (
                f"{tr('Image Size')}: "
                f"{self.canvas.pixmap.width()}x{self.canvas.pixmap.height()}"
            )
            if x >= 0 and y >= 0:
                coord_info = f"{tr('Position')}: ({x}, {y})"
            else:
                coord_info = f"{tr('Position')}: (-, -)"
        else:
            img_name_info = f"{tr('File Name')}: {tr('No image')}"
            img_size_info = f"{tr('Image Size')}: {tr('No image')}"
            coord_info = f"{tr('Position')}: (-, -)"
        
        self.coordLabel.setText(f"{img_name_info}\n{img_size_info}\n{coord_info}")
        # 固定 180px 会把长文件名从左侧裁掉（右对齐时剩个 ".e" 残影）。
        # 宽度按最长一行自适应，设上限防止异常长的文件名盖住画布。
        fm = self.coordLabel.fontMetrics()
        needed = max(
            fm.horizontalAdvance(line)
            for line in (img_name_info, img_size_info, coord_info)
        ) + 24  # padding 8px*2 + 余量
        new_width = max(180, min(needed, 480))
        if new_width != self.coordLabel.width():
            self.coordLabel.setFixedSize(new_width, 60)
            self.updateCoordLabelPosition()
        
    def scrollAreaResizeEvent(self, event):
        """处理滚动区域大小变化，更新坐标标签位置"""
        # 调用原始的resize事件处理
        QtWidgets.QScrollArea.resizeEvent(self.centralWidget(), event)
        
        # 更新坐标标签位置到右下角
        scroll_area = self.centralWidget()
        label_width = self.coordLabel.width()
        label_height = self.coordLabel.height()
        
        # 计算位置，考虑滚动条的存在
        margin = 10
        x = scroll_area.width() - label_width - margin
        y = scroll_area.height() - label_height - margin
        
        # 如果有垂直滚动条，需要减去滚动条宽度
        if scroll_area.verticalScrollBar().isVisible():
            x -= scroll_area.verticalScrollBar().width()
            
        # 如果有水平滚动条，需要减去滚动条高度
        if scroll_area.horizontalScrollBar().isVisible():
            y -= scroll_area.horizontalScrollBar().height()
        
        self.coordLabel.move(max(0, x), max(0, y))
        self.updateEditHudPosition()

    def updateEditHudPosition(self):
        """把编辑 HUD 钉在画布右上角（让出竖滚动条）。"""
        hud = getattr(self, '_editHud', None)
        scroll_area = self.centralWidget()
        if hud is None or scroll_area is None or not hud.isVisible():
            return
        margin = 10
        x = scroll_area.width() - hud.width() - margin
        if scroll_area.verticalScrollBar().isVisible():
            x -= scroll_area.verticalScrollBar().width()
        hud.move(max(0, x), margin)

    def _buildEditHud(self, scrollArea):
        """创建画布浮条。两段内容按需出现：

        - 审查会话段："Back to Batch Review"，点 `Edit in main window` 后出现；
        - 画笔段：半径实时读数 + Done + 滚轮说明，Edit Mask 激活时出现。
        """
        hud = QtWidgets.QFrame(scrollArea)
        hud.setObjectName('editHud')
        hud.setStyleSheet(
            "QFrame#editHud { background: rgba(45,45,45,215);"
            "  border-radius: 8px; }"
            "QFrame#editHud QLabel { color: #e8e8e8; background: transparent; }"
            "QFrame#editHud QPushButton { padding: 4px 10px; }")
        layout = QtWidgets.QHBoxLayout(hud)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self._hudBack = QtWidgets.QPushButton(
            newIcon('back'), tr('Back to Batch Review'), hud)
        self._hudBack.clicked.connect(self._finishReviewReturn)
        layout.addWidget(self._hudBack)

        self._hudBrush = QtWidgets.QWidget(hud)
        brush = QtWidgets.QHBoxLayout(self._hudBrush)
        brush.setContentsMargins(0, 0, 0, 0)
        brush.setSpacing(8)
        self._hudRadius = QtWidgets.QLabel()
        brush.addWidget(self._hudRadius)
        hint = QtWidgets.QLabel(tr('wheel = brush size · Ctrl+wheel = zoom'))
        hint.setStyleSheet('color: #aaaaaa; font-size: 11px;')
        brush.addWidget(hint)
        done = QtWidgets.QPushButton(tr('Done'))
        done.clicked.connect(self._exitBrushMode)
        brush.addWidget(done)
        layout.addWidget(self._hudBrush)

        # Region Detect section: replaces the modal MaskConfirmDialog so the
        # user can keep drawing regions without leaving the mode. Exiting uses
        # the checkable toolbar action, so no Done button lives here next to
        # Detect where it could be hit by mistake.
        self._hudRegion = QtWidgets.QWidget(hud)
        region = QtWidgets.QHBoxLayout(self._hudRegion)
        region.setContentsMargins(0, 0, 0, 0)
        region.setSpacing(8)
        region_hint = QtWidgets.QLabel(tr('Region Detect'))
        region_hint.setStyleSheet('color: #e8e8e8;')
        region.addWidget(region_hint)
        mode_label = QtWidgets.QLabel(tr('Mode:'))
        mode_label.setStyleSheet('color: #aaaaaa; font-size: 11px;')
        region.addWidget(mode_label)
        self._hudModeGroup = QtWidgets.QButtonGroup(hud)
        self._hudModeGroup.setExclusive(True)
        self._hudModeButtons = {}
        for mode, icon, text, tip in (
            ('rect', 'rect', tr('Rect'),
             tr('Drag a rectangle')),
            ('freehand', 'free_line', tr('Freehand'),
             tr('Draw a freehand region')),
        ):
            button = QtWidgets.QPushButton(newIcon(icon), text, hud)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setStyleSheet(
                'QPushButton { border: none; padding: 3px 8px;'
                '  border-radius: 4px; }'
                'QPushButton:checked { background: rgba(255,255,255,45); }')
            self._hudModeGroup.addButton(button)
            region.addWidget(button)
            self._hudModeButtons[button] = (mode, icon)
            if mode == self.canvas.region_mode:
                button.setChecked(True)
        self._hudModeGroup.buttonToggled.connect(self._onRegionModeChanged)
        self._refreshRegionModeIcons()
        count_label = QtWidgets.QLabel(tr('Count:'))
        count_label.setStyleSheet('color: #aaaaaa; font-size: 11px;')
        region.addWidget(count_label)
        self._hudRegionAuto = QtWidgets.QCheckBox(tr('Auto'))
        self._hudRegionAuto.setChecked(True)
        self._hudRegionAuto.setStyleSheet('color: #e8e8e8;')
        self._hudRegionAuto.stateChanged.connect(
            lambda _s: self._hudRegionCount.setEnabled(
                not self._hudRegionAuto.isChecked()))
        region.addWidget(self._hudRegionAuto)
        self._hudRegionCount = QtWidgets.QSpinBox()
        self._hudRegionCount.setRange(1, 999)
        self._hudRegionCount.setValue(3)
        self._hudRegionCount.setToolTip(tr('Expected object count'))
        region.addWidget(self._hudRegionCount)
        self._hudRegionCount.setEnabled(False)
        region_detect = QtWidgets.QPushButton(tr('Detect'))
        region_detect.clicked.connect(self._runRegionDetect)
        region.addWidget(region_detect)
        region.addSpacing(12)
        region_exit = QtWidgets.QPushButton(tr('Exit'))
        region_exit.clicked.connect(self._exitMaskMode)
        region.addWidget(region_exit)
        layout.addWidget(self._hudRegion)

        # Measure section: stage-aware prompt for the Set Scale draw flow,
        # replacing the painted yellow text that used to sit on the canvas.
        self._hudMeasure = QtWidgets.QLabel('', hud)
        layout.addWidget(self._hudMeasure)

        hud.adjustSize()
        hud.hide()
        self._editHud = hud

    def _onRegionModeChanged(self, button, checked):
        if checked:
            mode, _icon = self._hudModeButtons.get(button, (None, None))
            if mode:
                self.canvas.region_mode = mode
            self._refreshRegionModeIcons()

    def _refreshRegionModeIcons(self):
        # One lit, the other dimmed: the active mode shows the full icon, the
        # inactive one is the same glyph at reduced opacity.
        for button, (_mode, icon) in getattr(
                self, '_hudModeButtons', {}).items():
            button.setIcon(
                newIcon(icon) if button.isChecked()
                else self._dimIcon(icon))

    @staticmethod
    def _dimIcon(name):
        base = newIcon(name).pixmap(20, 20)
        dim = QtGui.QPixmap(base.size())
        dim.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(dim)
        painter.setOpacity(0.35)
        painter.drawPixmap(0, 0, base)
        painter.end()
        return QtGui.QIcon(dim)

    def _updateEditHud(self):
        """按当前会话状态刷浮条：返回键看审查会话，画笔段看 BRUSH 模式。"""
        hud = getattr(self, '_editHud', None)
        if hud is None:
            return
        reviewing = getattr(self, '_reviewReturnDialog', None) is not None
        brushing = self.canvas.isBrushing()
        masking = self.canvas.isMasking()
        measuring = self.canvas.isMeasuring()
        self._hudBack.setVisible(reviewing)
        self._hudBrush.setVisible(brushing)
        self._hudRegion.setVisible(masking)
        self._hudMeasure.setVisible(measuring)
        if brushing:
            self._hudRadius.setText(
                tr('Brush Ø {radius} px', radius=self.canvas.brush_radius))
        hud.setVisible(reviewing or brushing or masking or measuring)
        hud.adjustSize()
        self.updateEditHudPosition()

    def _onBrushRadiusChanged(self, radius):
        if getattr(self, '_hudRadius', None) is not None:
            self._hudRadius.setText(tr('Brush Ø {radius} px', radius=radius))

    def _onMeasureStageChanged(self, stage):
        if getattr(self, '_hudMeasure', None) is None:
            return
        if stage <= 0:
            self._hudMeasure.setText(tr(
                'Measuring: click the first point (right-click to cancel)'))
        else:
            self._hudMeasure.setText(tr(
                'Measuring: click the second point (right-click to cancel)'))
        self._updateEditHud()

    def _exitBrushMode(self):
        if not self.canvas.isBrushing():
            return
        self.actions.editMask.setChecked(False)
        self.toggleMaskBrush()

    def updateCoordLabelPosition(self):
        """更新坐标标签到右下角位置"""
        scroll_area = self.centralWidget()
        if scroll_area and hasattr(self, 'coordLabel'):
            label_width = self.coordLabel.width()
            label_height = self.coordLabel.height()
            
            # 计算位置，考虑滚动条的存在
            margin = 10
            x = scroll_area.width() - label_width - margin
            y = scroll_area.height() - label_height - margin
            
            # 如果有垂直滚动条，需要减去滚动条宽度
            if scroll_area.verticalScrollBar().isVisible():
                x -= scroll_area.verticalScrollBar().width()
                
            # 如果有水平滚动条，需要减去滚动条高度
            if scroll_area.horizontalScrollBar().isVisible():
                y -= scroll_area.horizontalScrollBar().height()
            
            self.coordLabel.move(max(0, x), max(0, y))

    # endregion


    # region Basic File operations
    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if (filename in self.imageList and
                self.fileListWidget.currentRow() !=
                self.imageList.index(filename)):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            return

        self.resetState()
        if filename is None:
            filename = self.settings.value('filename', '')
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                'Error opening file', 'No such file: <b>%s</b>' % filename)
            return False
        # assumes same name, but json extension
        self.status("Loading %s..." % os.path.basename(str(filename)))

        # Load image:
        # read data first and store for saving into label file.
        def _read(filename, default=None):
            try:
                with open(filename, 'rb') as f:
                    return f.read()
            except Exception:
                return default
        self.imageData = _read(filename, None)
        if self.imageData is not None:
            # the filename is image not JSON
            self.imagePath = filename
        
        # 使用 PIL 读取图像并处理 EXIF 方向信息
        try:
            with Image.open(filename) as pil_image:
                # 处理 EXIF 方向信息
                try:
                    # 获取 EXIF 数据中的方向信息
                    exif = pil_image._getexif()
                    if exif is not None:
                        orientation_key = 274  # EXIF 方向标签的键值
                        if orientation_key in exif:
                            orientation = exif[orientation_key]
                            # 根据方向信息旋转图像
                            if orientation == 2:
                                pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                            elif orientation == 3:
                                pil_image = pil_image.rotate(180, expand=True)
                            elif orientation == 4:
                                pil_image = pil_image.rotate(180, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                            elif orientation == 5:
                                pil_image = pil_image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                            elif orientation == 6:
                                pil_image = pil_image.rotate(-90, expand=True)
                            elif orientation == 7:
                                pil_image = pil_image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                            elif orientation == 8:
                                pil_image = pil_image.rotate(90, expand=True)
                except (AttributeError, KeyError, TypeError):
                    # 如果没有 EXIF 数据或无法读取方向信息，使用原始图像
                    pass
                
                # 转换为 RGB 格式（如果需要）
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # 转换为 QImage
                width, height = pil_image.size
                rgb_data = pil_image.tobytes()
                image = QtGui.QImage(rgb_data, width, height, QtGui.QImage.Format_RGB888)
        except Exception as e:
            # 如果 PIL 处理失败，回退到原来的方法
            self.status(f"EXIF processing failed, using fallback method: {str(e)}")
            image = QtGui.QImage.fromData(self.imageData)
        
        if image.isNull():
            formats = ['*.{}'.format(fmt.data().decode())
                       for fmt in QtGui.QImageReader.supportedImageFormats()]
            self.errorMessage(
                'Error opening file',
                '<p>Make sure <i>{0}</i> is a valid image file.<br/>'
                'Supported image formats: {1}</p>'
                .format(filename, ','.join(formats)))
            self.status("Error reading %s" % filename)
            return False
        self.original_image = image.copy()
        self.image = self.original_image
        self.filename = filename
        self.canvas.loadQImage(self.image, update_sam=False)
        self.canvas.setEnabled(True)
        self.color_correction_data = None
        self.ruler_detection_data = None
        self.canvas.clear_color_checker_markers()
        self.canvas.clear_ruler_ticks() 
        
        # 重置颜色校正状态 - 但在批量模式下保留按钮状态
        if not self.batch_mode:
            self.actions.colorChecker.setChecked(False)  # 重置按钮状态
            self.actions.rulerDetection.setChecked(False)  # 重置按钮状态

        self.adjustScale(initial=True)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.setClean()
        
        # 初始化坐标显示位置
        self.updateMousePosition(-1, -1)  # 初始化为无效坐标
        QtCore.QTimer.singleShot(100, self.updateCoordLabelPosition)  # 延迟更新位置以确保界面已渲染
        
        self.status("Loaded image, preparing segmentation...")
        self._startSamImageLoad(self.original_image, str(filename))

        # Restore previously saved objects, if any. Done after the SAM load is
        # queued so the sidecar shows up immediately instead of waiting on the
        # embedding.
        if not self.batch_mode:
            self.loadAnnotation(filename)
        return True
    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False, load=True):
        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(str(self.filename)) if self.filename else '.'
        formats = ['*.{}'.format(fmt.data().decode())
                   for fmt in QtGui.QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(
            formats + ['*%s' % ".json"])
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self, '%s - Choose Image or Label file' % __appname__,
            path, filters)
        if QT5:
            filename, _ = filename
        filename = str(filename)
        if filename:
            self.loadFile(filename)
    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.setClean()
        self.resetState()

    def updateFileMenu(self):
        current = self.filename
        def exists(filename):
            return os.path.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QtWidgets.QAction(
                icon, '&%d %s' % (i + 1, QtCore.QFileInfo(f).fileName()), self)
            action.setProperty('phenoIcon', 'labels')
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def fileSelectionAboutChanged(self, current, previous):
        if previous and not self.previous:
            if not self.mayContinue():  # 如果 mayContinue() 返回 False
                self.previous = previous
                return

    def resetPrevious(self):
        """延迟重置 self.previous，确保信号槽执行完毕"""
        self.previous = None

    def fileSelectionChanged(self):

        if self.previous:
            # print(self.previous.text())
            self.fileListWidget.setCurrentItem(self.previous)
            QtCore.QTimer.singleShot(0, lambda: self.resetPrevious())
        else:
            items = self.fileListWidget.selectedItems()
            if not items:
                return
            item = items[0]
            currIndex = self.imageList.index(str(item.text()))
            if currIndex < len(self.imageList):
                filename = self.imageList[currIndex]
                # if filename:
                #     self.loadFile(filename)
                if filename and filename != self.imagePath:
                    self.loadFile(filename)



    def mayContinue(self):
        if not self.dirty or self.dont_show_again:
            return True

        # 创建 QMessageBox 实例，而不是直接使用 `mb` 类
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Warning)
        msg_box.setWindowTitle(tr("Confirmation Required"))
        msg_box.setText(tr("This image has unsaved objects.\n\nSave them before continuing?"))
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.Save)

        # 添加 “Don't show again” 复选框
        dont_show_again = QtWidgets.QCheckBox(tr("Don't show this message again"))
        msg_box.setCheckBox(dont_show_again)  # ✅ 在实例上调用 `setCheckBox()`

        # 显示对话框
        answer = msg_box.exec_()

        # 如果用户勾选 "Don't show again"，更新变量
        if dont_show_again.isChecked():
            self.dont_show_again = True  # 记录用户选择

        if answer == QtWidgets.QMessageBox.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.Save:
            return self.saveAnnotation()
        self.setClean()
        return True


    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(str(self.filename)) if self.filename else '.'

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filename) \
                if self.filename else '.'

        targetDirPath = str(QtWidgets.QFileDialog.getExistingDirectory(
            self, '%s - Open Directory' % __appname__, defaultOpenDirPath,
            QtWidgets.QFileDialog.ShowDirsOnly |
            QtWidgets.QFileDialog.DontResolveSymlinks))
        self.importDirImages(targetDirPath)

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    def importDirImages(self, dirpath, load=True):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()
        self.fileListWidget.selectAllCheckbox.setCheckState(Qt.Unchecked)
        for imgPath in self.scanAllImages(dirpath):
            item = QtWidgets.QListWidgetItem(imgPath)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)
        self.openNextImg(load=load)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower()
                      for fmt in QtGui.QImageReader.supportedImageFormats()]
        images = []
        if not os.path.isdir(folderPath):
            return images

        # Only the requested folder: derived artifacts live in subfolders
        # (masks/, corrected/, _out/) and must not enter the review queue.
        for file in os.listdir(folderPath):
            if file.lower().endswith(tuple(extensions)):
                images.append(os.path.join(folderPath, file))
        images.sort(key=lambda x: x.lower())
        return images

    # endregion 


    # region shapeList operations
    def newShape(self, shape):
        self.setDirty()
        self.addShape(shape)

    def addShape(self, shape):
        item = QtWidgets.QListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.shapeList.itemsToShapes.append((item, shape))
        self.shapeList.addItem(item)

    def editShapeLabel(self, item):
        if self.canvas.isdrawing():
            return
        shape = self.shapeList.get_shape_from_item(item)
        text, result = QtWidgets.QInputDialog.getText(
            self, tr('Edit Label'), tr('New name:'), text=shape.label)
        if not result:
            return

        text = str(text).strip()
        if self.shapeList.check_label(text, shape=shape):
            self.pushUndoSnapshot(tr('Rename object'))
            shape.label = text
            item.setText(text)
            self.renameTraitRow(shape)
            self.setDirty()
        else:
            QtWidgets.QMessageBox.warning(
                self, tr('Error'),
                tr('This name is empty or already used. Please pick another one.'))

    def _onCanvasShapePicked(self, shape):
        """画布点中对象 -> 对象列表选中它；性状表行由 labelSelectionChanged 跟进。

        走 setCurrentItem 而不是直接碰画布/表格，让现有监听器照常更新
        画布高亮、Delete/Edit Mask 可用性和性状表行（按 uid 匹配）。空白
        处点击清空选择，与在列表里点空白一致。
        """
        if shape is None:
            self.shapeList.clearSelection()
            return
        item = self.shapeList.get_item_from_shape(shape)
        if item is not None:
            self.shapeList.setCurrentItem(item)

    def labelSelectionChanged(self):
        if self.canvas.isdrawing():
            return
        shapes = self.shapeList.get_selected_shapes()
        self.canvas.setSelectedShapes(shapes)
        self.actions.delete.setEnabled(bool(shapes))
        if not self.canvas.isBrushing():
            # The brush edits one mask at a time; greyed out on multi-selection.
            self.actions.editMask.setEnabled(
                len(shapes) == 1 and shapes[0].mask is not None)
        elif shapes:
            # Switching rows while brushing retargets the brush.
            self.canvas.setBrushTarget(shapes[0])

        # Mirror the selection onto the traits table: the row computed from
        # the selected object gets highlighted.
        uid = shapes[0].uid if shapes else None
        self._syncTraitsHighlight(uid)

    def _syncTraitsHighlight(self, uid):
        """Highlight the traits row belonging to ``uid`` (None clears)."""
        model = self.phenomicsTableModel
        model.setHighlightKey(uid)
        if uid is None:
            return
        row = model.rowOfKey(uid)
        if row >= 0:
            # Make the row visible if it scrolled out of view.
            self.table_view.scrollTo(model.index(row, 0),
                                     QtWidgets.QAbstractItemView.PositionAtCenter)

    def _onTraitsRowClicked(self, index):
        """Traits row -> select and highlight the matching object.

        Runs after sorting, so the model's key (not the visual row) is the
        source of truth. Selection goes through the object list so every
        existing listener (canvas, brush, delete button) reacts as usual.
        """
        uid = self.phenomicsTableModel.keyAt(index.row())
        if uid is None:
            return
        shape = None
        for candidate in self.canvas.shapes:
            if candidate.uid == uid:
                shape = candidate
                break
        if shape is None:
            return
        item = self.shapeList.get_item_from_shape(shape)
        if item is None:
            return
        # Guard: labelSelectionChanged re-highlights the table; setting the
        # selection is enough, it propagates through the normal path.
        self.shapeList.blockSignals(False)
        self.shapeList.setCurrentItem(item)

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def labelDeSelect(self):

        self.shapeList.clearSelection()
        self.canvas.deSelectShape()
        self.actions.delete.setEnabled(False)

    def labelChanged(self, item):
        shape = self.shapeList.get_shape_from_item(item)
        if shape is None:
            return
        text = str(item.text()).strip()
        # Inline edits go through the same uniqueness rule; a rejected name is
        # rolled back to the current label.
        if text and text != shape.label:
            if self.shapeList.check_label(text, shape=shape):
                shape.label = text
                self.renameTraitRow(shape)
            else:
                item.setText(shape.label)
        elif not text:
            item.setText(shape.label)
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)


    # endregion

    # region Batch review
    def _shapeContextMenu(self, pos):
        """Right-click on an object row: Edit Mask / Delete.

        The clicked row joins the selection first, the way every list does it —
        a right click that silently acted on a different selection would surprise.
        The actions' own enabled state does the rest: Edit Mask stays greyed
        unless exactly one object is selected.
        """
        item = self.shapeList.itemAt(pos)
        if item is None:
            return
        if not item.isSelected():
            self.shapeList.clearSelection()
            item.setSelected(True)
        menu = QtWidgets.QMenu(self)
        for act in (self.actions.editMask, self.actions.delete):
            menu.addAction(act)
        menu.exec_(self.shapeList.viewport().mapToGlobal(pos))

    def reviewBatchResults(self, folder=None):
        """Open the standalone review window for a batch output folder."""
        if not folder:
            folder = QFileDialog.getExistingDirectory(
                self, tr('Choose a batch output folder...'), self.lastOpenDir or '')
        if not folder:
            return
        self.lastOpenDir = folder

        # Images usually live somewhere else than the batch output, so pass the
        # directories we know about as extra search paths.
        image_dirs = []
        if self.filename:
            image_dirs.append(os.path.dirname(self.filename))
        for path in self.imageList[:1]:
            image_dirs.append(os.path.dirname(path))

        dialog = BatchReviewDialog(self, folder=folder, image_dirs=image_dirs)
        # Keep a reference so the non-modal window is not garbage collected.
        self._reviewDialogs = [d for d in getattr(self, '_reviewDialogs', [])
                               if d.isVisible()]
        self._reviewDialogs.append(dialog)
        dialog.show()

    def beginReviewReturn(self, dialog, entry=None, prior_filename=None):
        """批处理审查窗点了"Edit in main window"之后挂上返回通道。

        审查窗藏起来、画布右上角浮出"Back to Batch Review"按钮；同时把批处理
        当时应用过的处理状态映射回来 —— 做过校色就按校色图打开、Color
        Calibration 呈已按下，做过标尺检测则 Ruler Detection 呈已按下（尺度
        本身已由 loadAnnotation 从 sidecar 恢复）。取消勾选或重跑都只影响
        当前这张图。
        """
        self._reviewReturnDialog = dialog
        self._reviewPriorFilename = prior_filename
        processing = ((entry or {}).get('document', {}) or {}).get(
            'processing') or {}
        stem = os.path.splitext(os.path.basename(self.filename or ''))[0]

        if (processing.get('color_correction')
                and stem.endswith(annotation.CORRECTED_SUFFIX)):
            # 打开的是校色成品图：预载原图让"取消校正"真能退回原像素。
            # entry 的完整性状态决定谁能当回退源 —— moved 的副本内容一致可退，
            # replaced/missing 的不能拿来冒充原图。
            self._loadedColorCorrected = True
            state = (entry or {}).get('image_state', '')
            original = ((entry or {}).get('image_path')
                        if state in ('', 'moved') else None)
            self._colorRevertImage = (
                QtGui.QImage(original) if original else None)
            if self._colorRevertImage is not None:
                self.original_image = self._colorRevertImage
            # 写回时 sidecar 仍登记原图身份；文件缺失也保留记录的位置。
            self._colorOriginalPath = (
                (entry or {}).get('recorded_path') or original)
            self.actions.colorChecker.setChecked(True)
        if processing.get('ruler_detection'):
            self._loadedRulerDetection = True
            self.actions.rulerDetection.setChecked(True)

        self._updateEditHud()
        self.status(tr(
            'Editing for batch review \u2014 press "Back to Batch Review" when done.'))

    def _finishReviewReturn(self):
        """写回 sidecar 并返回审查窗。写向审查目录本身，不走 annotationPath —
        BATCH_PARAM 里残留的 save_dir 可能把结果导去上一批的输出目录。"""
        dialog = getattr(self, '_reviewReturnDialog', None)
        folder = getattr(dialog, '_folder', None) if dialog is not None else None
        written = None
        if self.filename and folder:
            stem = os.path.splitext(os.path.basename(self.filename))[0]
            # 在校色图上做的编辑同样写回原名那份 sidecar。
            if stem.endswith(annotation.CORRECTED_SUFFIX):
                stem = stem[:-len(annotation.CORRECTED_SUFFIX)]
            target = os.path.join(folder, stem + annotation.SIDECAR_SUFFIX)
            try:
                annotation.save_document(target, self.buildAnnotationDocument())
                self.setClean()
                written = target
                self.status(tr(
                    'Saved {count} object(s) to {name}.',
                    count=len(self.canvas.shapes),
                    name=stem + annotation.SIDECAR_SUFFIX))
            except OSError as error:
                QMessageBox.critical(
                    self, tr('Save Annotation'),
                    tr('Failed to save annotation: {message}', message=str(error)))
        elif self.filename:
            self.saveAnnotation(silent=True)

        self._reviewReturnDialog = None
        prior = self._reviewPriorFilename
        self._reviewPriorFilename = None
        self._updateEditHud()
        # 画布恢复到进入编辑前的状态：之前有别的图就重载回来（连同它的
        # sidecar/比例尺一起恢复），否则清成空态 —— 不把审查编辑的图留在界面上。
        if prior and prior != self.filename and os.path.isfile(prior):
            self.loadFile(prior)
        elif prior and prior == self.filename:
            pass  # 编辑的就是之前那张图，画布保持现状即可
        else:
            self.resetState()
            self.canvas.update()
        # 无论写没写成都要把审查窗还回来 —— 写失败已弹过错框，留着隐藏窗才是坑。
        if dialog is not None:
            dialog.returnFromMainWindow(written)
    # endregion

    # region Mask brush editing
    def toggleMaskBrush(self):
        """Enter/leave brush mode for the highlighted object."""
        if not self.actions.editMask.isChecked():
            self.setMode(Canvas.NORMAL)
            self.status(tr('Mask editing finished.'))
            self._updateEditHud()
            return

        shapes = self.shapeList.get_selected_shapes()
        target = shapes[0] if shapes else self.canvas.selectedShape
        if target is None or target.mask is None:
            self.actions.editMask.setChecked(False)
            QMessageBox.information(
                self, tr('Edit Mask'),
                tr('Select an object in the Objects panel first.'))
            self._updateEditHud()
            return

        self.canvas.setBrushTarget(target)
        self.setMode(Canvas.BRUSH)
        self.canvas.setFocus()
        self._updateEditHud()
        self.status(tr(
            'Editing "{name}": left button adds, right button removes, '
            'wheel resizes the brush, Ctrl+wheel zooms (radius {radius} px).',
            name=target.label, radius=self.canvas.brush_radius))

    def _onBrushStrokeStarted(self):
        # One undo step per stroke.
        self.pushUndoSnapshot(tr('Edit mask'))

    def _onBrushStrokeFinished(self, shape):
        if shape is None:
            return
        self.setDirty()
        # The mask changed, so any previously computed traits for it are wrong.
        self.invalidateTraitsTable()
        self.updateTraitsTableTitle()
        area = int((shape.mask > 0).sum()) if shape.mask is not None else 0
        self.status(tr(
            'Mask of "{name}" updated, area {area} px.',
            name=shape.label, area=area))
    # endregion

    # region Undo
    def _snapshotShapes(self):
        """Capture every object of the current image.

        Masks are stored packbits+zlib(level 1) rather than as raw copies: a
        4624x3472 mask is 15.3 MB raw but ~15 KB packed, and packing costs about
        4 ms, so a 20-step history stays in the low megabytes.
        """
        entries = []
        for shape in self.canvas.shapes:
            mask = shape.mask
            packed = None
            if mask is not None:
                binary = (np.asarray(mask) > 0).astype(np.uint8)
                packed = (
                    zlib.compress(np.packbits(binary.reshape(-1)).tobytes(), 1),
                    binary.shape,
                )
            entries.append({
                'uid': shape.uid,
                'label': shape.label,
                'name_prefix': shape.name_prefix,
                'source_prompt': shape.source_prompt,
                'detection_score': getattr(shape, 'detection_score', None),
                'points': [(p.x(), p.y()) for p in shape.points],
                'prompt_points': [(p.x(), p.y()) for p in shape.prompt_points],
                'prompt_labels': list(shape.prompt_labels),
                'mask': packed,
                'checked': self._isShapeChecked(shape),
            })
        return entries

    def _isShapeChecked(self, shape):
        item = self.shapeList.get_item_from_shape(shape)
        return item is None or item.checkState() == Qt.Checked

    def _restoreShapes(self, entries):
        """Rebuild the object list from a snapshot taken by _snapshotShapes."""
        self.shapeList.blockSignals(True)
        try:
            self.shapeList.clear()
            self.canvas.shapes = []
            self.canvas.selectedShapes = []
            self.canvas.visible = {}

            for entry in entries:
                shape = Shape()
                mask = None
                if entry['mask'] is not None:
                    payload, shape_size = entry['mask']
                    flat = np.unpackbits(
                        np.frombuffer(zlib.decompress(payload), dtype=np.uint8))
                    count = shape_size[0] * shape_size[1]
                    mask = flat[:count].reshape(shape_size).astype(np.uint8)
                shape.set(points=entry['points'], mask=mask, label=entry['label'])
                shape.uid = entry['uid']
                shape.name_prefix = entry['name_prefix']
                shape.source_prompt = entry['source_prompt']
                shape.detection_score = entry['detection_score']
                for (x, y), label in zip(entry['prompt_points'], entry['prompt_labels']):
                    shape.add_prompt_point(x, y, label)

                self.canvas.shapes.append(shape)
                # Restored uids are authoritative; keep the counter ahead of them
                # so future objects never collide with a restored one.
                if shape.uid:
                    self.canvas.totalShapes = max(self.canvas.totalShapes, shape.uid)
                self.addShape(shape)
                if not entry['checked']:
                    item = self.shapeList.get_item_from_shape(shape)
                    item.setCheckState(Qt.Unchecked)
                    self.canvas.setShapeVisible(shape, False)
        finally:
            self.shapeList.blockSignals(False)

        # The brush was editing an object instance that no longer exists after
        # the rebuild; retarget it to the restored object with the same uid.
        if self.canvas.isBrushing() and self.canvas.brush_target is not None:
            old_uid = getattr(self.canvas.brush_target, 'uid', None)
            target = None
            if old_uid is not None:
                for shape in self.canvas.shapes:
                    if shape.uid == old_uid:
                        target = shape
                        break
            self.canvas.setBrushTarget(target)

        self.canvas.update()

    def pushUndoSnapshot(self, description=None):
        """Record the current object state so it can be restored later."""
        self._undoStack.append((self._snapshotShapes(), description))
        if len(self._undoStack) > self.MAX_UNDO_STEPS:
            self._undoStack.pop(0)
        self.actions.undo.setEnabled(True)

    def clearUndoStack(self):
        self._undoStack = []
        if hasattr(self, 'actions'):
            self.actions.undo.setEnabled(False)

    def undoLastChange(self):
        if not self._undoStack:
            return
        entries, description = self._undoStack.pop()
        self._restoreShapes(entries)
        self.actions.undo.setEnabled(bool(self._undoStack))
        # The traits table was computed against the pre-undo objects.
        self.invalidateTraitsTable()
        self.updateTraitsTableTitle()
        self.setDirty()
        if description:
            self.status(tr('Undone: {action}', action=description))
        else:
            self.status(tr('Undone.'))
    # endregion

    # region Annotation persistence
    def batchAnnotationPath(self, filename):
        """批处理输出目录里与该图对应的 sidecar 路径（不一定存在）。

        批处理把每张图的对象写在输出目录（batch.process_checked_shapes），而不是
        图片旁边，所以复查或补标时必须回到那里去找。
        """
        params = getattr(self.canvas, 'BATCH_PARAM', None) or {}
        save_dir = params.get('save_dir')
        if not save_dir or not filename:
            return None
        return os.path.join(
            save_dir,
            os.path.splitext(os.path.basename(filename))[0]
            + annotation.SIDECAR_SUFFIX)

    def annotationPath(self, filename=None):
        """Sidecar path for the given (or current) image.

        输出目录里存在同名 sidecar 时就优先用它 —— 那是批处理写下的那一份 ——
        否则退回图片旁边的默认位置。读写都走这里，才不会两边各存一份、互相
        覆盖不上。
        """
        target = filename or self.filename
        if not target:
            return None
        # 批处理导出的 ``*_color_corrected.*`` 没有自己的 sidecar —— 打开它时
        # 要找的是去掉后缀的同名原图那份（批审查编辑校色图走的就是这条路）。
        stem = os.path.splitext(os.path.basename(target))[0]
        if stem.endswith(annotation.CORRECTED_SUFFIX):
            stem = stem[:-len(annotation.CORRECTED_SUFFIX)]
            target = os.path.join(
                os.path.dirname(target),
                stem + os.path.splitext(target)[1])
        batched = self.batchAnnotationPath(target)
        if batched and os.path.exists(batched):
            return batched
        return annotation.sidecar_path(target)

    def buildAnnotationDocument(self):
        """Snapshot the current image's objects as a sidecar document."""
        image_ref = self.filename
        digest = None
        stem = os.path.splitext(os.path.basename(self.filename or ''))[0]
        if (stem.endswith(annotation.CORRECTED_SUFFIX)
                and getattr(self, '_colorOriginalPath', None)):
            # 掩膜属于原图；校色图是派生产物，sidecar 仍登记原图身份。
            # 原图文件缺失时 image_digest 返回 None，md5 记为空而非错值。
            image_ref = self._colorOriginalPath
            digest = annotation.image_digest(image_ref)
        return annotation.build_document(
            image_ref,
            self.canvas.shapes,
            (self.canvas.pixmap.width(), self.canvas.pixmap.height()),
            scale_per_pixel=self.canvas.scale_per_pixel,
            scale_unit=self.canvas.scale_unit,
            color_correction=(self.color_correction_data is not None
                              or getattr(self, '_loadedColorCorrected', False)),
            ruler_detection=(self.ruler_detection_data is not None
                             or getattr(self, '_loadedRulerDetection', False)),
            digest=digest,
            # centers/tickpoints 是 numpy 数组，不能用 `or` 判真值，
            # 会触发 ndarray 歧义报错（崩溃日志 ValueError 就是它）。
            color_checker_centers=(
                (self.color_correction_data or {}).get('centers')
                if (self.color_correction_data or {}).get('centers') is not None
                else self.canvas.color_checker_centers),
            ruler_tickpoints=(
                (self.ruler_detection_data or {}).get('tickpoints')
                if (self.ruler_detection_data or {}).get('tickpoints') is not None
                else self.canvas.ruler_ticks),
            correction_matrix=(
                (self.color_correction_data or {}).get('correction_matrix')),
        )

    def saveAnnotation(self, silent=False):
        """Write the current objects next to the image. Returns True on success."""
        if not self.filename:
            if not silent:
                QMessageBox.information(
                    self, tr('Save Annotation'), tr('Please load an image first.'))
            return False

        path = self.annotationPath()
        try:
            annotation.save_document(path, self.buildAnnotationDocument())
        except OSError as error:
            if not silent:
                QMessageBox.critical(
                    self, tr('Save Annotation'),
                    tr('Failed to save annotation: {message}', message=str(error)))
            return False

        self.setClean()
        self.status(tr(
            'Saved {count} object(s) to {name}.',
            count=len(self.canvas.shapes),
            name=os.path.basename(path),
        ))
        return True

    def deleteAnnotation(self):
        """Delete this image's sidecar and return the view to unannotated.

        One destructive act, three effects: the JSON goes away, the canvas
        loses its objects and markers, and processing state (colour
        correction, ruler scale) resets to defaults -- the pair of
        "everything the file carried, gone". The image itself is untouched.
        """
        path = self.annotationPath() if self.filename else None
        if not path or not os.path.exists(path):
            self.status(tr('No saved annotation for this image.'))
            return
        answer = QMessageBox.warning(
            self, tr('Delete Annotation'),
            tr('Delete the saved annotation for this image?\n'
               'The image file itself is not affected.'),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        try:
            os.remove(path)
        except OSError as error:
            QMessageBox.critical(
                self, tr('Delete Annotation'),
                tr('Failed to delete annotation: {message}', message=str(error)))
            return

        self.shapeList.clear()
        self.canvas.shapes = []
        self.canvas.selectedShapes = []
        self.canvas.visible = {}
        self.canvas.clear_color_checker_markers()
        self.canvas.clear_ruler_ticks()

        # Processing state from the file must die with the file: raw image
        # back on screen, both toggles released, scale back to pixels.
        self.color_correction_data = None
        self.ruler_detection_data = None
        self._loadedColorCorrected = False
        self._loadedRulerDetection = False
        self._colorOriginalPath = None
        self._colorRevertImage = None
        self.actions.colorChecker.setChecked(False)
        self.actions.rulerDetection.setChecked(False)
        if self.original_image is not None:
            self.show_original_image()
        self.canvas.scale_per_pixel = 1.0
        self.canvas.scale_unit = 'pixel'
        self.updateScaleLabel()
        self.refreshTraitHeaders()
        self.results = []
        self.result_keys = []
        self.phenomicsTableModel.updateData([], self.traitHeaders())
        self.updateTraitsTableTitle()
        self.clearUndoStack()
        self.setClean()
        self.canvas.update()
        self.status(tr('Deleted annotation {name}.', name=os.path.basename(path)))

    def loadAnnotation(self, filename):
        """Restore objects from the sidecar of ``filename``, when present."""
        path = self.annotationPath(filename)
        document = annotation.load_document(path)
        if not document:
            return 0

        recorded = (document.get('image') or {}).get('md5')
        # 校色图的像素和原图本来就不同 —— 那是管线特意生成的变体，
        # 不属于"sidecar 写后图像被改动"的警告场景，直接放行。
        is_corrected_variant = os.path.splitext(os.path.basename(filename))[0].endswith(
            annotation.CORRECTED_SUFFIX)
        if (recorded and recorded != annotation.image_digest(filename)
                and not is_corrected_variant):
            # The image changed since the sidecar was written; pixel coordinates
            # can no longer be trusted, so ask instead of silently misplacing.
            answer = QMessageBox.warning(
                self, tr('Load Annotation'),
                tr('The saved annotation was created from a different version '
                   'of this image. Load it anyway?'),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return 0

        image_size = annotation.document_image_size(document) or (
            self.canvas.pixmap.width(), self.canvas.pixmap.height())

        shapes = []
        for data in document.get('objects') or []:
            try:
                shape = annotation.apply_dict_to_shape(
                    Shape(), data, image_size=image_size)
            except (KeyError, ValueError) as error:
                self.status(tr(
                    'Failed to load annotation: {message}', message=str(error)))
                continue
            shapes.append(shape)

        if not shapes:
            return 0

        scale = document.get('scale') or {}
        self.canvas.scale_per_pixel = float(scale.get('per_pixel', 1.0))
        self.canvas.scale_unit = str(scale.get('unit', 'pixel'))
        self.updateScaleLabel()
        # 标注文件自带比例，所以表头也要按文件里的单位重建。
        self.refreshTraitHeaders()

        # 恢复批处理记录在 sidecar 里的检测点位：色卡中心和标尺刻度只是
        # 展示信息，直接画回画布，不必重跑检测模型。
        processing = document.get('processing') or {}
        centers = processing.get('color_checker_centers')
        if centers:
            self.canvas.set_color_checker_markers(centers)
        tickpoints = processing.get('ruler_tickpoints')
        if tickpoints:
            self.canvas.set_ruler_ticks(tickpoints)
        if processing.get('ruler_detection'):
            self._loadedRulerDetection = True
            self.actions.rulerDetection.setChecked(True)
        if processing.get('color_correction'):
            stem = os.path.splitext(os.path.basename(self.filename or ''))[0]
            if stem.endswith(annotation.CORRECTED_SUFFIX):
                # The open file IS the corrected product: applying the matrix
                # again would double-correct. Mark the state and preload the
                # raw sibling as the revert source, mirroring the
                # batch-review path.
                self._loadedColorCorrected = True
                raw_name = (
                    stem[:-len(annotation.CORRECTED_SUFFIX)]
                    + os.path.splitext(self.filename)[1])
                raw_path = os.path.join(
                    os.path.dirname(self.filename), raw_name)
                self._colorOriginalPath = raw_path
                if os.path.exists(raw_path):
                    self._colorRevertImage = QtGui.QImage(raw_path)
                    self.original_image = self._colorRevertImage
                else:
                    self._colorRevertImage = None
                self.actions.colorChecker.setChecked(True)
            else:
                self._restoreColorCorrection(processing, centers)

        self.canvas.addLoadedShapes(shapes)
        self.setClean()
        self.status(tr(
            'Loaded {count} saved object(s) from {name}.',
            count=len(shapes),
            name=os.path.basename(path),
        ))
        return len(shapes)
    # endregion

    
    # region Phenomics Calculation
    def calcSelectedMasks(self):

        self.results = []
        self.result_keys = []
        self.headers = self.traitHeaders()

        checked_shapes = self.shapeList.get_checked_shapes()
        counts = len(checked_shapes)
        self.workers = []
        self.completed_tasks = 0

        if counts == 0:
            QMessageBox.information(
                self,
                tr('Calculate Selected Objects'),
                tr('Please select at least one Object.'),
            )
            return
        
        # Create a progress dialog
        self.progress_dialog_phenocalc = QtWidgets.QProgressDialog(f"Processing: 0/{counts}", "Cancel", 0, counts, self)
        self.progress_dialog_phenocalc.setWindowFlags(self.progress_dialog_phenocalc.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Hide the help button
        self.progress_dialog_phenocalc.setFixedSize(self.progress_dialog_phenocalc.sizeHint())  # Set fixed size

        self.progress_dialog_phenocalc.setWindowModality(Qt.WindowModal)
        self.progress_dialog_phenocalc.setValue(0)
        self.progress_dialog_phenocalc.canceled.connect(self.cancel_all_tasks)

        self.progress_dialog_phenocalc.setAutoClose(False)
        self.progress_dialog_phenocalc.setAutoReset(False)


        for shape in checked_shapes:
            worker = Worker(PhenomicsCalculator.calcPhenomics, f"Processing {shape}", 
                          shape, self.canvas.pixmap, self.canvas.scale_per_pixel)
            worker.signals.finished.connect(self.update_progress)
            worker.signals.error.connect(lambda msg: self.handle_error(msg, worker, self.progress_dialog_phenocalc))
            # Workers finish out of order, so each result carries the uid of the
            # object it came from; that is what lets the table follow later
            # renames and deletions.
            worker.signals.result.connect(
                lambda result, uid=shape.uid: self.handle_result(result, uid)
            )
            self.workers.append(worker)  # Keep references to workers
            self.threadpool.start(worker) # Start the worker in the threadpool


        self.gpuMonitor.set_busy(True)
        try:
            self.progress_dialog_phenocalc.exec_()
        finally:
            self.gpuMonitor.set_busy(False)

    def cancel_all_tasks(self):
        """取消所有任务"""
        # print("Cancelling all tasks...")
        for worker in self.workers:
            worker.stop()  # 设置 is_running = False，让任务终止
        self.workers.clear()
        self.progress_dialog_phenocalc.close()
        self.status("Traits Calculation Cancelled.")

    def update_progress(self, task_name):
        """任务完成时更新进度条"""
        if self.progress_dialog_phenocalc.wasCanceled():
            return
        self.completed_tasks += 1
        self.progress_dialog_phenocalc.setValue(self.completed_tasks)
        self.progress_dialog_phenocalc.setLabelText(f"Processing: {self.completed_tasks}/{self.progress_dialog_phenocalc.maximum()}")
        if self.completed_tasks >= self.progress_dialog_phenocalc.maximum():
            self.status("Traits Calculation Completed.")
            self.progress_dialog_phenocalc.setCancelButtonText("Finish")  # 改变按钮文本
            try:
                self.progress_dialog_phenocalc.canceled.disconnect(self.cancel_all_tasks)  # 断开取消信号
            except TypeError:
                pass  # 如果信号未连接，忽略错误
            self.progress_dialog_phenocalc.canceled.connect(self.progress_dialog_phenocalc.close)
            self.update_phenomics_table()

    def handle_error(self, error_msg, worker=None, waiting_dialog=None):
        """处理自动掩码检测错误"""
        QtWidgets.QMessageBox.critical(self, tr("Processing Error"), error_msg)
        self.status(error_msg)
        if worker:
            worker.stop()  # 停止任务
        if waiting_dialog:
            waiting_dialog.close_dialog()

    @QtCore.Slot()
    def _computeBatchTraits(self):
        """Compute this frame's traits and refresh the table -- GUI thread only.

        batchWork calls this via BlockingQueuedConnection: the worker cannot
        touch the canvas pixmap (a GUI object) itself, and a queued push would
        race the repaint that the crash reported as QBackingStore::endPaint().
        """
        rows, keys = [], []
        for shape in self.shapeList.get_checked_shapes():
            try:
                rows.append(PhenomicsCalculator.calcPhenomics(
                    shape, self.canvas.pixmap, self.canvas.scale_per_pixel))
                keys.append(shape.uid)
            except Exception:
                continue
        self.results = rows
        self.result_keys = keys
        self.headers = self.traitHeaders()
        self.update_phenomics_table()

    def handle_result(self, result, uid=None):
        self.results.append(result)
        self.result_keys.append(uid)

    def update_phenomics_table(self):
        self.phenomicsTableModel.updateData(
            self.results, self.headers, keys=self.result_keys)
        self.phenomicsTableModel.sort(0, Qt.AscendingOrder)
        # 使用模型的智能列宽设置
        self.phenomicsTableModel.setupTableView(self.table_view)
        self.table_view.reset()
        self.updateTraitsTableTitle()

    def updateTraitsTableTitle(self):
        """Mark the dock title when the table no longer matches the objects."""
        title = tr('Traits Table')
        if self.phenomicsTableModel.isStale():
            title += tr(' (outdated - recalculate)')
        self.table_dock.setWindowTitle(title)

    def invalidateTraitsTable(self, reason=None):
        """Flag the table as outdated after the objects changed."""
        if self.phenomicsTableModel.isEmpty():
            return
        if self.phenomicsTableModel.setStale(True):
            self.updateTraitsTableTitle()
            if reason:
                self.status(reason)

    def dropTraitRows(self, shapes):
        """Remove table rows for deleted objects, keeping the rest usable."""
        keys = [getattr(shape, 'uid', None) for shape in shapes]
        removed = self.phenomicsTableModel.removeRowsByKeys(keys)
        if not removed:
            return 0
        # The model owns the surviving rows now; mirror them so a later
        # recalculation starts from a consistent state.
        self.results = [list(row) for row in self.phenomicsTableModel._data]
        self.result_keys = list(self.phenomicsTableModel._row_keys)
        self.updateTraitsTableTitle()
        return removed

    def renameTraitRow(self, shape):
        """Follow a rename so the table's Name column stays truthful."""
        if self.phenomicsTableModel.renameRow(getattr(shape, 'uid', None), shape.label):
            self.results = [list(row) for row in self.phenomicsTableModel._data]
            self.result_keys = list(self.phenomicsTableModel._row_keys)


    def export(self):
        """导出表格数据"""
        if self.phenomicsTableModel.isEmpty():
            # Use `QMessageBox.question()` to ask the user whether to proceed
            reply = QMessageBox.question(self, tr("Export Prompt"), tr("No data available. Do you want to export the headers anyway?"),
                                        QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return  # User canceled the export, exit the function

        default_path = ""
        if self.filename:
            stem = os.path.splitext(os.path.basename(self.filename))[0]
            default_path = os.path.join(
                os.path.dirname(self.filename), stem + ".csv")
        file_path, _ = QFileDialog.getSaveFileName(
            self, tr("Save File"), default_path,
            "CSV Files (*.csv);;All Files (*)")

        if file_path:
            self.phenomicsTableModel.export(file_path)


    def manageAlgorithm(self):
        
        dialog = ManageAlgorithmDialog(
            self.custom_algorithms, 
            self.custom_headers,
            self.imported_names, 
            self.loaded_names, 

            on_save_callback=self.update_custom_algorithms)
        dialog.exec_()

    def update_custom_algorithms(self, custom_algorithms, custom_headers, imported_names, loaded_names):
        self.custom_algorithms = custom_algorithms
        self.custom_headers = custom_headers
        self.imported_names = imported_names
        self.loaded_names = loaded_names

        loaded_algorithms = {name: self.custom_algorithms[name] for name in loaded_names}
        loaded_headers = {name: self.custom_headers[name] for name in loaded_names}
        PhenomicsCalculator.registerAlgorithm(loaded_algorithms, loaded_headers)

        # 更新表格模型的标题
        new_headers = self.traitHeaders()
        self.phenomicsTableModel.updateData([], new_headers)
        # 重新设置表格视图以应用智能列宽管理
        self.phenomicsTableModel.setupTableView(self.table_view)

        self.statusBar().showMessage("Custom algorithms updated")

    # endregion

    # region Save Masks
    def save_selected_masks_worker(self, shapes, save_dir, filename):
        """后台线程：保存选中的 mask"""
        return batch.save_selected_masks_worker(shapes, save_dir, filename)

    def saveSelectedMasks(self):
        """导出勾选对象的掩膜为 1-bit PNG。

        Named for what it is rather than for the button that calls it: nothing reads
        these files back, and the sidecar carries the same masks as RLE.
        """
        checked_shapes = self.shapeList.get_checked_shapes()
        if not checked_shapes:
            QMessageBox.information(
                self,
                tr("Export Masks"),
                tr("Please select at least one Object."),
            )
            return

        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory to Save Objects", self.currentPath())
        # print(dir_path)
        if not dir_path:
            return

        # 创建 Worker 并启动
        worker = Worker(
            self.save_selected_masks_worker, 
            "Save Masks", 
            checked_shapes, dir_path, self.filename
        )

        # 连接信号
        worker.signals.result.connect(lambda msg: self.status(msg))  # 任务完成
        worker.signals.error.connect(lambda msg: self.handle_error(msg, worker, None))

        self.threadpool.start(worker)
        # endregion

    # region Semantic Detection
    def textPromptDetect(self):
        if not self._sam_ready or self.original_image is None:
            QMessageBox.information(
                self,
                tr("Semantic Detect"),
                tr("Please wait until the current image is ready."),
            )
            return

        dialog = TextPromptDialog(self, initial=self.lastTextPrompt)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        prompt = dialog.get_prompt()
        if not prompt:
            QMessageBox.information(
                self,
                tr("Semantic Detect"),
                tr("Please enter at least one text concept."),
            )
            return

        self.lastTextPrompt = prompt

        if self.batch_mode:
            # Recording only: nothing is detected on the current image.
            step = {"type": "text", "prompt": prompt}
            self.recordBatchStep(step)
            self.status(
                tr("Recorded batch step: {step}", step=describe_batch_step(step))
            )
            return

        self._startTextDetect(prompt)

    def _startTextDetect(self, prompt, skip_semantic_check=False):
        if self.text_detector is None:
            self.text_detector = TextDetector(device=self.canvas.sam.device)

        request_id = self._sam_request_id
        waiting_dialog = WaitingDialog(
            self, tr('Detecting "{prompt}"...', prompt=prompt)
        )
        waiting_dialog.show()

        self.actions.click.setEnabled(False)
        self.actions.auto.setEnabled(False)
        self.actions.textDetect.setEnabled(False)
        self.status(tr("Loading semantic detector and detecting objects..."))

        worker = Worker(
            self._detectTextPrompt,
            "Semantic Detect",
            self.original_image.copy(),
            prompt,
            request_id,
            skip_semantic_check,
        )
        worker.signals.result.connect(self._handleTextDetectResult)
        worker.signals.error.connect(
            lambda message: self._handleTextDetectError(message, waiting_dialog)
        )
        worker.signals.finished.connect(
            lambda _name: self._finishTextDetect(waiting_dialog)
        )
        self.sam_threadpool.start(worker)

    def _detectTextPrompt(
        self,
        image,
        prompt,
        request_id,
        skip_semantic_check=False,
    ):
        cv_image = qimage_to_cv2(image)
        if cv_image.ndim == 2:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        elif cv_image.shape[2] == 4:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
        else:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        result = {
            "request_id": request_id,
            "prompt": prompt,
            "detections": 0,
            "shapes": 0,
            "filtered": 0,
            "rejected": 0,
            "aliases": TextDetector.alias_notes(prompt),
            "mismatch": None,
        }

        if not skip_semantic_check:
            # Validate the prompt against the image before spending time on the
            # target detection and SAM segmentation.
            mismatch = self.text_detector.cross_check(rgb_image, prompt)
            if mismatch is not None:
                result["mismatch"] = mismatch
                return result

        # detect_with_report also tells us what the calibration-object veto
        # and the whole-frame filter removed, so the status bar can explain it.
        detections, rejected = self.text_detector.detect_with_report(
            rgb_image,
            prompt,
        )
        result["detections"] = len(detections)
        result["rejected"] = len(rejected)
        if request_id != self._sam_request_id:
            return result

        segmentation_result = self.canvas.segmentTextDetections(
            detections,
            request_id=request_id,
            prompt=prompt,
            image_rgb=rgb_image,
            detector=self.text_detector,
        )
        result["shapes"] = segmentation_result["created"]
        result["filtered"] = segmentation_result["filtered"]
        return result

    def _handleTextDetectResult(self, result):
        if result["request_id"] != self._sam_request_id:
            return

        mismatch = result.get("mismatch")
        if mismatch is not None:
            requested = ", ".join(
                mismatch.get("requested") or [result.get("prompt", "")]
            )
            self.status(
                tr(
                    'Semantic detection warning: "{requested}" may not match this image.',
                    requested=requested,
                )
            )
            # Let the waiting dialog close first, otherwise the warning would be
            # stacked on top of a modal progress dialog.
            QtCore.QTimer.singleShot(
                0,
                lambda r=result, m=mismatch: self._handleTextDetectMismatch(r, m),
            )
            return

        message = tr(
            "Semantic detection found {boxes} box(es), created {objects} object(s).",
            boxes=result["detections"],
            objects=result["shapes"],
        )
        if result["filtered"]:
            message += " " + tr(
                "Removed {count} size/shape outlier(s).", count=result["filtered"]
            )
        if result.get("rejected"):
            message += " " + tr(
                "Ignored {count} calibration object(s) or whole-frame box(es).",
                count=result["rejected"],
            )
        # Tell the user their wording was translated for the model; otherwise
        # object names built from the typed word look inconsistent with the
        # result.
        for raw, normalized in result.get("aliases") or []:
            message += " " + tr(
                'Searched "{raw}" as "{normalized}".',
                raw=raw, normalized=normalized,
            )

        self.status(message)

    def greenLeafDetect(self):
        """按颜色分割绿色叶片（剪下的叶片摆在灯箱/浅色背景上的场景）。

        批量模式下只记录步骤、不动当前图；否则直接在当前图上分割。原理与适用
        范围见 pheno/greenSeg.py。
        """
        if self.batch_mode:
            step = {"type": "green"}
            self.recordBatchStep(step)
            self.status(tr('Recorded batch step: {step}',
                           step=describe_batch_step(step)))
            return

        if self.original_image is None:
            QMessageBox.information(
                self, tr('Green Leaf Detect'),
                tr('Please load an image first.'))
            return

        result = self.canvas.segmentGreenLeafs(self.original_image)
        if result['created']:
            self.status(tr('Green leaf segmentation created {count} object(s).',
                           count=result['created']))
        else:
            self.status(tr('No green leaf was found. This method needs green '
                           'leaves on a light background.'))

    def detectTextPromptSync(self, prompt):
        """Run text detection + segmentation synchronously on the current image.

        Used by batch processing to replay a recorded prompt on every image.
        The semantic cross-check is skipped because batch steps are replayed
        as recorded.
        """
        if self.original_image is None:
            return {"created": 0, "filtered": 0}

        if self.text_detector is None:
            self.text_detector = TextDetector(device=self.canvas.sam.device)

        cv_image = qimage_to_cv2(self.original_image)
        if cv_image.ndim == 2:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        elif cv_image.shape[2] == 4:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
        else:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        detections = self.text_detector.detect(rgb_image, prompt)
        if not detections:
            return {"created": 0, "filtered": 0}

        return self.canvas.segmentTextDetections(
            detections,
            prompt=prompt,
            image_rgb=rgb_image,
            detector=self.text_detector,
        )

    def _handleTextDetectMismatch(self, result, mismatch):
        requested = ", ".join(mismatch.get("requested") or [result.get("prompt", "")])
        suggested = mismatch.get("suggested", "")
        detail = tr(
            'The image does not look like "{requested}". The detector is much more '
            'confident about "{suggested}" ({count} region(s), score {score:.2f}).',
            requested=requested,
            suggested=suggested,
            count=mismatch.get("suggested_count", 0),
            score=mismatch.get("suggested_score", 0.0),
        )
        advice = tr(
            "The result is likely unreliable. Try another object type, use Region "
            "Detect, or segment manually.\n\nDetect with the current prompt anyway?"
        )

        answer = QMessageBox.warning(
            self,
            tr("Semantic Detection Warning"),
            "%s\n\n%s" % (detail, advice),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._startTextDetect(
                result.get("prompt", ""),
                skip_semantic_check=True,
            )

    def _finishTextDetect(self, waiting_dialog):
        waiting_dialog.close_dialog()
        self.toggleActions(True)

    def _handleTextDetectError(self, message, waiting_dialog):
        waiting_dialog.close_dialog()
        self.toggleActions(True)
        QMessageBox.critical(self, tr("Semantic Detection Error"), message)
        self.status(tr("Semantic detection failed: {message}", message=message))

    # endregion

    # region Auto Mask
    def _onRegionDrawn(self, points):
        self.status(tr(
            'Region drawn ({count} pts). Press Detect on the panel, '
            'right-click to redraw.',
            count=len(points)))

    def _runRegionDetect(self):
        points = list(self.canvas.region_points)
        if len(points) < 3:
            self.status(tr('Draw a region on the image first.'))
            return
        fix_sample_count = (
            None if self._hudRegionAuto.isChecked()
            else self._hudRegionCount.value()
        )
        # Clear the lasso so the next region starts clean; the step/detection
        # below already owns a copy of the points.
        self.canvas.region_points = []
        self.canvas.update()

        if self.batch_mode:
            # Recipe recording stays in the mode: capturing several regions in
            # a row is the normal batch workflow.
            step = {
                "type": "region",
                "points": points,
                "fix_sample_count": fix_sample_count,
            }
            self.recordBatchStep(step)
            self.status(
                tr("Recorded batch step: {step}", step=describe_batch_step(step))
            )
        else:
            self.autoMaskDetect(points, fix_sample_count)
            # Single-shot use: the detection was launched, so leave the mode
            # rather than making the user hunt for an exit.
            self._exitMaskMode()

    def _exitMaskMode(self):
        self.canvas.region_points = []
        self.canvas.update()
        self.setMode(Canvas.NORMAL)
        self._updateEditHud()

    def autoMaskDetect(self, points, fix_sample_count):
        waiting_dialog = WaitingDialog(self, tr("Region detecting..."))
        waiting_dialog.show()
        request_id = self._sam_request_id

        worker = Worker(
            self.canvas.autoPredict, 
            "Auto Detect", 
            points, fix_sample_count
        )

        # 连接信号
        worker.signals.result.connect(
            lambda count, rid=request_id: self._handleAutoDetectResult(rid, count)
        )
        worker.signals.error.connect(lambda msg: self.handle_error(msg, worker, waiting_dialog))
        worker.signals.finished.connect(lambda: waiting_dialog.close_dialog())

        self.sam_threadpool.start(worker)

    def _handleAutoDetectResult(self, request_id, count):
        if request_id == self._sam_request_id:
            self.status(tr("Region detection created {count} object(s).", count=count))

    
        

    # endregion


    # region Color Calibration
                
    def colorChecker(self):
        """颜色校正功能：切换显示原始图像和校正后图像"""
        if not self.filename or self.image.isNull():
            QMessageBox.warning(self, tr("Color Checker"), tr("Please load an image first."))
            self.actions.colorChecker.setChecked(False)  # 重置按钮状态
            return

        if self.batch_mode:
            # Recording only: correction runs per image during batch processing.
            self._toggleBatchFlagStep(
                "color_correction", self.actions.colorChecker.isChecked()
            )
            return

        # 根据按钮状态切换
        if self.actions.colorChecker.isChecked():
            # 按钮被按下，显示校正后的图像
            if self.color_correction_data is None:
                # 第一次执行颜色校正（批量模式和普通模式都使用异步方式）
                self.perform_color_correction_async()
            else:
                # 已有校正数据，直接切换显示
                self.show_corrected_image()
        else:
            # 按钮被取消，显示原始图像。审查编辑打开的是校色成品图时，
            # 原图缺失就退无可退 —— 保持按下态并说明，别"切了但没变"。
            if (self._loadedColorCorrected
                    and self._colorRevertImage is None):
                self.status(tr(
                    'Original image not found — cannot undo colour '
                    'correction on this file.'))
                self.actions.colorChecker.setChecked(True)
                return
            self.show_original_image()
            self._loadedColorCorrected = False

    def _restoreColorCorrection(self, processing, centers):
        """Reapply the saved 4x3 matrix so a reopened file shows the same
        corrected view it was saved with -- a ~100 ms matrix multiply instead
        of rerunning chart detection.

        Sidecars written before matrices were persisted carry none: leave the
        markers drawn and ask for one manual recalibration; the matrix is
        stored from that save on, so it only bites once.
        """
        raw_matrix = processing.get('correction_matrix')
        if raw_matrix is None:
            self.status(tr(
                'This file was saved with colour correction but predates '
                'stored matrices - press Color Calibration once to recompute; '
                'it is stored from then on.'))
            return
        matrix = np.asarray(raw_matrix, dtype=np.float32)
        if matrix.shape not in ((4, 3), (3, 3)):
            self.status(tr('Saved correction matrix is malformed; skipped.'))
            return
        try:
            from pheno.color import apply_matrix
            image_bgr = qimage_to_cv2(self.original_image)
            if image_bgr.shape[-1] == 4:
                image_bgr = image_bgr[:, :, :3]
            image_rgb01 = cv2.cvtColor(
                image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            corrected_rgb = apply_matrix(image_rgb01, matrix)
            corrected_bgr = cv2.cvtColor(
                (corrected_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        except Exception:
            self.status(tr('Could not reapply the saved colour correction.'))
            return
        self.color_correction_data = {
            'corrected_image': corrected_bgr,
            'centers': [tuple(c) for c in (centers or [])],
            'original_image': image_bgr,
            'correction_matrix': matrix,
            'mean_delta_e': None,
            'max_delta_e': None,
            'raw_delta_e': None,
        }
        # setChecked only paints the button; the corrected view is swapped in
        # explicitly (and stays toggleable back to the raw image underneath).
        self.actions.colorChecker.setChecked(True)
        self.show_corrected_image()
        self.status(tr('Colour correction restored from file.'))

    def perform_color_correction_async(self):
        """异步执行颜色校正"""
        # 检查是否有 MCC 模块
        if not HAS_MCC:
            reply = QMessageBox.question(
                self, "MCC Module Not Available",
                "The OpenCV mcc module is not available in your installation.\n\n"
                "Would you like to:\n"
                "• Install opencv-contrib-python for automatic color checker detection, or\n"
                "• Use manual color adjustment instead?\n\n"
                "Click 'Yes' for manual adjustment or 'No' to cancel.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.manual_color_adjustment()
            
            # 重置按钮状态
            self.actions.colorChecker.setChecked(False)
            return
            
        # 显示处理对话框
        waiting_dialog = WaitingDialog(self, tr("Detecting color checker..."))
        waiting_dialog.show()
        
        # 创建 Worker 进行颜色校正，传递图像数据而不是文件路径
        worker = Worker(
            self.perform_color_correction,
            "Color Correction",
            self.original_image  # 传递原始 QImage 对象
        )
        
        # 连接信号 - 确保所有情况下都能关闭等待对话框
        def close_waiting_dialog():
            try:
                waiting_dialog.close_dialog()
            except:
                pass
        
        worker.signals.result.connect(self.handle_color_correction_result)
        worker.signals.error.connect(self.handle_color_correction_error)
        worker.signals.finished.connect(close_waiting_dialog)
        worker.signals.error.connect(lambda error_msg: close_waiting_dialog())  # 错误时也要关闭
        
        self.threadpool.start(worker)
    
    def show_corrected_image(self):
        """显示校正后的图像"""
        if self.color_correction_data:
            corrected_image = self.color_correction_data['corrected_image']
            centers = self.color_correction_data['centers']
            
            # 转换为 QImage 并更新显示
            height, width, channel = corrected_image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(corrected_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
            
            # 更新画布显示并设置颜色校正标记
            self.image = q_image
            self.canvas.loadQImage(q_image, update_sam=False)
            self.canvas.set_color_checker_markers(centers)
            self.paintCanvas()
            
            self.status("Displaying color corrected image with markers")
    
    def show_original_image(self):
        """显示原始图像"""
        if self.original_image:
            # 恢复原始图像
            self.image = self.original_image
            self.canvas.loadQImage(self.original_image, update_sam=False)
            self.canvas.clear_color_checker_markers()  # 清除标记
            self.paintCanvas()
            
            self.status("Displaying original image")
    
    def perform_color_correction(self, qimage):
        """执行颜色校正的后台任务

        实现在 pheno/color.py（无 Qt 依赖），与网页端共用同一套拟合与校验，
        避免两端对同一张图给出不同的 MeanRGB。矩阵为 4x3（3x3 + 偏置），在
        sRGB 编码空间拟合——实测比线性空间和 cv2.ccm 都准，理由见该模块文档。
        """
        from pheno.color import CalibrationError, correct_colors

        # 将 QImage 转换为 OpenCV 格式
        try:
            image = qimage_to_cv2(qimage)
            if image is None:
                raise ValueError("Could not convert QImage to OpenCV format")

            # 确保图像是3通道BGR格式（MCC模块要求）
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # BGRA -> BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                elif image.shape[2] != 3:
                    raise ValueError(f"Unsupported channel count: {image.shape[2]}")
            elif len(image.shape) == 2:  # Grayscale -> BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

        except Exception as e:
            raise ValueError(f"Image conversion failed: {str(e)}")

        try:
            report = correct_colors(image)
        except CalibrationError as e:
            raise RuntimeError(str(e))
        except Exception as e:
            raise RuntimeError(f"Color correction processing failed: {str(e)}")

        return {
            'corrected_image': report['corrected'],
            'centers': [tuple(center) for center in report['centers']],
            'original_image': image,
            'correction_matrix': report['matrix'],
            'mean_delta_e': report['mean_delta_e'],
            'max_delta_e': report['max_delta_e'],
            'raw_delta_e': report['raw_delta_e'],
        }
    
    def handle_color_correction_result(self, result):
        """处理颜色校正结果"""
        try:
            # 存储校正结果
            self.color_correction_data = result
            
            # 显示校正后的图像（因为按钮已经被选中）
            self.show_corrected_image()
            
            # 显示成功消息
            centers = result['centers']
            success_msg = f"Color correction completed! Detected {len(centers)} color patches."
            self.status(success_msg)
            
            # 根据模式显示不同的信息
            if self.batch_mode:
                QMessageBox.information(
                    self, "Color Correction Complete", 
                    f"Color correction completed successfully for the current image!\n"
                    f"Detected {len(centers)} color patches.\n\n"
                    f"The corrected image is now displayed. During batch processing, "
                    f"all images will be automatically color corrected.\n\n"
                    f"You can toggle the Color Calibration button to switch between "
                    f"original and corrected views, or disable it to skip color correction in batch processing."
                )
            else:
                # 普通模式下可以不显示额外弹窗，因为是切换功能
                pass
                
        except Exception as e:
            self.handle_color_correction_error(f"Error processing correction result: {str(e)}")
            # 出错时重置按钮状态
            self.actions.colorChecker.setChecked(False)
    
    def handle_color_correction_error(self, error_msg):
        """处理颜色校正错误"""
        # 只在非批量模式下重置按钮状态
        # 批量模式下，单个图像失败不应影响整体的颜色校正设置
        if not self.batch_mode:
            self.actions.colorChecker.setChecked(False)
        
        if "Color checker not detected" in error_msg:
            if self.batch_mode:
                # 批量模式下，只记录错误，不显示弹窗（避免中断批处理）
                self.status(f"Color correction skipped for current image: Color checker not detected")
            else:
                QMessageBox.warning(
                    self, "Color Checker Detection Failed", 
                    "Could not detect a valid MCC24 color checker in the image.\n\n"
                    "Please ensure:\n"
                    "• The image contains a standard 24-patch color checker\n"
                    "• The color checker is clearly visible and well-lit\n"
                    "• The color checker is not too small or distorted"
                )
        elif "No color checker found" in error_msg:
            if self.batch_mode:
                # 批量模式下，只记录错误，不显示弹窗
                self.status(f"Color correction skipped for current image: No color checker found")
            else:
                QMessageBox.warning(
                    self, "No Color Checker Found", 
                    "No color checker patches were found in the image.\n"
                    "Please make sure the color checker is clearly visible."
                )
        else:
            if self.batch_mode:
                # 批量模式下，只记录错误，不显示弹窗
                self.status(f"Color correction failed for current image: {error_msg}")
            else:
                QMessageBox.critical(
                    self, "Color Correction Error", 
                    f"An error occurred during color correction:\n{error_msg}"
                )
                
        # 更新状态栏消息
        if not self.batch_mode:
            self.status(f"Color correction failed: {error_msg}")
        # 批量模式下的状态消息已在上面的条件分支中设置
    
    def save_corrected_image(self, corrected_image):
        """保存校正后的图像"""
        # 生成默认文件名
        base_name = os.path.splitext(os.path.basename(self.filename))[0]
        default_name = f"{base_name}_color_corrected.jpg"
        default_path = os.path.join(self.currentPath(), default_name)
        
        # 打开保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Color Corrected Image", 
            default_path, 
            "JPEG Files (*.jpg);;PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            try:
                # 保存图像 (不带标记的干净版本)
                write_image_file(file_path, corrected_image)
                self.status(f"Color corrected image saved: {os.path.basename(file_path)}")
                QMessageBox.information(
                    self, "Image Saved", 
                    f"Color corrected image has been saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", 
                    f"Failed to save the corrected image:\n{str(e)}"
                )
                self.status(f"Failed to save corrected image: {str(e)}")
    
    def manual_color_adjustment(self):
        """手动颜色调整（备用方案）"""
        # 重置按钮状态
        self.actions.colorChecker.setChecked(False)
        
        QMessageBox.information(
            self, "Manual Color Adjustment",
            "Manual color adjustment feature will be implemented here.\n\n"
            "For automatic color checker detection, please install:\n"
            "pip install opencv-contrib-python\n\n"
            "This will provide the cv2.mcc module needed for automatic "
            "color checker detection and correction."
        )
    
    def apply_color_correction_to_image(self, qimage):
        """对图像应用颜色校正，返回校正后的图像或原始图像（如果校正失败）"""
        if not HAS_MCC:
            return qimage  # 如果没有MCC模块，返回原始图像
        
        try:
            result = self.perform_color_correction(qimage)
            # 转换校正后的图像回QImage格式
            corrected_image = result['corrected_image']
            height, width, channel = corrected_image.shape
            bytes_per_line = 3 * width
            corrected_qimage = QtGui.QImage(corrected_image.data, width, height, 
                                          bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
            return corrected_qimage
        except Exception as e:
            # 校正失败时返回原始图像
            return qimage
    # endregion


    # region Scale Calibration
    def setScale(self):
        pixmap_exists = True if not self.canvas.pixmap.isNull() else False
        dialog = SetScaleDialog(pixmap_exists=pixmap_exists, parent=self)
        
        # 标记对话框是否处于测量模式
        dialog.measuring_mode = False
        
        # 定义画线回调 - 使用非阻塞方式
        def start_measuring():
            # 标记进入测量模式
            dialog.measuring_mode = True
            # 隐藏对话框
            dialog.hide()
            
            # 设置一次性信号连接
            def on_measure_complete(length):
                # 断开信号
                self.canvas.line_drawn.disconnect(on_measure_complete)
                # 退出测量模式后收起 HUD 提示（mode 已回到 NORMAL）
                self._updateEditHud()
                
                # 如果测量成功，填入长度
                if length and length > 0:
                    dialog.pixel_input.setText(str(round(length, 2)))
                
                # 退出测量模式
                dialog.measuring_mode = False
                # 重新显示对话框并设置为模态
                dialog.show()
                dialog.raise_()  # 确保对话框在最前面
                dialog.activateWindow()  # 激活窗口
            
            # 连接信号
            self.canvas.line_drawn.connect(on_measure_complete)
            
            # 启动测量模式
            self.canvas.start_drawing_measure_line()
        
        dialog.set_draw_callback(start_measuring)

        # 重写对话框的accept方法，处理测量模式下的Apply
        original_accept = dialog.accept
        def custom_accept():
            if dialog.measuring_mode:
                # 如果在测量模式下，不关闭对话框
                return
            
            # 正常的accept流程
            original_accept()
            
            # 处理scale设置
            scale, unit = dialog.get_scale()
            if scale is not None:
                QMessageBox.information(self, tr("Scale Set"),
                                        f"Scale is {scale:.4f} {unit} per pixel.")
                self.canvas.scale_per_pixel = scale
                self.canvas.scale_unit = unit
                # 更新比例尺标签
                self.scaleLabel.setText(f"1 pixel = {scale:.4f} {unit.upper()}")
            else:
                QMessageBox.information(self, tr("Scale Set"), "Using ruler scale; if unavailable, using pixel scale.")
                self.set_ruler_scale()
            
            # 比例变了，表头单位与已算数值都要跟着更新
            self.refreshTraitHeaders()
        
        dialog.accept = custom_accept

        dialog.exec_()
    

    def rulerDetection(self):
        if not self.filename or self.image.isNull():
            QMessageBox.warning(self, tr("Ruler Detection"), tr("Please load an image first."))
            self.actions.rulerDetection.setChecked(False)  # 重置按钮状态
            return

        if self.batch_mode:
            # Recording only: detection runs per image during batch processing.
            self._toggleBatchFlagStep(
                "ruler_detection", self.actions.rulerDetection.isChecked()
            )
            return

        if self.actions.rulerDetection.isChecked():
            if self.ruler_detection_data is None:
                # 第一次执行标尺检测
                self.perform_ruler_detection_async()
            else:
                # 已有检测数据，直接应用
                self.show_ruler_tags(self.ruler_detection_data)
        else:
            self.hide_ruler_tags()
            self._loadedRulerDetection = False

    def perform_ruler_detection_async(self):
        
        # 显示处理对话框
        waiting_dialog = WaitingDialog(self, tr("Detecting the ruler..."))
        waiting_dialog.show()
        
        # 创建 Worker 进行颜色校正，传递图像数据而不是文件路径
        worker = Worker(
            self.perform_ruler_detection,
            "Ruler Detection",
            self.original_image  # 传递原始 QImage 对象
        )
        
        # 连接信号 - 确保所有情况下都能关闭等待对话框
        def close_waiting_dialog():
            try:
                waiting_dialog.close_dialog()
            except:
                pass
        
        worker.signals.result.connect(self.handle_ruler_detection_result)
        worker.signals.error.connect(self.handle_ruler_detection_error)
        worker.signals.finished.connect(close_waiting_dialog)
        worker.signals.error.connect(lambda error_msg: close_waiting_dialog())  # 错误时也要关闭
        
        self.threadpool.start(worker)


    def show_ruler_tags(self, result):
        self.canvas.set_ruler_ticks(result.get('tickpoints', []))

    def hide_ruler_tags(self):
        self.canvas.clear_ruler_ticks()

    def set_ruler_scale(self, result=None):
        # result 默认 None：setScale 的「沿用标尺比例」分支就是这样无参调用的，
        # 而下面的 result is None 分支正是为这种情况写的（原先会 TypeError）。
        if SetScaleDialog.last_mode_index == 1:
            # 如果用户上次选择了自定义比例尺，则不覆盖
            return

        if result is None:
            self.canvas.scale_per_pixel = 1.0  # 默认每像素1单位
            self.canvas.scale_unit = "pixel"
            self.scaleLabel.setText("1 pixel = 1 pixel")
        else:
            self.canvas.scale_per_pixel = result.get('pred_pix_mm', 1.0)  # 每像素毫米数
            self.canvas.scale_unit = "mm"
            # 更新比例尺标签
            self.scaleLabel.setText(f"1 pixel = {result.get('pred_pix_mm', 1.0):.4f} mm")

        # 标尺检测同样改变了量纲，表头必须同步，否则表头写 mm 而数字仍是像素。
        self.refreshTraitHeaders()



    def perform_ruler_detection(self, original_image):
        try:
            image = qimage_to_cv2(original_image)
            if image is None:
                raise ValueError("Could not convert QImage to OpenCV format")
            
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # BGRA -> BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                elif image.shape[2] != 3:
                    raise ValueError(f"Unsupported channel count: {image.shape[2]}")
            elif len(image.shape) == 2:  # Grayscale -> BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
            
        except Exception as e:
            raise ValueError(f"Image conversion failed: {str(e)}")
        
        try:
            tickpoints, pred_cm_pix = self.canvas.ruler_net.get_tickpoints_and_scale(image)

        except Exception as e:
            raise ValueError(f"Ruler detection failed: {str(e)}")

        if tickpoints is None or not pred_cm_pix:
            raise ValueError("No usable ruler scale was detected.")

        # 诊断而非修正：这批照片沿 40 cm 尺子的比例渐变中位数约 4.5%，
        # 说明相机不完全正对（已排除径向畸变，见 pheno/RulerNet.py 注释）。
        # 只报告，不改动任何数值。
        uniformity = None
        try:
            uniformity = self.canvas.ruler_net.measure_scale_uniformity(image, tickpoints)
        except Exception:
            uniformity = None

        result = {
            'tickpoints': tickpoints,
            'pred_pix_mm': 1 / (pred_cm_pix / 10.0)  # 转换为每像素毫米数
        }
        if uniformity:
            result['scale_drift'] = float(uniformity['drift'])
        return result

    def handle_ruler_detection_result(self, result):
        try:
            self.ruler_detection_data = result
            self.show_ruler_tags(result)
            self.set_ruler_scale(result)
            if self.batch_mode:
                QMessageBox.information(self, "Ruler Detection Complete", 
                                        "Ruler detection completed successfully for the current image!\n\n"
                                        f"Detected scale: {result['pred_pix_mm']:.4f} pixels per millimeter.\n\n"
                                        "During batch processing, all images will be automatically processed for ruler detection.\n\n"
                                        "You can toggle the Ruler Detection button to show/hide ruler tags, "
                                        "and disable it to skip ruler detection in batch processing.")
            else:
                pass
        except Exception as e:
            self.handle_ruler_detection_error(f"Error processing ruler detection result: {str(e)}")
            self.actions.rulerDetection.setChecked(False)

    def handle_ruler_detection_error(self, error_msg):
        if not self.batch_mode:
            self.actions.rulerDetection.setChecked(False)

        if self.batch_mode:
            self.status(f"Ruler detection skipped for current image: {error_msg}")
        else:
            QMessageBox.warning(self, tr("Ruler Detection Error"), error_msg)
            self.status(f"Ruler detection failed: {error_msg}")

    


    # endregion
