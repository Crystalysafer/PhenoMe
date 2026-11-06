"""
Simple log display for Bean Phenomics application startup
Shows CUDA initialization logs in real-time
"""

import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QLabel, QDialog, QComboBox)
from PyQt5.QtGui import QFont, QPixmap
import time
import os

from pheno.i18n import tr, set_language, get_language

softwareName = "PhenoMe"

class InitializationWorker(QThread):
    """Worker thread for initialization tasks with log capture"""
    
    # Signals for log updates
    log_update = pyqtSignal(str)
    initialization_complete = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.log_buffer = io.StringIO()
        
    def run(self):
        """Run initialization tasks and capture all output"""
        try:
            # Capture stdout and stderr
            with redirect_stdout(self.log_buffer), redirect_stderr(self.log_buffer):
                
                # Step 0: honour the startup device preference before any CUDA
                # detection runs. 'auto' leaves the environment untouched so a
                # hand-set PHENOME_FORCE_CPU still wins.
                if QtCore.QSettings('pheno', 'pheno').value('model/device', 'auto') == 'cpu':
                    os.environ['PHENOME_FORCE_CPU'] = '1'
                    self.emit_log("CPU-only mode selected in startup preferences")

                # Step 1: Setup CUDA environment
                self.emit_log(f"=== {softwareName} Initialization ===")
                self.emit_log("Starting CUDA environment setup...")

                from pheno.cuda import setup_cuda_environment, setup_pytorch_optimizations
                setup_cuda_environment()
                time.sleep(0.2)
                
                self.emit_log("Setting up PyTorch optimizations...")
                setup_pytorch_optimizations()
                time.sleep(0.2)
                
                # Step 2: Initialize CUDA environment
                self.emit_log("Initializing CUDA environment...")
                from pheno.cuda import initialize_cuda_environment
                cuda_info = initialize_cuda_environment()
                time.sleep(0.2)
                
                # Step 3: Import PyTorch
                self.emit_log("Loading PyTorch libraries...")
                import torch
                time.sleep(0.2)
                
                # Step 4: Final status
                if cuda_info.get('cuda_available', False):
                    device_name = cuda_info.get('device_name', 'Unknown GPU')
                    self.emit_log(f"✓ GPU acceleration ready: {device_name}")
                else:
                    self.emit_log("✓ CPU mode ready")
                
                self.emit_log("=== Initialization Complete ===")
                self.emit_log("Starting main application...")
                
                time.sleep(1)
                
                # Emit completion
                self.initialization_complete.emit(cuda_info)
                
        except Exception as e:
            error_msg = f"✗ Initialization failed: {str(e)}"
            self.emit_log(error_msg)
            self.initialization_complete.emit({'error': str(e)})
    
    def emit_log(self, message):
        """Emit log message"""
        self.log_update.emit(message)
        # Also capture any buffered output
        buffer_content = self.log_buffer.getvalue()
        if buffer_content.strip():
            for line in buffer_content.strip().split('\n'):
                if line.strip():
                    self.log_update.emit(line)
            # Clear buffer
            self.log_buffer.truncate(0)
            self.log_buffer.seek(0)


class LogDialog(QDialog):
    """Simple log display dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{softwareName} - Starting Up")
        self.setFixedSize(900, 600)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowStaysOnTopHint)

        try:
            self.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), 'icons', 'icon.png')))
        except Exception as e:
            print(f"Error loading icon: {e}")
        
        # Center on screen
        self.center_on_screen()
        
        # Persisted prefs (model/device/language) -- created before setup_ui
        # uses them.
        self.settings = QtCore.QSettings('pheno', 'pheno')
        set_language(self.settings.value('ui/language', 'en'))

        # Countdown state for the GRUB-style auto-advance after init.
        self._countdown_timer = None
        self._countdown_left = 0
        self._status_state = 'init'

        # Setup UI
        self.setup_ui()

        # Initialize worker
        self.worker = InitializationWorker()
        self.worker.log_update.connect(self.append_log)
        self.worker.initialization_complete.connect(self.on_complete)

        # Store cuda info
        self.cuda_info = {}
        
    def center_on_screen(self):
        """Center dialog on screen"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            int((screen.width() - size.width()) / 2),
            int((screen.height() - size.height()) / 2)
        )
        
    def setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header section with logo and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(20)
        
        # Logo
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), 'icons', 'icon.png')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale logo to appropriate size
            scaled_pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        else:
            # Fallback if logo not found
            logo_label.setText("🔬")
            logo_label.setStyleSheet("font-size: 48px;")
        logo_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(logo_label)
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)
        
        title = QLabel(softwareName)
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #1976D2;
                margin: 0;
            }
        """)
        title_layout.addWidget(title)
        
        self.subtitle = QLabel(tr("Phenotyping Analysis Platform"))
        self.subtitle.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #757575;
                margin: 0;
            }
        """)
        title_layout.addWidget(self.subtitle)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()  # Push content to left

        # Language switcher, top-right corner. Writes the same 'ui/language'
        # key the main window reads, so the choice carries over.
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("中文", "zh_CN")
        idx = self.lang_combo.findData(
            'zh_CN' if get_language() == 'zh_CN' else 'en')
        self.lang_combo.setCurrentIndex(max(idx, 0))
        self.lang_combo.setFixedWidth(90)
        self.lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        header_layout.addWidget(self.lang_combo)

        layout.addLayout(header_layout)
        
        # Separator line
        separator = QLabel()
        separator.setStyleSheet("""
            QLabel {
                background-color: #E0E0E0;
                max-height: 1px;
                margin: 10px 0;
            }
        """)
        layout.addWidget(separator)

        # Startup preferences: segmentation backend + compute device. SAM is
        # only constructed when the main window loads, so both choices apply
        # to this run; _save_prefs also exports PHENOME_FORCE_CPU so the
        # device change does not have to wait for the next launch.
        from pheno.SAM import (
            SEGMENTER_SETTING, available_segmenters, current_segmenter_key,
            segmenter_label)

        prefs_row = QHBoxLayout()
        prefs_row.setSpacing(8)

        self.model_label = QLabel(tr("Segmentation model:"))
        prefs_row.addWidget(self.model_label)
        self.model_combo = QComboBox()
        for key in available_segmenters():
            self.model_combo.addItem(segmenter_label(key), key)
        idx = self.model_combo.findData(current_segmenter_key())
        self.model_combo.setCurrentIndex(max(idx, 0))
        self.model_combo.currentIndexChanged.connect(self._save_prefs)
        prefs_row.addWidget(self.model_combo, 1)

        prefs_row.addSpacing(16)
        self.compute_label = QLabel(tr("Compute:"))
        prefs_row.addWidget(self.compute_label)
        self.device_combo = QComboBox()
        self.device_combo.addItem(tr("Auto (GPU when available)"), "auto")
        self.device_combo.addItem(tr("CPU only"), "cpu")
        idx = self.device_combo.findData(
            str(self.settings.value('model/device', 'auto')))
        self.device_combo.setCurrentIndex(max(idx, 0))
        self.device_combo.currentIndexChanged.connect(self._save_prefs)
        prefs_row.addWidget(self.device_combo, 1)

        prefs_row.addStretch()
        layout.addLayout(prefs_row)

        self.note_label = QLabel(
            tr("Model and device choices apply when the main window loads."))
        self.note_label.setStyleSheet("color: #757575; font-size: 11px;")
        layout.addWidget(self.note_label)

        # Log display
        self.log_text = QTextEdit()
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #FAFAFA;
                color: #333333;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 12px;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 15px;
                line-height: 1.4;
            }
        """)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        # Status
        self.status_label = QLabel(tr("Initializing system components..."))
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #1976D2;
                font-weight: 500;
                padding: 10px;
                background-color: #F3F8FF;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.status_label)
        
    def _save_prefs(self):
        from pheno.SAM import SEGMENTER_SETTING
        self.settings.setValue(SEGMENTER_SETTING, self.model_combo.currentData())
        device = self.device_combo.currentData()
        self.settings.setValue('model/device', device)
        # The init worker already read the preference before the dialog was
        # interactive, so apply the choice ourselves: SAM loads after this
        # dialog closes and consults PHENOME_FORCE_CPU / the persisted pref.
        if device == 'cpu':
            os.environ['PHENOME_FORCE_CPU'] = '1'
        else:
            os.environ.pop('PHENOME_FORCE_CPU', None)
        self._restart_countdown()

    def _on_lang_changed(self):
        lang = self.lang_combo.currentData()
        self.settings.setValue('ui/language', lang)
        set_language(lang)
        self.retranslate()
        self._restart_countdown()

    COUNTDOWN_SECONDS = 5

    def _popup_open(self):
        """True while any combo's dropdown is open -- never tick it down."""
        return any(c.view().isVisible() for c in
                   (self.model_combo, self.device_combo, self.lang_combo))

    def _update_countdown_text(self):
        self.status_label.setText(tr(
            "Starting in {n}s — press any key to start, "
            "or change an option to keep this window open",
            n=self._countdown_left))

    def keyPressEvent(self, event):
        """Any key during the countdown enters the app immediately."""
        if (self._countdown_timer and self._countdown_timer.isActive()
                and not event.isAutoRepeat()):
            self._countdown_timer.stop()
            self.accept()
            return
        super().keyPressEvent(event)

    def _restart_countdown(self):
        """Any option change resets the auto-advance timer (GRUB-style)."""
        if self._countdown_timer and self._countdown_timer.isActive():
            self._countdown_left = self.COUNTDOWN_SECONDS
            self._update_countdown_text()

    def _countdown_tick(self):
        if self._popup_open():
            return
        self._countdown_left -= 1
        if self._countdown_left <= 0:
            self._countdown_timer.stop()
            self.accept()
            return
        self._update_countdown_text()

    def retranslate(self):
        self.subtitle.setText(tr("Phenotyping Analysis Platform"))
        self.model_label.setText(tr("Segmentation model:"))
        self.compute_label.setText(tr("Compute:"))
        self.device_combo.setItemText(0, tr("Auto (GPU when available)"))
        self.device_combo.setItemText(1, tr("CPU only"))
        self.note_label.setText(tr(
            "Model and device choices apply when the main window loads."))
        if self._countdown_timer and self._countdown_timer.isActive():
            self._update_countdown_text()
        elif self._status_state == 'init':
            self.status_label.setText(tr("Initializing system components..."))
        elif self._status_state == 'error':
            self.status_label.setText(
                "❌ " + tr("Initialization failed - Please check the log above"))

    def append_log(self, message):
        """Append message to log"""
        # Add timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Color coding with more subtle colors
        if "✓" in message or "completed" in message.lower():
            color = "#2E7D32"  # Green
        elif "⚠" in message or "warning" in message.lower():
            color = "#F57C00"  # Orange  
        elif "✗" in message or "error" in message.lower() or "failed" in message.lower():
            color = "#D32F2F"  # Red
        elif "===" in message:
            color = "#1976D2"  # Blue for headers
        else:
            color = "#424242"  # Dark gray
            
        # Append with color
        self.log_text.setTextColor(QtGui.QColor(color))
        self.log_text.append(formatted_message)
        
        # Auto scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def on_complete(self, cuda_info):
        """Handle initialization completion"""
        self.cuda_info = cuda_info
        
        if 'error' in cuda_info:
            self._status_state = 'error'
            self.status_label.setText(
                "❌ " + tr("Initialization failed - Please check the log above"))
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #D32F2F;
                    background-color: #FFEBEE;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 6px;
                }
            """)
            # Keep dialog open longer for error review
            QTimer.singleShot(5000, self.accept)
        else:
            self._status_state = 'ok'
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #2E7D32;
                    background-color: #E8F5E8;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 6px;
                }
            """)
            # GRUB-style countdown: auto-advance after a few seconds, any
            # option change resets the timer, an open dropdown pauses it.
            self._countdown_left = self.COUNTDOWN_SECONDS
            self._countdown_timer = QTimer(self)
            self._countdown_timer.timeout.connect(self._countdown_tick)
            self._countdown_timer.start(1000)
            self._update_countdown_text()
        
    def start_initialization(self):
        """Start the initialization process"""
        self.worker.start()
        
    def get_cuda_info(self):
        """Get the cuda initialization results"""
        return self.cuda_info


def show_startup_dialog():
    """Show startup dialog with log display"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    dialog = LogDialog()
    dialog.start_initialization()
    
    # Show dialog and wait for completion
    result = dialog.exec_()
    
    return dialog.get_cuda_info()


if __name__ == "__main__":
    # Test the startup dialog
    app = QApplication(sys.argv)
    cuda_info = show_startup_dialog()
    print("CUDA Info:", cuda_info)
