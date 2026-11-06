import os
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from PyQt5 import QtCore, QtWidgets

from pheno import __appname__
from pheno.lib import newIcon
from pheno.app import MainWindow
from pheno.splash import show_startup_dialog


def _install_crash_logging():
    """Keep a trace of fatal errors even in the windowed (console=False) build.

    Writes a Python-level faulthandler dump plus an excepthook traceback to
    %LOCALAPPDATA%/PhenoMe/logs/, so a crash that shows nothing in the console
    still leaves something a user can send back. Returns a report() helper so
    every failure path -- startup errors included -- produces exactly ONE
    dialog: the first error pops it, later exceptions only append to the log.
    """
    try:
        import faulthandler
        import traceback

        base = os.getenv('LOCALAPPDATA') or str(Path.home())
        log_dir = Path(base) / 'PhenoMe' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        crash_log = log_dir / f"crash-{datetime.now():%Y%m%d-%H%M%S}.log"
        fh = open(crash_log, 'a', encoding='utf-8')
        faulthandler.enable(fh)
        dialog_shown = [False]

        def report(exc_type, exc, tb, title="PhenoMe", prefix=None):
            try:
                traceback.print_exception(exc_type, exc, tb, file=fh)
                fh.flush()
            except Exception:
                pass
            if dialog_shown[0]:
                return crash_log
            dialog_shown[0] = True
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror(
                    title,
                    f"{prefix or 'PhenoMe encountered an unexpected error:'}\n{exc}\n\n"
                    f"A crash report was written to:\n{crash_log}",
                )
                root.destroy()
            except Exception:
                pass
            return crash_log

        def excepthook(exc_type, exc, tb):
            # Ctrl+C and interpreter shutdown are user actions, not crashes.
            if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
                sys.__excepthook__(exc_type, exc, tb)
                return
            report(exc_type, exc, tb)
            sys.__excepthook__(exc_type, exc, tb)

        sys.excepthook = excepthook
        return report
    except Exception:
        return None


def main():
    """
    主函数 - 启动应用程序
    """
    try:
        report = _install_crash_logging()

        # 显示启动对话框并初始化CUDA环境
        # 这会处理所有的CUDA环境设置、PyTorch加载等初始化工作
        cuda_info = show_startup_dialog()
        
        # 检查初始化是否成功
        if 'error' in cuda_info:
            raise Exception(cuda_info['error'])
        
        # 创建PyQt应用程序（如果还没有创建）
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        
        app.setApplicationName(__appname__)
        app.setWindowIcon(newIcon('icon'))

        # 在构建窗口之前应用主题，避免先用旧样式绘制一遍再重绘。
        # 偏好与语言一样存在 QSettings 里；桌面端与网页端共用同一套配色
        # （见 pheno/theme.py）。
        from pheno.theme import apply_theme

        saved_theme = QtCore.QSettings('pheno', 'pheno').value('ui/theme', 'classic')
        apply_theme(app, str(saved_theme or 'classic'))

        # 创建主窗口
        win = MainWindow()
        
        # 将CUDA信息传递给主窗口（如果需要）
        if hasattr(win, 'set_cuda_info'):
            win.set_cuda_info(cuda_info)
        
        win.show()
        win.raise_()
        
        # 启动事件循环
        sys.exit(app.exec_())
        
    except Exception as e:
        # 处理启动时的异常 -- 走同一个崩溃报告入口，保证始终只有一个弹窗
        if report is not None:
            report(type(e), e, e.__traceback__, title="启动错误",
                   prefix="应用程序启动失败:")
        else:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "启动错误",
                f"应用程序启动失败:\n{str(e)}\n\n请检查系统环境和依赖是否正确安装。"
            )
            root.destroy()
        sys.exit(1)


if __name__ == '__main__':
    main()