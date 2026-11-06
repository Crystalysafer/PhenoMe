"""
CUDA Environment Management and Optimization Module
Handles GPU environment detection, memory management, and compatibility
"""

import os
import re
import sys
import gc
import warnings
import torch
from typing import Dict, Optional, Tuple

try:
    from packaging import version
    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False


# Set by disable_cuda_for_session() once a hard CUDA failure has been seen;
# every subsequent device decision then goes straight to CPU.
_cuda_disabled_reason: Optional[str] = None


# =============================================================================
# Environment Setup
# =============================================================================

def force_cpu_requested() -> bool:
    """Whether the session should run on CPU despite a working GPU.

    Two sources, either wins:
    - PHENOME_FORCE_CPU env var (scripts, or the splash's step-0 hook)
    - the persisted 'model/device' startup preference -- honoured only while a
      QApplication exists so headless batch scripts are not silently pinned to
      CPU by a stale GUI preference.
    """
    env = os.getenv('PHENOME_FORCE_CPU', '').lower()
    if env in ('1', 'true', 'yes'):
        return True
    if env in ('0', 'false', 'no'):
        return False  # explicit opt-out (e.g. a benchmark script) beats the pref
    try:
        from qtpy import QtCore, QtWidgets
    except ImportError:
        from PyQt5 import QtCore, QtWidgets
    try:
        if QtWidgets.QApplication.instance() is None:
            return False
        return str(QtCore.QSettings('pheno', 'pheno').value(
            'model/device', 'auto')) == 'cpu'
    except Exception:
        return False


def setup_cuda_environment():
    """Set up CUDA environment variables"""
    
    print("Setting up CUDA environment variables...")
    
    # expandable_segments is Linux-only; leaving it on Windows just emits a
    # UserWarning from every .to(device) call and is silently ignored.
    alloc_conf = 'roundup_power2_divisions:8'
    if not sys.platform.startswith('win'):
        alloc_conf = 'expandable_segments:True,' + alloc_conf

    # Critical CUDA environment variables
    cuda_env = {
        # PyTorch CUDA memory allocation configuration.
        # NOTE: PYTORCH_NO_CUDA_MEMORY_CACHING is deliberately absent. PyTorch
        # only checks whether that variable EXISTS, not its value, so setting it
        # to '0' disables the caching allocator instead of enabling it -- which
        # both slows allocation down and makes memory_allocated() report 0.
        'PYTORCH_CUDA_ALLOC_CONF': alloc_conf,
        
        # CUDA runtime configuration
        'CUDA_LAUNCH_BLOCKING': '0',       # Async execution for better performance
        'CUDA_CACHE_DISABLE': '0',         # Enable CUDA caching
        'CUDA_VISIBLE_DEVICES': '0',       # Use only the first GPU
        
        # PyTorch specific settings
        'TORCH_CUDNN_V8_API_ENABLED': '1', # Enable cuDNN v8 API
    }
    
    # Only set environment variables that haven't been set
    for key, value in cuda_env.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"Set {key}={value}")
        else:
            print(f"Already exists {key}={os.environ[key]}")
    
    print("[OK] CUDA environment variables setup completed")


def setup_pytorch_optimizations():
    """Set up PyTorch optimizations"""
    
    # These settings are executed after importing torch
    optimization_flags = {
        'OMP_NUM_THREADS': '4',            # OpenMP thread count
        'MKL_NUM_THREADS': '4',            # Intel MKL thread count
        'NUMEXPR_NUM_THREADS': '4',        # NumExpr thread count
        'OPENBLAS_NUM_THREADS': '4',       # OpenBLAS thread count
    }
    
    for key, value in optimization_flags.items():
        if key not in os.environ:
            os.environ[key] = value
    
    print("[OK] PyTorch optimization settings completed")


# =============================================================================
# Memory Management
# =============================================================================

class CUDAMemoryManager:
    """CUDA Memory Manager"""
    
    def __init__(self):
        self.original_settings = {}
        self.is_initialized = False
        
    def optimize_for_packaged_app(self):
        """Optimize CUDA settings for packaged application"""
        
        if self.is_initialized:
            return
            
        print("Optimizing CUDA memory settings...")
        
        # 1. Set PyTorch CUDA memory management environment variables.
        # Same platform caveat as setup_cuda_environment(); in practice these
        # are already set by the time this runs, so nothing is overwritten.
        alloc_conf = 'max_split_size_mb:128'
        if not sys.platform.startswith('win'):
            alloc_conf = 'expandable_segments:True,' + alloc_conf
        cuda_env_vars = {
            'PYTORCH_CUDA_ALLOC_CONF': alloc_conf,
            'CUDA_LAUNCH_BLOCKING': '0',  # Async execution
            'CUDA_CACHE_DISABLE': '0',    # Enable caching
            'CUDA_VISIBLE_DEVICES': '0',  # Use only the first GPU
        }
        
        for key, value in cuda_env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
                print(f"Set {key}={value}")
        
        # 2. Configure PyTorch memory strategy
        if torch.cuda.is_available():
            try:
                # Enable memory pool
                torch.cuda.empty_cache()
                
                # Set memory allocation strategy
                torch.backends.cudnn.benchmark = True  # Optimize cudnn
                torch.backends.cudnn.enabled = True
                
                # Set memory growth strategy
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    # For 6GB GPU, set to use 85% memory (~5GB)
                    torch.cuda.set_per_process_memory_fraction(0.85)
                    print("Set GPU memory limit to 85%")
                
                print("[OK] CUDA memory optimization completed")
                
            except Exception as e:
                print(f"[WARN] CUDA optimization partially failed: {e}")
        
        self.is_initialized = True
    
    def get_memory_info(self):
        """Get GPU memory information.

        ``free``/``used`` come from the driver (``mem_get_info``) so they include
        the CUDA context and anything allocated outside PyTorch's allocator --
        that is what actually determines whether the next allocation fits.
        ``allocated``/``reserved`` are PyTorch's own tensor and pool totals and
        are always <= the driver figure.
        """
        # Reading status never launches kernels, so keep reporting even after
        # disable_cuda_for_session() -- the user still wants to see what the
        # stale context is holding.
        if not torch.cuda.is_available():
            return None

        try:
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(device) / 1024**3     # GB
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            total = total_bytes / 1024**3
            free = free_bytes / 1024**3
            used = total - free
            
            return {
                'device': device,
                'total_gb': round(total, 2),
                'allocated_gb': round(allocated, 2),
                'cached_gb': round(cached, 2),
                'used_gb': round(used, 2),
                'free_gb': round(free, 2),
                'usage_percent': round((used / total) * 100, 1) if total else 0.0,
            }
        except Exception as e:
            print(f"Failed to get memory info: {e}")
            return None
    
    def aggressive_cleanup(self):
        """Aggressive memory cleanup"""
        if torch.cuda.is_available():
            try:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Synchronize CUDA
                torch.cuda.synchronize()
                
                print("[OK] Performed aggressive memory cleanup")
                
            except Exception as e:
                print(f"Memory cleanup failed: {e}")
    
    def check_memory_before_operation(self, required_gb=1.0):
        """Check memory before operation"""
        if _cuda_disabled_reason is not None:
            return True  # CPU mode: nothing to check
        memory_info = self.get_memory_info()
        if memory_info is None:
            return True  # CPU mode
        
        if memory_info['free_gb'] < required_gb:
            print(f"[WARN] Insufficient GPU memory: need {required_gb}GB, available {memory_info['free_gb']}GB")
            
            # Try to clean up memory
            self.aggressive_cleanup()
            
            # Re-check
            memory_info = self.get_memory_info()
            if memory_info['free_gb'] < required_gb:
                print(f"[WARN] Still insufficient after cleanup: available {memory_info['free_gb']}GB")
                return False
        
        return True
    
    def fallback_to_cpu_if_needed(self):
        """Fall back to CPU mode if necessary"""
        if _cuda_disabled_reason is not None:
            return 'cpu'
        if force_cpu_requested():
            return 'cpu'
        ok, why = cuda_supported()
        if not ok:
            print(f"[WARN] {why} -- running on CPU")
            return 'cpu'
        memory_info = self.get_memory_info()
        if memory_info is None:
            return 'cpu'
        
        # If available memory is less than 500MB, suggest CPU mode
        if memory_info['free_gb'] < 0.5:
            print("[WARN] GPU memory critically low, recommend CPU mode")
            return 'cpu'
        
        return 'cuda:0'


def _parse_arch(entry: str) -> Optional[Tuple[Tuple[int, int], str]]:
    """Parse a torch arch entry like 'sm_120', 'sm_90a' or 'compute_90'.

    Returns ((major, minor), kind) where kind is 'sm' (compiled binary) or
    'compute' (PTX, JIT-compiled by the driver).
    """
    m = re.match(r"(sm|compute)_(\d+)([a-zA-Z]?)$", entry)
    if not m:
        return None
    digits = m.group(2)
    return (int(digits[:-1]), int(digits[-1])), m.group(1)


def _arch_supported(major: int, minor: int) -> bool:
    """Whether this PyTorch build can actually launch kernels on the GPU.

    torch.cuda.is_available() only proves the driver is there; a wheel built
    for e.g. sm_50–sm_90 (cu124) still reports CUDA as available on an
    sm_120 Blackwell card and then dies on the first kernel launch. A device
    is usable when the build ships a binary for the same major and equal or
    lower minor, or a PTX blob old enough for the driver to JIT-compile.
    """
    try:
        archs = torch.cuda.get_arch_list()
    except Exception:
        return True  # cannot determine; let torch try and fall back on failure
    for entry in archs:
        parsed = _parse_arch(entry)
        if parsed is None:
            continue
        (maj, minor_), kind = parsed
        if kind == "sm" and maj == major and minor_ <= minor:
            return True
        if kind == "compute" and (maj, minor_) <= (major, minor):
            return True
    return False


def cuda_supported() -> Tuple[bool, str]:
    """Whether CUDA inference can actually run; returns (ok, reason)."""
    if _cuda_disabled_reason is not None:
        return False, _cuda_disabled_reason
    try:
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False"
        major, minor = torch.cuda.get_device_capability(0)
        if _arch_supported(major, minor):
            return True, ""
        archs = torch.cuda.get_arch_list()
        return False, (
            f"GPU compute capability sm_{major}{minor} is not covered by this "
            f"PyTorch build {torch.__version__} (compiled for: "
            f"{', '.join(archs)})"
        )
    except Exception as exc:
        return False, f"CUDA probe failed: {exc}"


def disable_cuda_for_session(reason):
    """Blacklist CUDA for the rest of this process after a hard failure."""
    global _cuda_disabled_reason
    if _cuda_disabled_reason is None:
        _cuda_disabled_reason = str(reason)
        print(f"[WARN] CUDA disabled for this session: {_cuda_disabled_reason}")


# Global memory manager instance
memory_manager = CUDAMemoryManager()


def optimize_cuda_for_exe():
    """Optimize CUDA for EXE environment (main entry point)"""
    memory_manager.optimize_for_packaged_app()
    return memory_manager


def get_optimal_device():
    """Get optimal device"""
    return memory_manager.fallback_to_cpu_if_needed()


def check_memory(required_gb=1.0):
    """Check if memory is sufficient"""
    return memory_manager.check_memory_before_operation(required_gb)


def cleanup_gpu_memory():
    """Clean up GPU memory"""
    memory_manager.aggressive_cleanup()


def print_memory_status():
    """Print memory status"""
    info = memory_manager.get_memory_info()
    if info:
        print(f"GPU Memory Status: {info['used_gb']:.2f}GB/{info['total_gb']:.2f}GB "
              f"({info['usage_percent']:.1f}%), Free: {info['free_gb']:.2f}GB "
              f"[torch tensors {info['allocated_gb']:.2f}GB, pool {info['cached_gb']:.2f}GB]")
    else:
        print("GPU memory info unavailable")


# =============================================================================
# CUDA Compatibility Manager (Advanced)
# =============================================================================

class CUDACompatibilityManager:
    """CUDA Compatibility Manager"""
    
    def __init__(self):
        self.cuda_info = {}
        self.config = self._load_config()
        self._setup_environment()
    
    def _load_config(self) -> Dict:
        """Load CUDA configuration"""
        config = {
            'show_status': os.getenv('SHOW_CUDA_STATUS', 'false').lower() == 'true',
            'strict_gpu': os.getenv('STRICT_GPU_MODE', 'false').lower() == 'true',
            'memory_fraction': float(os.getenv('GPU_MEMORY_FRACTION', '0.8')),
            'verbose_log': os.getenv('VERBOSE_CUDA_LOG', 'false').lower() == 'true',
            'test_on_startup': os.getenv('TEST_CUDA_ON_STARTUP', 'true').lower() == 'true',
            'auto_fallback': os.getenv('AUTO_FALLBACK', 'cpu'),
            'min_cuda_version': os.getenv('MIN_CUDA_VERSION', '11.0'),
            'cudnn_benchmark': os.getenv('CUDNN_BENCHMARK', 'true').lower() == 'true',
        }
        return config
    
    def _setup_environment(self):
        """Set up environment variables"""
        # CUDA environment variables
        os.environ['CUDA_LAUNCH_BLOCKING'] = os.getenv('CUDA_LAUNCH_BLOCKING', '0')
        os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
        
        # Ignore compatibility warnings
        if not self.config['verbose_log']:
            warnings.filterwarnings('ignore', category=UserWarning, module='torch')
            warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
    
    def detect_cuda_environment(self) -> Dict:
        """Detect CUDA environment"""
        cuda_info = {
            'available': False,
            'version': None,
            'runtime_version': None,
            'device_count': 0,
            'devices': [],
            'driver_version': None,
            'cudnn_version': None,
            'compute_capability': None,
            'memory_total': 0,
            'memory_available': 0,
            'compatible': False,
            'fallback_reason': None
        }
        
        try:
            # Basic CUDA check
            if torch.cuda.is_available():
                cuda_info['available'] = True
                cuda_info['device_count'] = torch.cuda.device_count()
                
                # Get CUDA version info
                cuda_info['version'] = torch.version.cuda
                
                # Get device information
                for i in range(cuda_info['device_count']):
                    device_props = torch.cuda.get_device_properties(i)
                    device_info = {
                        'name': device_props.name,
                        'compute_capability': f"{device_props.major}.{device_props.minor}",
                        'memory_total': device_props.total_memory,
                        'multi_processor_count': device_props.multi_processor_count
                    }
                    cuda_info['devices'].append(device_info)
                
                # Use first device info as main info
                if cuda_info['devices']:
                    main_device = cuda_info['devices'][0]
                    cuda_info['compute_capability'] = main_device['compute_capability']
                    cuda_info['memory_total'] = main_device['memory_total']
                
                # Check compatibility
                cuda_info['compatible'] = self._check_compatibility(cuda_info)
                
                if self.config['verbose_log']:
                    print(f"CUDA version: {cuda_info['version']}")
                    print(f"Device count: {cuda_info['device_count']}")
                    for i, device in enumerate(cuda_info['devices']):
                        print(f"Device {i}: {device['name']}")
                
            else:
                cuda_info['fallback_reason'] = "CUDA unavailable"
                if self.config['verbose_log']:
                    print("[WARN] CUDA unavailable")
                    
        except Exception as e:
            cuda_info['fallback_reason'] = f"CUDA detection error: {str(e)}"
            if self.config['verbose_log']:
                print(f"[WARN] CUDA detection error: {e}")
        
        self.cuda_info = cuda_info
        return cuda_info
    
    def _check_compatibility(self, cuda_info: Dict) -> bool:
        """Check CUDA compatibility"""
        try:
            # Check CUDA version
            if cuda_info['version'] and PACKAGING_AVAILABLE:
                cuda_version = version.parse(cuda_info['version'])
                min_version = version.parse(self.config['min_cuda_version'])
                if cuda_version < min_version:
                    cuda_info['fallback_reason'] = f"CUDA version too old: {cuda_info['version']} < {self.config['min_cuda_version']}"
                    return False
            
            # Check compute capability
            if cuda_info['compute_capability']:
                min_compute = float(os.getenv('MIN_COMPUTE_CAPABILITY', '6.0'))
                current_compute = float(cuda_info['compute_capability'])
                if current_compute < min_compute:
                    cuda_info['fallback_reason'] = f"Compute capability too low: {current_compute} < {min_compute}"
                    return False
                # A GPU newer than this torch build (e.g. sm_120 on cu124)
                # passes the lower-bound check but has no runnable kernels.
                parts = str(cuda_info['compute_capability']).split('.')
                if not _arch_supported(int(parts[0]), int(parts[1])):
                    archs = torch.cuda.get_arch_list()
                    cuda_info['fallback_reason'] = (
                        f"GPU compute capability sm_{parts[0]}{parts[1]} exceeds this "
                        f"PyTorch build {torch.__version__} (compiled for: "
                        f"{', '.join(archs)}) -- will run on CPU")
                    return False
            
            # Test CUDA functionality
            if self.config['test_on_startup']:
                return self._test_cuda_functionality()
            
            return True
            
        except Exception as e:
            cuda_info['fallback_reason'] = f"Compatibility check failed: {str(e)}"
            return False
    
    def _test_cuda_functionality(self) -> bool:
        """Test CUDA functionality"""
        try:
            # Create test tensor
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.mm(test_tensor, test_tensor.t())
            
            # Cleanup memory
            del test_tensor, result
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            self.cuda_info['fallback_reason'] = f"CUDA functionality test failed: {str(e)}"
            return False
    
    def get_recommended_device(self) -> str:
        """Get recommended compute device"""
        if self.cuda_info['available'] and self.cuda_info['compatible']:
            # Select GPU with most memory
            if self.cuda_info['devices']:
                best_device = max(range(len(self.cuda_info['devices'])), 
                                key=lambda i: self.cuda_info['devices'][i]['memory_total'])
                return f"cuda:{best_device}"
        
        return "cpu"
    
    def get_status_message(self) -> str:
        """Get status message"""
        if not self.cuda_info['available']:
            return f"GPU unavailable: {self.cuda_info.get('fallback_reason', 'Unknown reason')}"
        
        if not self.cuda_info['compatible']:
            return f"GPU compatibility issue: {self.cuda_info.get('fallback_reason', 'Unknown reason')}"
        
        device_name = self.cuda_info['devices'][0]['name'] if self.cuda_info['devices'] else 'Unknown device'
        return f"GPU available: {device_name} (CUDA {self.cuda_info['version']})"
    
    def should_use_gpu(self) -> bool:
        """Determine whether to use GPU"""
        return (self.cuda_info['available'] and 
                self.cuda_info['compatible'] and 
                self.cuda_info['device_count'] > 0)


def get_cuda_manager() -> CUDACompatibilityManager:
    """Get CUDA compatibility manager instance"""
    if not hasattr(get_cuda_manager, '_instance'):
        get_cuda_manager._instance = CUDACompatibilityManager()
    return get_cuda_manager._instance


def smart_cuda_check() -> Dict:
    """Smart CUDA check function"""
    manager = get_cuda_manager()
    cuda_info = manager.detect_cuda_environment()
    
    return {
        'use_gpu': manager.should_use_gpu(),
        'device': manager.get_recommended_device(),
        'status': manager.get_status_message(),
        'details': cuda_info,
        'fallback_ok': True
    }


# =============================================================================
# CUDA Environment Check and Initialization
# =============================================================================

def basic_cuda_check():
    """Basic CUDA check (fallback solution)"""
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return {
                'use_gpu': True,
                'device': 'cuda:0',
                'status': f"GPU available: {device_name} (CUDA {cuda_version})",
                'fallback_ok': True
            }
        else:
            return {
                'use_gpu': False,
                'device': 'cpu',
                'status': "GPU unavailable, using CPU mode",
                'fallback_ok': True
            }
    except Exception as e:
        return {
            'use_gpu': False,
            'device': 'cpu',
            'status': f"CUDA check failed: {e}",
            'fallback_ok': True
        }


def check_cuda():
    """
    Intelligent CUDA compatibility check
    Supports different versions of CUDA environments and GPU configurations
    """
    print("Checking CUDA environment...")

    if force_cpu_requested():
        # Skip the GPU probe entirely: it would allocate a CUDA context that
        # stays resident for the whole session even though nothing uses it.
        cuda_result = {
            'use_gpu': False,
            'device': 'cpu',
            'status': "CPU-only mode requested (startup preference)",
            'fallback_ok': True,
        }
    else:
        # Try to import advanced CUDA tools
        try:
            cuda_result = smart_cuda_check()
        except Exception:
            print("[WARN] Advanced CUDA tools unavailable, using basic CUDA check")
            cuda_result = basic_cuda_check()
    
    # Display status information
    print(f"Compute device: {cuda_result['device']}")
    print(f"Status: {cuda_result['status']}")
    
    # Check if user confirmation is needed
    show_dialog = os.getenv('SHOW_CUDA_STATUS', 'false').lower() == 'true'
    strict_mode = os.getenv('STRICT_GPU_MODE', 'false').lower() == 'true'
    
    if show_dialog or (not cuda_result['use_gpu'] and strict_mode):
        _show_cuda_dialog(cuda_result, strict_mode)
    
    # Set global device information (for use by other parts of the application)
    os.environ['TORCH_DEVICE'] = cuda_result['device']
    os.environ['USE_GPU'] = str(cuda_result['use_gpu']).lower()
    
    return cuda_result


def _show_cuda_dialog(cuda_result, strict_mode):
    """Show CUDA status dialog"""
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    
    try:
        if cuda_result['use_gpu']:
            # GPU available
            messagebox.showinfo(
                "GPU Status", 
                f"[OK] {cuda_result['status']}\n\nApplication will run with GPU acceleration"
            )
        else:
            # GPU unavailable
            if strict_mode:
                # Strict mode: GPU required
                messagebox.showerror(
                    "GPU Required", 
                    f"✗ {cuda_result['status']}\n\n"
                    "This application requires GPU support.\nPlease ensure:\n"
                    "• NVIDIA graphics driver is installed\n"
                    "• Compatible CUDA version is installed\n"
                    "• GPU has sufficient available memory"
                )
                root.destroy()
                sys.exit(1)
            else:
                # Non-strict mode: ask if continue
                result = messagebox.askyesno(
                    "GPU Status", 
                    f"[WARN] {cuda_result['status']}\n\n"
                    "Application will run in CPU mode, which will be slower.\n"
                    "Continue?",
                    icon='warning'
                )
                if not result:
                    root.destroy()
                    sys.exit(0)
    finally:
        root.destroy()


def initialize_cuda_environment():
    """
    Initialize CUDA environment
    This is the standard entry point when the application starts
    """
    cuda_result = check_cuda()
    
    # If using GPU, perform memory optimization
    if cuda_result['use_gpu']:
        try:
            print("Optimizing CUDA memory management for packaged application...")
            optimize_cuda_for_exe()
            print_memory_status()
        except Exception as e:
            print(f"[WARN] CUDA memory optimization failed: {e}")
    
    return cuda_result


def get_device_info():
    """Get current device information"""
    device = os.environ.get('TORCH_DEVICE', 'cpu')
    use_gpu = os.environ.get('USE_GPU', 'false').lower() == 'true'
    
    return {
        'device': device,
        'use_gpu': use_gpu
    }


def is_cuda_available():
    """Quick check if CUDA is available"""
    return os.environ.get('USE_GPU', 'false').lower() == 'true'


# =============================================================================
# Module Initialization
# =============================================================================

# Execute environment setup when module is imported (except when run as main)
if __name__ != "__main__":
    try:
        setup_cuda_environment()
        setup_pytorch_optimizations()
    except Exception as e:
        print(f"[WARN] CUDA environment setup failed: {e}")


# =============================================================================
# Test Script
# =============================================================================

if __name__ == "__main__":
    print("CUDA Compatibility Detection Tool")
    print("=" * 50)
    
    # Test environment setup
    setup_cuda_environment()
    setup_pytorch_optimizations()
    
    # Test CUDA detection
    result = initialize_cuda_environment()
    
    print(f"Recommended device: {result['device']}")
    print(f"Status: {result['status']}")
    print(f"Use GPU: {result['use_gpu']}")
    
    # Test memory management
    if result['use_gpu']:
        print("\nTesting memory management...")
        print_memory_status()
        
        # Test memory check
        if check_memory(1.0):
            print("[OK] Memory check passed")
        else:
            print("[WARN] Insufficient memory")
    
    print("\n" + "=" * 50)
    print("Testing completed")
