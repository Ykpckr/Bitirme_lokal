"""
Device Management Module

Handles device selection and configuration for training.
Supports CUDA, DirectML, and CPU with automatic fallback.
"""

import torch
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and configuration for training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize device manager with configuration.
        
        Args:
            config: Device configuration dictionary
        """
        self.config = config
        self.device = None
        self.amp_enabled = False
        
    def select_training_device(self) -> str:
        """
        Choose the best available device for training.
        
        Returns:
            Device string suitable for Ultralytics: 'cuda:0', 'dml', or 'cpu'
        """
        device_priority = self.config.get('device_priority', ['cuda', 'dml', 'cpu'])
        
        for device_type in device_priority:
            if device_type == 'cuda':
                device = self._try_cuda()
                if device:
                    self.device = device
                    self.amp_enabled = self.config.get('amp', {}).get('cuda', True)
                    logger.info(f"✅ Selected CUDA device: {device}")
                    return device
                    
            elif device_type == 'dml':
                device = self._try_directml()
                if device:
                    self.device = device
                    self.amp_enabled = self.config.get('amp', {}).get('directml', False)
                    logger.info(f"✅ Selected DirectML device: {device}")
                    return device
                    
            elif device_type == 'cpu':
                device = self._configure_cpu()
                self.device = device
                self.amp_enabled = self.config.get('amp', {}).get('cpu', False)
                logger.info(f"✅ Selected CPU device: {device}")
                return device
                
        # Fallback to CPU if nothing else works
        logger.warning("⚠️ No preferred device available, falling back to CPU")
        self.device = 'cpu'
        self.amp_enabled = False
        return 'cpu'
    
    def _try_cuda(self) -> Optional[str]:
        """Try to use CUDA device with optional kernel testing."""
        if not torch.cuda.is_available():
            logger.debug("CUDA not available")
            return None
            
        cuda_config = self.config.get('cuda', {})
        test_kernel = cuda_config.get('test_kernel', True)
        fallback_on_error = cuda_config.get('fallback_on_error', True)
        
        if test_kernel:
            try:
                # Test with a small computation to ensure CUDA works
                x = torch.randn(32, 32, device="cuda")
                _ = x @ x.T  # Trigger a compute kernel
                logger.debug("CUDA kernel test passed")
                return "cuda:0"
            except Exception as e:
                if fallback_on_error:
                    logger.warning(f"⚠️ CUDA available but kernel test failed: {e}")
                    return None
                else:
                    raise RuntimeError(f"CUDA kernel test failed: {e}")
        else:
            return "cuda:0"
    
    def _try_directml(self) -> Optional[str]:
        """Try to use DirectML device (Windows GPU acceleration)."""
        directml_config = self.config.get('directml', {})
        test_ops = directml_config.get('test_ops', True)
        fallback_on_error = directml_config.get('fallback_on_error', True)
        
        try:
            import torch_directml  # type: ignore
            
            if test_ops:
                # Quick smoke test
                dml_device = torch_directml.device()
                x = torch.randn(16, 16, device=dml_device)
                _ = x @ x.T
                logger.debug("DirectML test passed")
                
            return "dml"
            
        except ImportError:
            logger.debug("DirectML not available (torch-directml not installed)")
            return None
        except Exception as e:
            if fallback_on_error:
                logger.warning(f"⚠️ DirectML test failed: {e}")
                return None
            else:
                raise RuntimeError(f"DirectML test failed: {e}")
    
    def _configure_cpu(self) -> str:
        """Configure CPU device."""
        cpu_config = self.config.get('cpu', {})
        threads = cpu_config.get('threads')
        
        if threads is not None:
            torch.set_num_threads(threads)
            logger.debug(f"Set CPU threads to {threads}")
            
        return "cpu"
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the selected device.
        
        Returns:
            Dictionary containing device information
        """
        if self.device is None:
            return {"error": "No device selected"}
            
        info = {
            "device": self.device,
            "amp_enabled": self.amp_enabled,
            "torch_version": torch.__version__,
        }
        
        if self.device.startswith("cuda"):
            info.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
            })
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
                
        elif self.device == "dml":
            try:
                import torch_directml
                info["directml_available"] = True
            except ImportError:
                info["directml_available"] = False
                
        return info
    
    def log_device_info(self):
        """Log device information."""
        info = self.get_device_info()
        
        logger.info("=" * 60)
        logger.info("DEVICE INFORMATION")
        logger.info("=" * 60)
        
        for key, value in info.items():
            logger.info(f"{key:20s}: {value}")
            
        if self.device == "dml":
            logger.info("Note: Using DirectML backend. Training will run on GPU via DirectML.")
            logger.info("Some operations may be slower than native CUDA.")
        elif self.device == "cpu":
            logger.info("Note: No usable GPU backend detected.")
            logger.info("Training will run on CPU (significantly slower).")
            
        logger.info("=" * 60)


def create_device_manager(config: Dict[str, Any]) -> DeviceManager:
    """
    Factory function to create a device manager.
    
    Args:
        config: Device configuration dictionary
        
    Returns:
        Configured DeviceManager instance
    """
    return DeviceManager(config)
