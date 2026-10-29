"""Load the credited SPaGS wrapper without importing its full trainer package."""
import importlib.util
import sys
from pathlib import Path


_source = (Path(__file__).resolve().parents[1] / "third_party/SPaGS/SPaGSCudaBackend"
           / "SPaGSCudaBackend/torch_bindings/rasterization.py")
_spec = importlib.util.spec_from_file_location("native_gs._spags_rasterization", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"SPaGS Python wrapper unavailable: {_source}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
SPaGSRasterizer = _module.SPaGSRasterizer
