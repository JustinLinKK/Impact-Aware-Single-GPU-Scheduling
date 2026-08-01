"""PerfSeer SeerNet runtime bridge.

Loads the external Predictor repository (SeerNetMulti + fx-trace converter +
teacher featurizer) and predicts the six PerfSeer targets for a training-script
source file:

    train_util (%), train_mem (MiB), train_time (ms/step),
    infer_util (%), infer_mem (MiB), infer_time (ms/step)

The Predictor repo is not installed as a package; it is loaded from
``repo_path`` (directory containing ``predictor/`` and ``teacher/``).
"""

from __future__ import annotations

import ast
import logging
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("localml_scheduler.prediction.perfseer")

TARGET_NAMES = ("train_util", "train_mem", "train_time", "infer_util", "infer_mem", "infer_time")

_PRECISION_MAP = {
    "fp32": "fp32_ieee",
    "fp32_ieee": "fp32_ieee",
    "float32": "fp32_ieee",
    "tf32": "tf32",
    "bf16": "bf16_amp",
    "bf16_amp": "bf16_amp",
    "bfloat16": "bf16_amp",
    "fp16": "fp16_amp",
    "fp16_amp": "fp16_amp",
    "float16": "fp16_amp",
    "amp": "bf16_amp",
}


def normalize_precision(precision: str | None) -> str:
    return _PRECISION_MAP.get(str(precision or "").strip().lower(), "fp32_ieee")


_RESOLUTION_PATTERNS = (
    r"Resize\(\((\d+)",
    r"Resize\((\d+)",
    r"IMG_SIZE\s*=\s*(\d+)",
    r"image_size\s*=\s*(\d+)",
    r"\(\s*(\d{2,4})\s*,\s*\1\s*\)",
)

# Generated scripts occasionally use model names rejected by current timm.
_MODEL_NAME_ALIASES = {
    "efficientnet_v2_s": "tf_efficientnetv2_s",
    "efficientnet_v2_m": "tf_efficientnetv2_m",
    "efficientnet_v2_l": "tf_efficientnetv2_l",
}


def detect_input_resolution(source_text: str) -> int | None:
    import re

    for pattern in _RESOLUTION_PATTERNS:
        match = re.search(pattern, source_text)
        if match:
            value = int(match.group(1))
            if 16 <= value <= 2048:
                return value
    return None


def detect_entry_spec(source_text: str) -> tuple[str | None, dict[str, Any]]:
    """Pick the nn.Module subclass a training script instantiates as its model.

    Preference order: class instantiated inside a ``get_model*`` function
    (with literal / module-constant constructor kwargs resolved), otherwise
    the last nn.Module subclass defined in the file.
    """
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None, {}
    module_classes: list[str] = []
    constants: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant):
                constants[node.targets[0].id] = node.value.value
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if "Module" in ast.dump(base):
                    module_classes.append(node.name)
                    break
    if not module_classes:
        return None, {}
    class_set = set(module_classes)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("get_model"):
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in class_set:
                    kwargs: dict[str, Any] = {}
                    for keyword in call.keywords:
                        if keyword.arg is None:
                            continue
                        if isinstance(keyword.value, ast.Constant):
                            kwargs[keyword.arg] = keyword.value.value
                        elif isinstance(keyword.value, ast.Name) and keyword.value.id in constants:
                            kwargs[keyword.arg] = constants[keyword.value.id]
                    return call.func.id, kwargs
    return module_classes[-1], {}


def detect_entry_class(source_text: str) -> str | None:
    return detect_entry_spec(source_text)[0]


@dataclass(frozen=True, slots=True)
class PerfSeerResult:
    train_util_percent: float
    train_mem_mib: float
    train_step_time_ms: float
    infer_util_percent: float
    infer_mem_mib: float
    infer_step_time_ms: float
    entry_class: str
    node_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_util_percent": self.train_util_percent,
            "train_mem_mib": self.train_mem_mib,
            "train_step_time_ms": self.train_step_time_ms,
            "infer_util_percent": self.infer_util_percent,
            "infer_mem_mib": self.infer_mem_mib,
            "infer_step_time_ms": self.infer_step_time_ms,
            "entry_class": self.entry_class,
            "node_count": self.node_count,
        }


class PerfSeerRuntime:
    """Lazily loaded SeerNet inference runtime with a small LRU cache."""

    def __init__(
        self,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        cache_size: int = 256,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device_name = str(device or "cpu")
        self.cache_size = max(0, int(cache_size))
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: str | None = None
        self._cache: OrderedDict[tuple, PerfSeerResult] = OrderedDict()
        self._torch = None
        self._model = None
        self._stats = None
        self._pipeline = None
        self._converter = None

    def load_error(self) -> str | None:
        self._ensure_loaded()
        return self._load_error

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._load_error is None
        with self._lock:
            if self._loaded:
                return self._load_error is None
            try:
                self._load()
            except Exception as exc:  # fail closed: scheduler falls back to other providers
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.warning("PerfSeer runtime unavailable: %s", self._load_error)
            self._loaded = True
        return self._load_error is None

    def _load(self) -> None:
        predictor_dir = self.repo_path / "predictor"
        teacher_dir = self.repo_path / "teacher"
        if not predictor_dir.is_dir() or not teacher_dir.is_dir():
            raise FileNotFoundError(f"Predictor repo layout not found under {self.repo_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        for path in (str(predictor_dir), str(teacher_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        import torch

        import converter as perfseer_converter
        import pipeline as perfseer_pipeline
        from model import SeerNetConfig, SeerNetMulti

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        cfg = SeerNetConfig(**ckpt["cfg"])
        model = SeerNetMulti(cfg)
        model.load_state_dict(ckpt["model"])
        device = torch.device(self.device_name if self.device_name != "cuda" or torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        stats = ckpt["stats"]
        expected_global = int(ckpt["cfg"]["global_dim"])
        featurized_global = int(stats["g_mean"].shape[0]) + len(perfseer_pipeline.PRECISIONS)
        if expected_global != featurized_global:
            raise ValueError(
                f"checkpoint global_dim {expected_global} incompatible with featurizer ({featurized_global}); "
                "use a checkpoint trained with the current teacher/pipeline.py"
            )

        self._torch = torch
        self._model = model
        self._device = device
        self._stats = stats
        self._pipeline = perfseer_pipeline
        self._converter = perfseer_converter

    def predict_source(
        self,
        source_path: str | Path,
        *,
        batch_size: int,
        input_resolution: int | None = None,
        input_shape: tuple[int, ...] | None = None,
        precision: str | None = None,
        entry_class: str | None = None,
    ) -> PerfSeerResult | None:
        """Predict PerfSeer targets for one training script; None on any failure."""
        if not self._ensure_loaded():
            return None
        source = Path(source_path)
        if not source.is_file():
            logger.debug("PerfSeer: source not found: %s", source)
            return None
        source_text = source.read_text()
        if "AutoModel.from_pretrained" in source_text or "from transformers import" in source_text or "import transformers" in source_text:
            logger.info("PerfSeer: skipping %s (HF transformers backbone is not fx-traceable)", source.name)
            return None
        precision_key = normalize_precision(precision)
        resolution = int(input_resolution or detect_input_resolution(source_text) or 224)
        if input_shape and len(input_shape) >= 2:
            shape = tuple(int(v) for v in input_shape)
            if len(shape) == 3:
                shape = (int(batch_size),) + shape
        else:
            shape = (int(batch_size), 3, resolution, resolution)
        cache_key = (str(source), source.stat().st_mtime_ns, shape, precision_key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached
        try:
            result = self._predict(source, shape, precision_key, entry_class)
        except Exception as exc:
            logger.info("PerfSeer prediction failed for %s: %s: %s", source.name, type(exc).__name__, exc)
            return None
        if self.cache_size:
            self._cache[cache_key] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def _predict(
        self,
        source: Path,
        shape: tuple[int, ...],
        precision_key: str,
        entry_class: str | None,
    ) -> PerfSeerResult:
        import numpy as np

        torch = self._torch
        pipeline = self._pipeline
        converter = self._converter

        detected_entry, ctor_kwargs = detect_entry_spec(source.read_text())
        entry = entry_class or detected_entry
        if not entry:
            raise ValueError("no nn.Module entry class detected")
        if entry_class and entry_class != detected_entry:
            ctor_kwargs = {}
        model_name = ctor_kwargs.get("model_name")
        if isinstance(model_name, str) and model_name in _MODEL_NAME_ALIASES:
            ctor_kwargs["model_name"] = _MODEL_NAME_ALIASES[model_name]

        spec = converter.SourceModelSpec(
            source_path=source,
            entry=entry,
            input_shapes=(shape,),
            constructor_kwargs=ctor_kwargs,
        )
        graph = converter.convert_source_to_networkx(spec)
        xo, xc, ei, ec, gc = pipeline.featurize_graph(graph)

        st = self._stats
        x = np.concatenate([xo, (xc - st["x_mean"]) / st["x_std"]], axis=1).astype(np.float32)
        e = ((ec - st["e_mean"]) / st["e_std"]).astype(np.float32) if ec.shape[0] else np.zeros((0, 3), np.float32)
        g = ((gc - st["g_mean"]) / st["g_std"]).astype(np.float32)
        prec_oh = np.zeros(len(pipeline.PRECISIONS), dtype=np.float32)
        prec_oh[pipeline.PREC_INDEX[precision_key]] = 1.0

        class _Batch:
            pass

        data = _Batch()
        data.x = torch.from_numpy(x).to(self._device)
        data.edge_index = torch.from_numpy(ei).to(self._device)
        data.edge_attr = torch.from_numpy(e).to(self._device)
        data.u = torch.from_numpy(np.concatenate([g, prec_oh])[None, :]).to(self._device)
        data.batch = torch.zeros(x.shape[0], dtype=torch.long, device=self._device)
        data.num_graphs = 1

        with torch.no_grad():
            pred_std = self._model(data).cpu().numpy()[0]
        y_log = pred_std * st["y_std"] + st["y_mean"]
        y_raw = np.maximum(np.expm1(y_log), 0.0)

        return PerfSeerResult(
            train_util_percent=float(y_raw[0]),
            train_mem_mib=float(y_raw[1]),
            train_step_time_ms=float(y_raw[2]),
            infer_util_percent=float(y_raw[3]),
            infer_mem_mib=float(y_raw[4]),
            infer_step_time_ms=float(y_raw[5]),
            entry_class=entry,
            node_count=int(x.shape[0]),
        )
