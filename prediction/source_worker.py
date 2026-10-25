"""Isolated source conversion worker; imports torch only after hiding GPUs."""

from __future__ import annotations

import os
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


def convert_released_source_options_worker(connection, runtime_src, normalization_path, requests, cpu_threads):
    """Capture the same strict backward graph and feature schema as training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    try:
        from dataclasses import replace
        import hashlib
        sys.path[:0] = [str(Path(runtime_src).parent / "aliases"), runtime_src]
        import torch
        from perfseer_v3.features import apply_normalization, batch_graph_features
        from perfseer_v3.graph_ir_v3 import GraphIRV3
        from perfseer_v3.hardware import assert_physical_hardware_identity
        from perfseer_v31.capture_training import capture, graph_from_design
        from perfseer_v32.features import _build_features
        torch.set_num_threads(cpu_threads)
        normalizer = torch.load(normalization_path, map_location="cpu", weights_only=False)
        options = []
        for request in requests:
            metadata = request["metadata"]
            accumulation = int(metadata.get("gradient_accumulation_steps", 1))
            if request.get("graph_path"):
                graph = GraphIRV3.load(request["graph_path"])
                if (not graph.training_mode or graph.coverage.backward_capture_quality != "strict"
                        or not any(n.phase == "backward" for n in graph.nodes)):
                    raise ValueError("strict training graph with backward phase is required")
                if metadata.get("graph_step_semantics") != "complete_optimizer_step":
                    raise ValueError("precomputed graph must declare complete_optimizer_step semantics")
                if int(graph.training_config.get("gradient_accumulation_steps", 1)) != accumulation:
                    raise ValueError("training graph accumulation differs from requested optimizer step")
                if int(graph.metadata.get("batch_size", 0)) != request["batch_size"]:
                    raise ValueError("training graph physical batch differs from scheduling option")
                if graph.metadata.get("target_hardware_id") != metadata.get("graph_hardware_id"):
                    raise ValueError("training graph requires explicit matching graph_hardware_id")
                assert_physical_hardware_identity(metadata["graph_hardware_id"], request["hardware"]["name"])
            else:
                from perfseer_v3.inputs import build_inputs
                from perfseer_v3.load import load_model_entry
                from perfseer_v3.source import SourceModelSpecV3
                if accumulation != 1:
                    raise ValueError("source capture requires gradient_accumulation_steps=1; supply a complete optimizer-step graph otherwise")
                if request["precision"] != "fp32_ieee":
                    raise ValueError("non-FP32 source capture requires an explicit training_graph_path")
                loss_name, optimizer = metadata.get("loss"), metadata.get("optimizer")
                if loss_name not in ("mean", "mse", "cross_entropy") or optimizer not in ("adam", "adamw", "sgd"):
                    raise ValueError("source prediction requires explicit loss (mean/mse/cross_entropy) and optimizer (adam/adamw/sgd)")
                spec = SourceModelSpecV3.from_shapes(request["source_path"], request["entry"],
                    tuple(tuple(s) for s in request["input_shapes"]), input_dtypes=tuple(request["input_dtypes"]),
                    constructor_args=tuple(request["constructor_args"]), constructor_kwargs=request["constructor_kwargs"])
                torch.manual_seed(0)
                model = load_model_entry(spec.source_path, spec.entry, constructor_args=spec.constructor_args,
                                        constructor_kwargs=dict(spec.constructor_kwargs)).cpu().train()
                args, kwargs = build_inputs(spec.positional_inputs, seed=0)
                with torch.no_grad():
                    output = model(*args, **kwargs)
                if not isinstance(output, torch.Tensor):
                    raise ValueError("source capture currently requires one tensor output")
                target = (torch.zeros((output.shape[0], *output.shape[2:]), dtype=torch.long)
                          if loss_name == "cross_entropy" else torch.zeros_like(output))

                class DeclaredLossModel(torch.nn.Module):
                    def __init__(self, inner):
                        super().__init__()
                        self.inner = inner
                    def forward(self, inputs):
                        return self.inner(*inputs, **kwargs)
                    def compute_loss(self, prediction, batch, adapter):
                        if loss_name == "cross_entropy":
                            return torch.nn.functional.cross_entropy(prediction, batch)
                        if loss_name == "mse":
                            return torch.nn.functional.mse_loss(prediction, batch)
                        return prediction.mean()

                training = {**dict(metadata.get("training_config") or {}),
                    "precision": request["precision"], "microbatch_size": request["batch_size"],
                    "optimizer": {**dict(metadata.get("optimizer_config") or {}), "name": optimizer},
                    "scheduler": dict(metadata.get("scheduler") or {"name": "unknown"}),
                    "backend": metadata.get("execution_backend", "unknown"),
                    "steps_per_epoch": request["raw_steps"], "gradient_accumulation_steps": 1}
                design, audit = capture(DeclaredLossModel(model), args, target, True, training,
                    architecture_key=hashlib.sha256(Path(request["source_path"]).read_bytes()).hexdigest())
                if not audit["numerically_finite_and_close"]:
                    raise ValueError("native backward capture failed numerical replay")
                graph = graph_from_design(design)
            # Set target identity before constructing and scaling any features.
            graph = replace(graph, metadata={**{k: v for k, v in graph.metadata.items() if k != "hardware_profile"},
                "target_hardware_id": request["hardware"]["name"],
                "hardware_features": {k: v for k, v in request["hardware"].items() if k != "name" and v is not None}},
                training_config={**graph.training_config, "steps_per_epoch": request["raw_steps"]})
            feature = apply_normalization(_build_features(graph), normalizer)
            batch = batch_graph_features([feature])
            packed = (batch.x_cont, batch.op_exact_id, batch.phase_id, batch.dtype_id, batch.node_flags,
                      batch.edge_index, batch.edge_cont, batch.edge_role_id, batch.edge_flags, batch.batch,
                      batch.u_cont, batch.quality, batch.precision_id, batch.optimizer_id,
                      batch.hardware_cont, batch.hardware_missing_mask, torch.tensor([request["raw_steps"]]))
            options.append(tuple(t.cpu().numpy() for t in packed))
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU conversion unexpectedly initialized CUDA")
        connection.send({"ok": True, "tensors_by_option": options, "cuda_initialized": False})
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def convert_source_worker(
    connection: Connection,
    submodule_src: str,
    request: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    try:
        source_root = str(Path(submodule_src).resolve())
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        import torch

        torch.set_num_threads(int(request.get("cpu_threads", 1)))
        from perfseer_student import encode_source

        encoded = encode_source(
            request["source_path"],
            request["entry"],
            request["input_shapes"],
            request["precision"],
            constructor_args=request["constructor_args"],
            constructor_kwargs=request["constructor_kwargs"],
            input_dtypes=request["input_dtypes"],
        )
        connection.send(
            {
                "ok": True,
                "tensors": tuple(tensor.cpu().numpy() for tensor in encoded.as_tuple()),
            }
        )
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def convert_source_options_worker(
    connection: Connection,
    submodule_src: str,
    requests: list[dict[str, Any]],
) -> None:
    """Trace one model once, then shape-propagate/featurize every batch option."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    try:
        if not requests:
            connection.send({"ok": True, "tensors_by_option": []})
            return
        source_root = str(Path(submodule_src).resolve())
        if source_root not in sys.path:
            sys.path.insert(0, source_root)

        import torch

        torch.set_num_threads(int(requests[0].get("cpu_threads", 1)))
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp
        from perfseer_source_converter.converter import (
            SourceModelSpec,
            _example_inputs,
            _fx_to_networkx,
            _load_source_model,
        )
        from perfseer_student.features import featurize_graph

        def source_spec(request: dict[str, Any]) -> SourceModelSpec:
            return SourceModelSpec(
                source_path=request["source_path"],
                entry=request["entry"],
                input_shapes=request["input_shapes"],
                constructor_args=tuple(request["constructor_args"]),
                constructor_kwargs=dict(request["constructor_kwargs"]),
                input_dtypes=tuple(request["input_dtypes"]),
            )

        model = _load_source_model(source_spec(requests[0]))
        traced = symbolic_trace(model)
        tensors_by_option: list[tuple[Any, ...]] = []
        for request in requests:
            spec = source_spec(request)
            with torch.no_grad():
                ShapeProp(traced).propagate(*_example_inputs(spec))
            graph = _fx_to_networkx(traced)
            x, edge_index, edge_attr, u = featurize_graph(graph, request["precision"])
            x_tensor = torch.from_numpy(x)
            tensors = (
                x_tensor,
                torch.from_numpy(edge_index),
                torch.from_numpy(edge_attr),
                torch.from_numpy(u),
                torch.zeros(x_tensor.shape[0], dtype=torch.long),
            )
            tensors_by_option.append(tuple(tensor.cpu().numpy() for tensor in tensors))
        connection.send({"ok": True, "tensors_by_option": tensors_by_option})
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()
