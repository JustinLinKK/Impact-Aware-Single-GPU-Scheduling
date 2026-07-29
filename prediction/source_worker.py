"""Isolated source conversion worker; imports torch only after hiding GPUs."""

from __future__ import annotations

import os
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


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
