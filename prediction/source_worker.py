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
