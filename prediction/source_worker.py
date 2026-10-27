"""Isolated source conversion worker; imports torch only after hiding GPUs."""

from __future__ import annotations

import os
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


def convert_released_source_options_worker(connection, runtime_src, normalization_path, requests, cpu_threads):
    """Capture explicitly described workloads in an isolated CPU child."""
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
        from localml_scheduler.prediction.source_contract import (
            ContractError, WorkloadContract, at_path, build_adapter_batch,
            build_tree, fingerprint_source, disable_network,
            import_source_module, load_entry, tree_tensor_numel,
        )
        torch.set_num_threads(cpu_threads)
        normalizer = torch.load(normalization_path, map_location="cpu", weights_only=False)
        options = []
        audits = []
        disable_network()
        for request in requests:
            metadata = request["metadata"]
            if request.get("graph_path"):
                graph = GraphIRV3.load(request["graph_path"])
                if (not graph.training_mode or graph.coverage.backward_capture_quality != "strict"
                        or not any(n.phase == "backward" for n in graph.nodes)):
                    raise ValueError("strict training graph with backward phase is required")
                if not graph.coverage.replay_validated or graph.coverage.replay_samples < 1:
                    raise ValueError("precomputed training graph requires validated numeric replay evidence")
                requested_precision = request.get("precision")
                if requested_precision is None:
                    raise ValueError("precomputed training graph requires explicit requested workload precision")
                if _normalize_graph_precision(requested_precision) != _normalize_graph_precision(graph.precision):
                    raise ValueError("precomputed graph precision differs from requested workload precision")
                expected_optimizer = _declared_graph_optimizer(metadata)
                actual_optimizer = (graph.training_config.get("optimizer")
                                    or graph.training_config.get("optimizer_config")
                                    or graph.optimizer_config)
                if not isinstance(actual_optimizer, dict) or not actual_optimizer.get("name"):
                    raise ValueError("precomputed training graph lacks explicit optimizer provenance")
                scheduler = graph.training_config.get("scheduler")
                if not isinstance(scheduler, dict) or not scheduler.get("name") or scheduler.get("name") == "unknown":
                    raise ValueError("precomputed training graph lacks explicit scheduler provenance")
                if not isinstance(graph.training_config.get("backend"), str) or not graph.training_config["backend"].strip():
                    raise ValueError("precomputed training graph lacks explicit backend provenance")
                if expected_optimizer is not None:
                    _check_graph_optimizer(expected_optimizer, actual_optimizer)
                contract_training = (metadata.get("workload_contract", {}).get("training", {})
                                     if isinstance(metadata.get("workload_contract"), dict) else {})
                accumulation = int(metadata.get("gradient_accumulation_steps",
                                                contract_training.get("gradient_accumulation_steps", 1)))
                if metadata.get("graph_step_semantics") != "complete_optimizer_step":
                    raise ValueError("precomputed graph must declare complete_optimizer_step semantics")
                if int(graph.training_config.get("gradient_accumulation_steps", 1)) != accumulation:
                    raise ValueError("training graph accumulation differs from requested optimizer step")
                if int(graph.metadata.get("batch_size", 0)) != request["batch_size"]:
                    raise ValueError("training graph physical batch differs from scheduling option")
                if graph.metadata.get("target_hardware_id") != metadata.get("graph_hardware_id"):
                    raise ValueError("training graph requires explicit matching graph_hardware_id")
                assert_physical_hardware_identity(metadata["graph_hardware_id"], request["hardware"]["name"])
                graph_audit = {
                    "mode": "precomputed_graph", "entry": None,
                    "node_count": len(graph.nodes),
                    "backward_node_count": sum(node.phase == "backward" for node in graph.nodes),
                    "numeric_parity": bool(graph.coverage.replay_validated),
                    "precision": graph.precision,
                    "optimizer": dict(actual_optimizer),
                    "source_dependency_fingerprint": None,
                }
            else:
                contract_value = request.get("workload_contract")
                metadata_contract = metadata.get("workload_contract")
                if contract_value is not None and metadata_contract is not None and contract_value != metadata_contract:
                    raise ContractError("conflicting_contracts", "request.workload_contract differs from metadata.workload_contract")
                contract = WorkloadContract.parse(
                    contract_value if contract_value is not None else metadata_contract,
                    request=request, metadata=metadata,
                )
                if int(metadata.get("gradient_accumulation_steps", 1)) != 1:
                    raise ContractError("unsupported_accumulation", "source capture supports only explicitly declared gradient_accumulation_steps=1; supply a complete optimizer-step graph otherwise")
                if request.get("precision") not in (None, contract.precision):
                    raise ContractError("precision_mismatch", "request precision does not match the explicit workload contract")
                source_path = Path(request["source_path"]).expanduser().resolve()
                import_roots = contract.raw.get("import_roots", metadata.get("import_roots", []))
                if not isinstance(import_roots, list) or any(not isinstance(root, str) for root in import_roots):
                    raise ContractError("invalid_import_roots", "import_roots must be a list of paths")
                dependency_receipt = fingerprint_source(source_path, {**metadata, "workload_contract": contract.raw})
                # Keep declared project roots available after import as well:
                # factories and CandidateAdapter methods may use lazy sibling
                # imports during build, batch creation, or training_step.
                for root in reversed([str(source_path.parent), *(str(Path(root).expanduser().resolve()) for root in import_roots)]):
                    if root not in sys.path:
                        sys.path.insert(0, root)
                torch.manual_seed(0)
                module = import_source_module(source_path, import_roots=import_roots, entry=contract.entry)
                model, candidate_adapter, adapter_details = load_entry(module, contract, torch=torch, request=request)
                if contract.mode == "module":
                    input_spec = contract.raw["inputs"]
                    rng = torch.Generator().manual_seed(0)
                    args = tuple(build_tree(value, batch_size=int(request["batch_size"]), torch=torch, generator=rng)
                                 for value in input_spec["positional"])
                    kwargs = {key: build_tree(value, batch_size=int(request["batch_size"]), torch=torch, generator=rng)
                              for key, value in input_spec["keyword"].items()}
                    target_spec = contract.loss["target"]
                    legacy_target = isinstance(target_spec, dict) and target_spec.get("legacy_output_shape") is True
                    if legacy_target:
                        with torch.no_grad():
                            target_probe = model(*args, **kwargs)
                        target_probe = at_path(target_probe, contract.loss.get("output_path", []))
                        if not isinstance(target_probe, torch.Tensor):
                            raise ContractError("invalid_legacy_target", "legacy output-shaped target requires a tensor output")
                        if contract.loss["kind"] == "cross_entropy":
                            if target_probe.ndim < 2:
                                raise ContractError("invalid_legacy_target", "legacy cross_entropy requires logits with a class dimension at index 1")
                            target = torch.zeros((target_probe.shape[0], *target_probe.shape[2:]), dtype=torch.int64)
                        else:
                            target = torch.zeros_like(target_probe, memory_format=torch.preserve_format)
                    else:
                        target = build_tree(target_spec, batch_size=int(request["batch_size"]), torch=torch,
                                            generator=torch.Generator().manual_seed(10_000))
                    actual_numel = tree_tensor_numel((args, kwargs, target))
                    if actual_numel > contract.max_input_numel:
                        raise ContractError("input_bound_exceeded", "materialized inputs exceed max_input_numel", context={"actual_numel": actual_numel, "maximum": contract.max_input_numel})

                    class DeclaredLossModel(torch.nn.Module):
                        def __init__(self, inner, recipe):
                            super().__init__()
                            self.inner = inner
                            self.recipe = recipe

                        def forward(self, inputs):
                            positional, keyword = inputs
                            return self.inner(*positional, **keyword)

                        def compute_loss(self, prediction, batch, adapter):
                            selected = at_path(prediction, self.recipe.get("output_path", []))
                            kind = self.recipe["kind"]
                            if kind == "mean":
                                return selected.mean()
                            if kind == "mse":
                                return torch.nn.functional.mse_loss(selected.float(), batch.float())
                            if kind == "cross_entropy":
                                return torch.nn.functional.cross_entropy(selected, batch.long())
                            raise ContractError("unsupported_loss", f"unsupported loss {kind!r}")

                    capture_model, inputs, capture_batch, capture_adapter = DeclaredLossModel(model, contract.loss), (args, kwargs), target, object()
                else:
                    batch, adapter_context = build_adapter_batch(
                        candidate_adapter, model, adapter_details, torch=torch,
                        max_input_numel=contract.max_input_numel, optimizer_spec=contract.optimizer,
                    )

                    class CandidateStepModel(torch.nn.Module):
                        def __init__(self, inner, source_adapter, context):
                            super().__init__()
                            self.inner = inner
                            self.source_adapter = source_adapter
                            self.context = context

                        def forward(self, inputs):
                            loss_value = self.source_adapter.training_step(self.inner, inputs, self.context)
                            if not isinstance(loss_value, torch.Tensor) or loss_value.numel() != 1:
                                raise ContractError("invalid_adapter_loss", "CandidateAdapter.training_step must return a scalar loss tensor")
                            return loss_value

                        def compute_loss(self, prediction, batch, adapter):
                            return prediction

                    capture_model, inputs, capture_batch, capture_adapter = CandidateStepModel(model, candidate_adapter, adapter_context), batch, None, object()
                training_spec = contract.raw.get("training", {})
                if not isinstance(training_spec.get("scheduler"), dict) or not training_spec["scheduler"].get("name"):
                    raise ContractError("missing_scheduler", "training.scheduler.name must be explicit; use 'none' when no scheduler is used")
                training = {**dict(contract.raw.get("training_config", {})),
                    "precision": contract.precision,
                    "microbatch_size": int(request["batch_size"]),
                    "optimizer": contract.optimizer,
                    "scheduler": dict(training_spec["scheduler"]),
                    "backend": training_spec.get("backend"),
                    "steps_per_epoch": int(request["raw_steps"]),
                    "gradient_accumulation_steps": 1}
                design, audit = capture(capture_model, inputs, capture_batch, capture_adapter, training,
                    architecture_key=hashlib.sha256(source_path.read_bytes()).hexdigest())
                if not audit["numerically_finite_and_close"]:
                    raise ContractError("numerical_replay_failed", "native forward/loss/backward replay failed")
                graph = graph_from_design(design)
                graph_audit = {
                    "mode": contract.mode, "entry": contract.entry,
                    "node_count": len(graph.nodes),
                    "backward_node_count": sum(node.phase == "backward" for node in graph.nodes),
                    "numeric_parity": bool(audit["numerically_finite_and_close"]),
                    "precision": contract.precision,
                    "optimizer": dict(contract.optimizer),
                    "max_input_numel": contract.max_input_numel,
                    "materialized_input_numel": actual_numel if contract.mode == "module" else tree_tensor_numel(batch),
                    "source_dependency_fingerprint": dependency_receipt,
                    "legacy_target_semantics": "zero_like_first_forward_output" if contract.mode == "module" and isinstance(contract.loss.get("target"), dict) and contract.loss["target"].get("legacy_output_shape") else None,
                }
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
            audits.append(graph_audit)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU conversion unexpectedly initialized CUDA")
        connection.send({"ok": True, "tensors_by_option": options, "audits": audits, "cuda_initialized": False})
    except BaseException as exc:
        detail = exc.as_dict() if hasattr(exc, "as_dict") else {"code": "capture_failed", "message": str(exc), "context": {}}
        connection.send({"ok": False, "error": f"{detail['code']}: {detail['message']}",
                         "error_detail": detail, "phase": "source_contract_or_capture", "cuda_initialized": False})
    finally:
        connection.close()


def _normalize_graph_precision(value):
    aliases = {"fp32": "fp32_ieee", "float32": "fp32_ieee", "bf16": "bf16_amp",
               "bfloat16": "bf16_amp", "fp16": "fp16_amp", "float16": "fp16_amp"}
    return aliases.get(str(value).strip().lower(), str(value).strip().lower())


def _declared_graph_optimizer(metadata):
    contract = metadata.get("workload_contract")
    training = contract.get("training", {}) if isinstance(contract, dict) else {}
    optimizer = training.get("optimizer") if isinstance(training, dict) else None
    if optimizer is None:
        optimizer = metadata.get("optimizer_config")
    if optimizer is None and metadata.get("optimizer") is not None:
        optimizer = {"name": metadata["optimizer"]}
    if isinstance(optimizer, str):
        return {"name": optimizer}
    return optimizer if isinstance(optimizer, dict) else None


def _check_graph_optimizer(expected, actual):
    if not isinstance(actual, dict):
        raise ValueError("precomputed graph lacks optimizer provenance needed to verify the requested recipe")
    expected_name = str(expected.get("name", "")).lower()
    actual_name = str(actual.get("name", actual.get("optimizer", ""))).lower()
    if expected_name and expected_name != actual_name:
        raise ValueError(f"precomputed graph optimizer {actual_name!r} differs from requested {expected_name!r}")
    mismatches = {}
    for key, value in expected.items():
        if key == "name":
            continue
        actual_value = actual.get(key)
        canonical_expected = tuple(value) if isinstance(value, list) else value
        canonical_actual = tuple(actual_value) if isinstance(actual_value, list) else actual_value
        if canonical_expected != canonical_actual:
            mismatches[key] = {"requested": value, "graph": actual_value}
    if mismatches:
        raise ValueError(f"precomputed graph optimizer configuration differs from requested recipe: {mismatches}")


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
