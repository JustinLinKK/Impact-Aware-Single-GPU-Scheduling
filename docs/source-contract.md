# Released-source capture contract

Released-source capture converts a declared workload into the existing 17-tensor
PerfSeer graph input. The source executes only in the isolated CPU worker; CUDA
visibility is disabled, network socket connections are denied, and the selected
workload is captured with strict backward graph capture plus eager numerical
replay. The graph schema and released student input schema are unchanged.

The preferred request metadata field is perfseer_model.workload_contract.
Version 1 contracts use version = "mlevolve.perfseer_workload.v1". They declare
the exact model entry, batch structure, loss semantics, precision, optimizer,
scheduler, backend, and local import roots. Unknown or incomplete recipes fail
with a structured error; the converter does not infer training semantics from
arbitrary scripts.

## Module workload

The module mode accepts a named nn.Module instance, class, or factory. Example:

    {
      "version": "mlevolve.perfseer_workload.v1",
      "mode": "module",
      "entry": "build_model",
      "import_roots": ["/workspace/project"],
      "constructor_args": [],
      "constructor_kwargs": {},
      "max_input_numel": 16000000,
      "inputs": {
        "positional": [
          {"shape": ["$batch", 12], "dtype": "float32", "distribution": "normal"},
          {"shape": ["$batch", 8], "dtype": "int64", "min": 0, "max": 32}
        ],
        "keyword": {"mask": {"shape": ["$batch", 8], "dtype": "bool"}}
      },
      "loss": {
        "kind": "mse",
        "output_path": ["prediction"],
        "target": {"shape": ["$batch", 1], "dtype": "float32", "fill": 0.0}
      },
      "training": {
        "precision": "fp32_ieee",
        "gradient_accumulation_steps": 1,
        "optimizer": {
          "name": "adamw",
          "lr": 0.001,
          "weight_decay": 0.0001,
          "foreach": null,
          "fused": null
        },
        "scheduler": {"name": "none"},
        "backend": "declared-backend-name"
      }
    }

TensorSpecs can be nested lists and objects. $batch is replaced by each
scheduling batch size. Float tensors need fill or distribution = "normal";
integer tensors need explicit half-open min and max. Supported dtypes are
float32, int64/int32/int16/int8, uint8, and bool. Structured outputs can be
selected by loss.output_path; only mean, mse, and cross_entropy are available
in this mode.

max_input_numel limits aggregate positional, keyword, and target elements (or
the CandidateAdapter batch). The default is 16 million elements and the hard
limit is 64 million. The declaration is checked before tensor allocation and
the materialized inputs are checked again.

Adam/AdamW resolve documented PyTorch defaults for betas, eps, and amsgrad;
SGD resolves momentum, dampening, and nesterov. Those resolved values are
included in the capture audit. foreach and fused must be a boolean or null.
null preserves PyTorch's pinned-version/backend default instead of claiming
that the option was explicitly false. The runtime fingerprint records the
PyTorch version.

## CandidateAdapter workload

Use mode = "candidate_adapter" when the real workload defines
CandidateAdapter.build_model(context), build_train_batch(scenario, device),
and training_step(model, batch, context). The contract must explicitly select
loss.kind = "candidate_adapter_training_step", supply context and scenario,
and give an exact adapter.batch_spec matching the returned
dictionary/tuple/tensor shapes and dtypes. The worker supplies device = "cpu"
and the requested batch_size; the declared batch fixture is supplied as
scenario["fixture"]. The training_step is called directly, so its actual loss
is captured rather than a surrogate loss.

If the adapter builds an optimizer, its exact type, resolved defaults,
parameter groups, and trainable-parameter coverage are checked against the
declared optimizer. The adapter owns loss semantics: its exact
training_step runs, including any source-defined criterion. Leave criterion
out of context when the adapter supplies its own default. Optional
adapter.context.criterion or adapter.criterion can provide a declared built-in
mse/cross_entropy criterion or a source_entry nn.Module criterion. Adapters requiring
dataset files, pretrained downloads, GPU execution, AMP, or gradient
accumulation other than one need a separate strict training graph.

## Safety and compatibility

Source import uses the selected entry, explicit import_roots, and only
literal top-level constants, safe pathlib constants, or a selected instance
built from allowlisted torch.nn constructors. The __main__ block is skipped.
Reachable pretrained-download calls and dynamic imports in the selected
workload are rejected. Local Python dependencies are statically hashed; if the
dependency closure is incomplete, persistent conversion caching must not
treat the source hash alone as sufficient.

The previous explicit top-level entry, input_shapes, input_dtypes, loss, and
optimizer fields remain supported. For this compatibility form only, the
previous zero-like target shaped from the first forward output is retained and
reported in the audit as legacy_target_semantics. New contracts must declare
their target or use the CandidateAdapter's exact training_step.
