from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

from .core import read_opset

MODELS = ("mlp", "cnn", "resnet18", "unet", "lstm", "gru", "transformer", "gpt", "lora", "autoencoder")
SUITES = ("inventory", "operators", "tensors", "autograd", "correctness", "optimizers", "precision",
          "capture", "throughput", "native_cpu", "native_cuda", "transfer", "graphs", "datasets",
          "sessions", "placement", "concurrency", "isolation", "checkpoint", "migration", "disconnect",
          "faults", "limits", "soak", "wan", "shim", "c_api", "protocol", "compatibility")


@dataclass
class Case:
    id: str
    suite: str
    handler: str
    params: dict = field(default_factory=dict)
    timeout: int = 300
    plan_refs: tuple = ("P0.RT.02", "P0.RT.04")

    def json(self):
        return asdict(self)


def catalog(c, suites):
    out = []

    def add(suite, key, handler=None, timeout=None, refs=None, **p):
        if suite in suites:
            out.append(Case(f"{suite}/{key}", suite, handler or suite, p,
                            timeout or c["timeout_seconds"], refs or ("P0.RT.02", "P0.RT.04")))

    for what in ("client", "capabilities", "transport", "cli_status", "cli_verify", "cli_doctor"):
        add("inventory", what, what=what)
    if "operators" in suites:
        add("operators", "prepare", "prepare_ops", timeout=900)
        for name, entry in read_opset().items():
            add("operators", name, "operator", name=name, op_id=entry[0], flags=entry[1], tier=entry[2],
                capability=entry[6])
    from .operations import GRAD_NAMES, NAMES
    for name in NAMES:
        add("tensors", name, "tensor", name=name)
    for name, order in itertools.product(GRAD_NAMES, (1, 2)):
        add("autograd", f"{name}/order{order}", name=name, order=order)
    for model, mode, kind in itertools.product(MODELS, ("eager", "capture"), ("cuda", "cpu")):
        add("correctness", f"{model}/{mode}/{kind}", "train", model=model, mode=mode,
            target=f"remote_{kind}", check=True, batch=8, residency="upload", sync_every=1, optimizer="sgd")
    for name, mode, zero in itertools.product(
            ("sgd", "momentum", "nesterov", "adam", "adamw", "adamw_capturable", "adamw_fused", "sgd_fused",
             "adagrad", "adagrad_fused", "rmsprop", "adadelta", "adamax", "rprop", "asgd", "lbfgs"),
            ("eager", "capture"), (True, False)):
        add("optimizers", f"{name}/{mode}/zero_none{zero}", "train", model="mlp", mode=mode,
            target="remote_cuda", check=True, batch=8, residency="upload", sync_every=1,
            optimizer=name, zero_none=zero)
    for dtype, mode in itertools.product(("float64", "float32", "float16", "bfloat16", "amp_bf16", "amp_fp16"),
                                         ("eager", "capture")):
        add("precision", f"{dtype}/{mode}", "train", model="mlp", mode=mode, dtype=dtype,
            target="remote_cuda", check=True, batch=8, residency="upload", sync_every=1, optimizer="sgd")
    for what in ("changed_values", "changed_shape", "changed_dtype", "mutated_argument", "changed_scalar",
                 "python_branch", "stale_view", "rng", "nested_output", "gradient_accumulation", "lr_schedule"):
        add("capture", what, what=what)
    for suite, target in (("throughput", "remote_cuda"), ("native_cpu", "native_cpu"), ("native_cuda", "native_cuda")):
        modes = ("eager", "capture") if suite == "throughput" else ("eager",)
        for model, batch, mode, residency in itertools.product(MODELS, c["batch_sizes"], modes, ("resident", "upload")):
            # Training BatchNorm sees [1, 512, 1, 1] for this workload.
            if model == "resnet18" and batch == 1:
                continue
            add(suite, f"{model}/b{batch}/{mode}/{residency}", "train", model=model, mode=mode, target=target,
                check=False, batch=batch, residency=residency, sync_every=0, optimizer="momentum")
    for sync, mode in itertools.product((1, 5, 0), ("eager", "capture")):
        add("throughput", f"sync/{mode}/every{sync}", "train", model="mlp", mode=mode, target="remote_cuda",
            check=False, batch=32, residency="upload", sync_every=sync, optimizer="momentum")
    for n, direction in itertools.product(c["transfer_bytes"], ("upload", "fetch", "roundtrip")):
        add("transfer", f"{direction}/{n}", nbytes=n, direction=direction)
    for n in c["matrix_sizes"]:
        add("graphs", f"matmul/{n}", what="matmul", size=n)
    for n, boundary in itertools.product((1, 8, 64, c["max_graph_chain"]), ("each", "end")):
        add("graphs", f"chain/{n}/{boundary}", what="chain", size=n, boundary=boundary)
    for what in ("cache_reuse", "shape_churn", "scalar_churn", "multi_output", "dead_code", "alias_mutation"):
        add("graphs", what, what=what)
    for what in ("roundtrip", "resume_upload", "dedup_upload", "loader", "capture_loader", "bad_checksum", "bad_manifest"):
        add("datasets", what, what=what, timeout=600)
    for what in ("open_close", "multi_device", "move_gradients", "cross_session_reject", "closed_tensor", "seed", "threads"):
        add("sessions", what, what=what)
    for mode in ("auto", "remote", "local", "remote-cpu"):
        add("placement", mode, mode=mode)
    for n in c["concurrency"]:
        add("concurrency", f"clients{n}", n=n, timeout=max(600, c["timeout_seconds"]))
    for what in ("cross_session_handle", "cross_identity_handle", "cross_identity_dataset"):
        add("isolation", what, what=what, refs=("P0.RT.06", "P2.ST.01"))
    for what in ("weights", "full_training_state", "corrupt_checkpoint", "tied_weights", "server_snapshot"):
        add("checkpoint", what, what=what, refs=("P0.RC.01", "P0.RT.06"))
    add("migration", "manual_full_state", refs=("P0.RC.01", "P1.RC.05"), timeout=600)
    for what in ("idle_reconnect", "inflight_disconnect", "resume_expired"):
        add("disconnect", what, what=what)
    for what in ("pause_resume", "kill_session", "quota", "network_partition", "worker_kill"):
        add("faults", what, what=what, timeout=max(c["timeout_seconds"], c["recovery_deadline_seconds"] + 60),
            refs=("P1.RC.05", "P3.RC.02"))
    for what in ("allocation", "handle_churn", "session_limit", "graph_limit", "bounds_error", "bad_shape", "division_zero"):
        add("limits", what, what=what)
    for mode in ("eager", "capture"):
        add("soak", mode, mode=mode, timeout=c["soak_seconds"] + 180)
    for label, model, mode, residency in itertools.product(c["network_targets"], MODELS, ("eager", "capture"),
                                                          ("resident", "upload")):
        add("wan", f"{label}/{model}/{mode}/{residency}", "train", network=label, model=model, mode=mode,
            target="remote_cuda", check=False, batch=32, residency=residency, sync_every=1, optimizer="momentum")
    for label in c["network_targets"]:
        add("wan", f"{label}/link", "link", network=label)
    for what in ("forward_backward", "amp", "cuda_api"):
        add("shim", what, what=what)
    for what in ("c_matmul", "cpp_mlp"):
        add("c_api", what, what=what, timeout=900)
    for what in ("unknown_op", "invalid_graph", "unknown_handle", "frame_limit", "terminal_sanitization", "pin_mismatch"):
        add("protocol", what, what=what, refs=("P0.RT.03", "P2.ST.01"))
    for what in ("compile", "export", "torchscript", "activation_checkpoint", "custom_autograd", "gradient_hook",
                 "shared_parameters", "gradient_clip", "scheduler", "dataloader", "dataloader_spawn", "amp_overflow"):
        add("compatibility", what, what=what, timeout=600 if what=="compile" else None)
    ids = [x.id for x in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate benchmark IDs")
    return out
