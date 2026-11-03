"""Command line interface.

Design constraints (AGENTS.md §8.2):
* every command works in PowerShell, cmd and POSIX shells;
* ``--json`` gives machine-readable output and stable exit codes;
* expensive or destructive actions support ``--dry-run``;
* commands that need torch fail with an actionable install hint instead of a traceback.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from ..errors import FoundryError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def _print(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    print(_human(payload))


def _human(payload: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(payload, dict):
        lines = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_human(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {value}")
        return "\n".join(lines)
    if isinstance(payload, list):
        if not payload:
            return f"{pad}(empty)"
        lines = []
        for item in payload:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_human(item, indent + 1))
            else:
                lines.append(f"{pad}- {item}")
        return "\n".join(lines)
    return f"{pad}{payload}"


def _foundry(args: argparse.Namespace):
    from ..services import Foundry

    return Foundry(args.home, probe_backends=not getattr(args, "no_probe", False))


# --------------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    payload = foundry.doctor()
    if args.json:
        _print(payload, True)
        return EXIT_OK
    hardware = payload["hardware"]
    print("HCS Neural Foundry - doctor")
    print(f"  workspace      : {payload['workspace']}")
    print(f"  schema version : {payload['database']['schema_version']}")
    print(f"  os             : {hardware['os']} {hardware['os_version']}")
    print(f"  cpu            : {hardware['cpu']['brand']} ({hardware['cpu']['logical_cores']} logical)")
    print(f"  ram            : {hardware['memory']['total_bytes'] / 1e9:.1f} GB")
    print(f"  disk free      : {hardware['memory']['disk_free_bytes'] / 1e9:.1f} GB")
    print("  backends:")
    for backend in hardware["backends"]:
        if backend["usable"]:
            marker = "usable"
        elif backend["declared"]:
            marker = "declared"
        else:
            marker = "absent"
        reason = f" - {backend['fallback_reason']}" if backend.get("fallback_reason") else ""
        print(f"    - {backend['kind']:<9} {marker:<9} {backend['name']}{reason}")
    print(f"  families       : {', '.join(f['id'] for f in payload['families'])}")
    print(f"  suites         : {', '.join(s['id'] for s in payload['suites'])}")
    for note in hardware["notes"]:
        print(f"  note: {note}")
    return EXIT_OK


def cmd_hardware(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    _print(foundry.hardware.model_dump(mode="json"), args.json)
    return EXIT_OK


def cmd_summary(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    _print(foundry.summary(), args.json)
    return EXIT_OK


def cmd_model_list(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    models = [model.model_dump(mode="json") for model in foundry.list_models()]
    _print(models, args.json)
    return EXIT_OK


def cmd_model_create(args: argparse.Namespace) -> int:
    from ..models.defaults import presets
    from ..models.registry import default_config
    from ..models.solver import parse_parameter_budget

    foundry = _foundry(args)
    if args.preset:
        presets_map = presets()
        if args.preset not in presets_map:
            raise FoundryError(
                f"unknown preset '{args.preset}'",
                hint=f"available: {sorted(presets_map)}",
            )
        config = presets_map[args.preset]["config"]
    elif args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    elif args.params:
        target = parse_parameter_budget(args.params)
        solution = foundry.solve(
            target,
            policy=args.policy,
            family=args.family,
            top_k=1,
        )
        best = solution["candidates"][0]
        config = best["config"]
        if not args.json:
            print(f"# solver: {solution['explanation']}")
    else:
        config = default_config(__import__("hcs_foundry.schemas", fromlist=["FamilyId"]).FamilyId(args.family))

    if args.dry_run:
        from ..models import build_model, parse_config

        parsed = parse_config(config)
        handle = build_model(parsed.model_dump(mode="json"))
        _print(
            {
                "dry_run": True,
                "family": parsed.family,
                "parameters": handle.parameter_count,
                "config": parsed.model_dump(mode="json"),
            },
            args.json,
        )
        return EXIT_OK

    model = foundry.create_model(args.name or "Model", config, notes=args.notes)
    payload = model.model_dump(mode="json")
    payload["human_parameters"] = f"{model.parameter_count:,}"
    _print(payload, args.json)
    return EXIT_OK


def cmd_model_show(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    payload = foundry.get_model(args.model).model_dump(mode="json")
    payload["parameters"] = foundry.parameter_report(args.model)
    payload["checkpoints"] = [
        record.model_dump(mode="json") for record in foundry.repo.list_checkpoints(args.model)
    ]
    _print(payload, args.json)
    return EXIT_OK


def cmd_model_estimate(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    _print(
        foundry.estimate(args.model, batch_size=args.batch_size, block_size=args.block_size), args.json
    )
    return EXIT_OK


def cmd_solve(args: argparse.Namespace) -> int:
    from ..models.solver import parse_parameter_budget

    foundry = _foundry(args)
    target = parse_parameter_budget(args.params)
    solution = foundry.solve(
        target, policy=args.policy, family=args.family, top_k=args.top, constraints=None
    )
    if args.json:
        _print(solution, True)
        return EXIT_OK
    print(solution["explanation"])
    print(f"search space: {solution['search_space_size']} designs")
    for index, candidate in enumerate(solution["candidates"], start=1):
        actual = candidate["actual_parameters"]
        config = candidate["config"]
        detail = ", ".join(
            f"{key}={config[key]}"
            for key in ("d_model", "n_layers", "n_heads", "ffn_dim")
            if key in config
        )
        print(
            f"  {index}. {actual:,} params (error {candidate['budget_error']:+,}, "
            f"{candidate['budget_error_pct']:.3f}%) [{detail}]"
        )
    return EXIT_OK


def cmd_weights_list(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    tensors = [tensor.model_dump(mode="json") for tensor in foundry.list_tensors(args.model)]
    if args.filter:
        tensors = [t for t in tensors if args.filter.lower() in t["name"].lower()]
    if not args.json:
        print(f"{'tensor':<44} {'shape':<18} {'numel':>9} {'mean':>10} {'std':>10} {'finite':>6}")
        for tensor in tensors:
            print(
                f"{tensor['name']:<44} {str(tuple(tensor['shape'])):<18} {tensor['numel']:>9,} "
                f"{tensor['mean']:>10.5f} {tensor['std']:>10.5f} {str(tensor['finite']):>6}"
            )
        return EXIT_OK
    _print(tensors, True)
    return EXIT_OK


def cmd_weights_slice(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    offset = tuple(int(v) for v in args.offset.split(",")) if args.offset else None
    shape = tuple(int(v) for v in args.shape.split(",")) if args.shape else None
    payload = foundry.tensor_slice(args.model, args.tensor, offset=offset, shape=shape)
    if args.json:
        _print(payload, True)
        return EXIT_OK
    print(f"{payload['name']} offset={payload['offset']} shape={payload['shape']} truncated={payload['truncated']}")
    values = payload["values"]
    width = 8
    for start in range(0, len(values), width):
        print("  " + " ".join(f"{value: .5f}" for value in values[start : start + width]))
    return EXIT_OK


def cmd_weights_histogram(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    payload = foundry.tensor_histogram(args.model, args.tensor, bins=args.bins)
    if args.json:
        _print(payload, True)
        return EXIT_OK
    peak = max(payload["counts"] or [1])
    for index, count in enumerate(payload["counts"]):
        left = payload["bins"][index]
        bar = "#" * int(60 * count / peak)
        print(f"{left:>10.5f} {count:>8} {bar}")
    return EXIT_OK


def cmd_weights_patch(args: argparse.Namespace) -> int:
    from ..agents import build_patch
    from ..schemas import PatchOperation

    foundry = _foundry(args)
    if args.file:
        payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
        if args.dry_run:
            _print({"dry_run": True, "patch": payload}, args.json)
            return EXIT_OK
    else:
        if not (args.tensor and args.op and args.params):
            raise FoundryError(
                "patch needs either --file or (--tensor, --op, --params)",
                hint='example: weights patch --model demo --tensor blocks.0.attn.q_proj.weight '
                '--op scale --params \'{"factor": 0.99}\'',
            )
        handle, record = foundry.handle_for(args.model, args.base or "latest")
        if record is None:
            raise FoundryError("no checkpoint available to patch")
        from ..checkpoints import flattened_state

        state = flattened_state(handle.module)["tensors"]
        tensor = state.get(args.tensor)
        if tensor is None:
            raise FoundryError(f"tensor '{args.tensor}' not found")
        patch = build_patch(
            model_id=args.model,
            architecture_hash=handle.architecture_hash,
            base_checkpoint_sha256=record.sha256,
            tensor_name=args.tensor,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            operation=PatchOperation(args.op),
            parameters=json.loads(args.params),
            created_by=args.author,
            note=args.note or "",
        )
        payload = patch.model_dump(mode="json")
        if args.dry_run:
            _print({"dry_run": True, "patch": payload}, args.json)
            return EXIT_OK

    result = foundry.apply_patch(args.model, payload)
    if not args.json:
        validation = result["validation"]
        print(f"applied patch {result['patch']['patch_id']} on {result['patch']['target_tensor']}")
        print(f"  changed elements : {validation['changed_elements']:,}")
        print(f"  max |delta|      : {validation['max_abs_delta']:.6g}")
        print(f"  norm change      : {validation['relative_norm_change']:+.4%}")
        print(f"  new checkpoint   : {result['checkpoint']['sha256'][:16]}…")
    else:
        _print(result, True)
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    from ..schemas import OptimizerConfig, TrainingConfig

    foundry = _foundry(args)
    config = TrainingConfig(
        model_id=args.model,
        dataset_id=args.dataset or "auto",
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        optimizer=OptimizerConfig(lr=args.lr, grad_clip=args.grad_clip),
        seed=args.seed,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        max_seconds=args.max_seconds,
    )
    if config.dataset_id == "auto":
        model = foundry.get_model(args.model)
        config.dataset_id = "tiny-text" if model.config.family == "transformer-lm" else "tiny-shapes"

    if args.dry_run:
        handle, _ = foundry.handle_for(args.model, None)

        spec = foundry.repo.get_dataset(config.dataset_id)
        from ..training.trainer import SupervisedTrainer

        trainer = SupervisedTrainer(
            foundry.workspace, foundry.repo, handle, config, dataset=spec, profile=foundry.hardware
        )
        estimate = trainer.estimate()
        loss = trainer.dry_run()
        _print(
            {
                "dry_run": True,
                "estimate": estimate.model_dump(mode="json"),
                "dry_run_loss": loss,
                "dataset": spec.model_dump(mode="json"),
            },
            args.json,
        )
        return EXIT_OK

    def progress(point: Any) -> None:
        if args.json:
            print(json.dumps({"step": point.step, "loss": point.loss, "lr": point.lr}), flush=True)
        else:
            message = f"step {point.step:>5} loss {point.loss:.5f} lr {point.lr:.2e}"
            if point.val_loss is not None:
                message += f" val_loss {point.val_loss:.5f}"
            print(message, flush=True)

    run = foundry.train(config, progress=progress)
    payload = run.model_dump(mode="json")
    if not args.json:
        print(f"run {run.id} finished with state '{run.state.value}'")
        for key in ("final_loss", "val_loss", "steps_completed", "seconds", "steps_per_second"):
            if key in run.summary:
                print(f"  {key}: {run.summary[key]}")
    else:
        _print(payload, True)
    return EXIT_OK if run.state.value in {"completed", "cancelled", "interrupted"} else EXIT_ERROR


def cmd_eval(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    report = foundry.evaluate(
        args.model,
        suite=args.suite,
        checkpoint=args.checkpoint,
        seed=args.seed,
        batches=args.batches,
    )
    if args.json:
        _print(report.model_dump(mode="json"), True)
    else:
        print(f"evaluation {report.run_id} state={report.state.value} suite={report.suite_id}")
        for metric in report.metrics:
            direction = "↑" if metric.higher_is_better else "↓"
            baseline = f" (baseline {metric.baseline:.4g})" if metric.baseline is not None else ""
            print(f"  {metric.name:<22} {metric.value:>12.6g} {direction}{baseline}")
        for note in report.notes:
            print(f"  note: {note}")
    return EXIT_OK if report.state.value == "completed" else EXIT_ERROR


def cmd_evolve(args: argparse.Namespace) -> int:
    from ..schemas import EvolutionConfig

    foundry = _foundry(args)
    config = EvolutionConfig(
        model_id=args.model,
        population=args.population,
        generations=args.generations,
        sigma=args.sigma,
        lr=args.lr,
        seed=args.seed,
        dataset_id="tiny-shapes",
    )
    payload = foundry.evolve(config)
    if args.json:
        _print(payload, True)
        return EXIT_OK
    print(f"archive cells : {len(payload['archive'])} (coverage {payload['archive_coverage']:.1%})")
    print(f"initial score : {payload['initial']['score']:.4f}")
    print(f"best score    : {payload['best']['score']:.4f}")
    baseline = payload.get("baseline", {})
    if baseline:
        print(f"gradient baseline (example-matched): {baseline.get('score')}")
    print(f"checkpoint    : {payload['checkpoint']['sha256'][:16]}…")
    print(f"note          : {payload['claim_boundary']}")
    return EXIT_OK


def cmd_generate(args: argparse.Namespace) -> int:
    from ..schemas import GeneratorConfig

    foundry = _foundry(args)
    corpus = [int(value) for value in args.corpus.split(",")] if args.corpus else [11, 22, 33, 44, 55, 66, 77, 88]
    config = GeneratorConfig(
        target_model_id=args.model,
        corpus=corpus,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    payload = foundry.generate(config, method=args.method)
    if args.json:
        _print(payload, True)
        return EXIT_OK
    print(f"weight generator ({payload['method']}, MLP target)")
    if "alignment" in payload:
        print(f"  alignment verified : {payload['alignment']['all_verified']}")
    for entry in payload.get("in_distribution", []):
        print(
            f"  seen task {entry['task_seed']:<6} generated {entry['generated_accuracy']:.3f} "
            f"vs trained {entry['trained_accuracy']:.3f} (distance {entry['relative_parameter_distance']:.2e})"
        )
    diversity = payload.get("sampling_diversity")
    if diversity:
        print(f"  sampling diversity : {diversity['relative_distance']:.2e} (collapsed={diversity['collapsed']})")
    print(f"  generated accuracy : {payload['generated_mean_accuracy']:.4f}")
    print(f"  random-init        : {payload['random_init_mean_accuracy']:.4f}")
    print(f"  gradient-trained   : {payload['gradient_trained_mean_accuracy']:.4f}")
    print(f"  claim boundary     : {payload['claim_boundary']}")
    print(f"  report             : {payload['report_path']}")
    return EXIT_OK


def cmd_rl(args: argparse.Namespace) -> int:
    from ..schemas import RlConfig

    foundry = _foundry(args)
    config = RlConfig(
        env_id="rule-bandit",
        steps=args.steps,
        group_size=args.group_size,
        batch_contexts=args.batch_contexts,
        lr=args.lr,
        kl_beta=args.kl_beta,
        seed=args.seed,
        context_dim=args.context_dim,
        n_actions=args.n_actions,
        eval_contexts=args.eval_contexts,
    )

    def progress(entry: dict[str, object]) -> None:
        if args.json:
            print(json.dumps(entry), flush=True)
        else:
            print(
                f"step {entry['step']:>5} loss {entry['loss']:>9} kl {entry['kl']:>8} "
                f"reward {entry['mean_reward']:.3f}",
                flush=True,
            )

    run = foundry.run_rl(config, progress=progress)
    if args.json:
        _print(run.model_dump(mode="json"), True)
        return EXIT_OK
    held_out = run.summary.get("held_out", {})
    print(f"run {run.id} finished with state '{run.state.value}'")
    print(f"  held-out accuracy : {held_out.get('policy_accuracy')}")
    print(f"  reference policy  : {held_out.get('reference_accuracy')}")
    print(f"  chance            : {held_out.get('chance_accuracy')}")
    print(f"  isolated verifier : {run.summary.get('isolated_verification', {}).get('state')}")
    print(f"  policy artifact   : {run.summary.get('artifact')}")
    print(f"  claim boundary    : {run.summary.get('claim_boundary')}")
    return EXIT_OK


def cmd_export(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    payload = foundry.export(args.model, fmt=args.format, checkpoint=args.checkpoint)
    if args.json:
        _print(payload, True)
        return EXIT_OK
    print(f"exported {args.format} -> {payload['path']}")
    print(f"  bytes          : {payload['bytes']:,}")
    print(f"  max |delta|    : {payload['max_abs_diff']:.3g} (tolerance {payload['tolerance']})")
    print(f"  parity         : {payload['parity_ok']}")
    print(f"  provider       : {payload['provider']}")
    print(f"  graph nodes    : {payload['graph']['node_count']}")
    return EXIT_OK


def cmd_history(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    payload = foundry.history(model_id=args.model, kind=args.kind, limit=args.limit)
    if args.json:
        _print(payload, True)
        return EXIT_OK
    for run in payload:
        metrics = run.get("summary", {}).get("metrics") or {}
        summary = ", ".join(f"{k}={v:.4g}" for k, v in list(metrics.items())[:3])
        print(f"{run['started_at']} {run['kind']:<12} {run['state']:<10} {run['model_id']:<20} {summary}")
    return EXIT_OK


def cmd_dataset_list(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    payload = [spec.model_dump(mode="json") for spec in foundry.repo.list_datasets()]
    _print(payload, args.json)
    return EXIT_OK


def cmd_proposal_list(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    payload = foundry.repo.list_proposals(args.model)
    _print(payload, args.json)
    return EXIT_OK


def cmd_proposal_create(args: argparse.Namespace) -> int:
    from ..schemas import PatchOperation

    foundry = _foundry(args)
    payload = foundry.propose_patch(
        args.model,
        agent_id=args.agent,
        goal=args.goal,
        hypothesis=args.hypothesis,
        tensor=args.tensor,
        operation=PatchOperation(args.op),
        parameters=json.loads(args.params),
        base_checkpoint=args.base or "latest",
    )
    _print(payload, args.json)
    return EXIT_OK


def cmd_proposal_approve(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    payload = foundry.approve_proposal(args.proposal, approver=args.approver)
    _print(payload, args.json)
    return EXIT_OK


def cmd_proposal_verify(args: argparse.Namespace) -> int:
    foundry = _foundry(args)
    payload = foundry.verify_proposal(args.proposal, suite=args.suite)
    _print(payload, args.json)
    return EXIT_OK if payload.get("state") == "completed" else EXIT_ERROR


def cmd_api(args: argparse.Namespace) -> int:
    from ..api import serve

    serve(host=args.host, port=args.port, home=args.home, token=args.token, reload=False)
    return EXIT_OK


def cmd_mcp(args: argparse.Namespace) -> int:
    from ..mcp_server import main as mcp_main

    return mcp_main(["--home", str(args.home or "")] if args.home else [])


def cmd_preset_list(args: argparse.Namespace) -> int:
    from ..models.defaults import presets

    payload = [
        {"name": name, "description": value["description"], "config": value["config"]}
        for name, value in presets().items()
    ]
    _print(payload, args.json)
    return EXIT_OK


# --------------------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcs-foundry",
        description="HCS Neural Foundry — local-first neural network workbench",
    )
    parser.add_argument("--home", default=None, help="workspace directory (default: ~/.hcs-foundry)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--no-probe", action="store_true", help="skip backend benchmarks (faster doctor)")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="environment and capability report")
    doctor.set_defaults(func=cmd_doctor)

    hardware = sub.add_parser("hardware", help="hardware profile as JSON")
    hardware.set_defaults(func=cmd_hardware)

    summary = sub.add_parser("summary", help="workspace summary")
    summary.set_defaults(func=cmd_summary)

    dataset = sub.add_parser("dataset", help="dataset commands")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_list = dataset_sub.add_parser("list", help="list datasets")
    dataset_list.set_defaults(func=cmd_dataset_list)

    model = sub.add_parser("model", help="model commands")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    model_list = model_sub.add_parser("list", help="list models")
    model_list.set_defaults(func=cmd_model_list)
    model_create = model_sub.add_parser("create", help="create a model")
    model_create.add_argument("--name", default=None)
    model_create.add_argument("--family", default="transformer-lm", choices=["mlp", "transformer-lm"])
    model_create.add_argument("--params", default=None, help="budget such as 250k, 1.2M")
    model_create.add_argument("--policy", default="closest", choices=["exact", "closest", "closest_under", "at_least"])
    model_create.add_argument("--preset", default=None)
    model_create.add_argument("--config", default=None, help="path to an architecture JSON file")
    model_create.add_argument("--notes", default="")
    model_create.add_argument("--dry-run", action="store_true")
    model_create.set_defaults(func=cmd_model_create)
    model_show = model_sub.add_parser("show", help="show a model with parameter breakdown")
    model_show.add_argument("--model", required=True)
    model_show.set_defaults(func=cmd_model_show)
    model_estimate = model_sub.add_parser("estimate", help="resource estimate")
    model_estimate.add_argument("--model", required=True)
    model_estimate.add_argument("--batch-size", type=int, default=8)
    model_estimate.add_argument("--block-size", type=int, default=32)
    model_estimate.set_defaults(func=cmd_model_estimate)

    solve = sub.add_parser("solve", help="solve a parameter budget")
    solve.add_argument("--params", required=True)
    solve.add_argument("--policy", default="closest", choices=["exact", "closest", "closest_under", "at_least"])
    solve.add_argument("--family", default="transformer-lm", choices=["mlp", "transformer-lm"])
    solve.add_argument("--top", type=int, default=5)
    solve.set_defaults(func=cmd_solve)

    preset = sub.add_parser("preset", help="built-in architecture presets")
    preset_sub = preset.add_subparsers(dest="preset_command", required=True)
    preset_list = preset_sub.add_parser("list")
    preset_list.set_defaults(func=cmd_preset_list)

    weights = sub.add_parser("weights", help="weight studio")
    weights_sub = weights.add_subparsers(dest="weights_command", required=True)
    weights_list = weights_sub.add_parser("list", help="list tensors")
    weights_list.add_argument("--model", required=True)
    weights_list.add_argument("--filter", default=None)
    weights_list.set_defaults(func=cmd_weights_list)
    weights_slice = weights_sub.add_parser("slice", help="bounded tensor read")
    weights_slice.add_argument("--model", required=True)
    weights_slice.add_argument("--tensor", required=True)
    weights_slice.add_argument("--offset", default=None)
    weights_slice.add_argument("--shape", default=None)
    weights_slice.set_defaults(func=cmd_weights_slice)
    weights_hist = weights_sub.add_parser("histogram", help="value histogram")
    weights_hist.add_argument("--model", required=True)
    weights_hist.add_argument("--tensor", required=True)
    weights_hist.add_argument("--bins", type=int, default=32)
    weights_hist.set_defaults(func=cmd_weights_histogram)
    weights_patch = weights_sub.add_parser("patch", help="apply an immutable weight patch")
    weights_patch.add_argument("--model", required=True)
    weights_patch.add_argument("--file", default=None)
    weights_patch.add_argument("--tensor", default=None)
    weights_patch.add_argument("--op", default=None)
    weights_patch.add_argument("--params", default=None)
    weights_patch.add_argument("--base", default=None)
    weights_patch.add_argument("--author", default="user")
    weights_patch.add_argument("--note", default="")
    weights_patch.add_argument("--dry-run", action="store_true")
    weights_patch.set_defaults(func=cmd_weights_patch)

    train = sub.add_parser("train", help="run supervised training")
    train.add_argument("--model", required=True)
    train.add_argument("--dataset", default=None)
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--block-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=3e-3)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--eval-every", type=int, default=50)
    train.add_argument("--checkpoint-every", type=int, default=100)
    train.add_argument("--max-seconds", type=int, default=600)
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("eval", help="run an evaluation suite")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--suite", default="auto")
    evaluate.add_argument("--checkpoint", default=None)
    evaluate.add_argument("--seed", type=int, default=1234)
    evaluate.add_argument("--batches", type=int, default=4)
    evaluate.set_defaults(func=cmd_eval)

    evolve = sub.add_parser("evolve", help="evolution strategies + MAP-Elites")
    evolve.add_argument("--model", required=True)
    evolve.add_argument("--population", type=int, default=16)
    evolve.add_argument("--generations", type=int, default=5)
    evolve.add_argument("--sigma", type=float, default=0.02)
    evolve.add_argument("--lr", type=float, default=0.05)
    evolve.add_argument("--seed", type=int, default=0)
    evolve.set_defaults(func=cmd_evolve)

    generate = sub.add_parser("generate", help="train a weight generator")
    generate.add_argument("--model", required=True)
    generate.add_argument("--method", default="hypernetwork", choices=["hypernetwork", "flow"])
    generate.add_argument("--corpus", default=None, help="comma separated task seeds")
    generate.add_argument("--epochs", type=int, default=200)
    generate.add_argument("--latent-dim", type=int, default=8)
    generate.add_argument("--seed", type=int, default=0)
    generate.set_defaults(func=cmd_generate)

    export = sub.add_parser("export", help="export to ONNX with parity check")
    export.add_argument("--model", required=True)
    export.add_argument("--format", default="onnx", choices=["onnx"])
    export.add_argument("--checkpoint", default="latest")
    export.set_defaults(func=cmd_export)

    history = sub.add_parser("history", help="list runs")
    history.add_argument("--model", default=None)
    history.add_argument("--kind", default=None)
    history.add_argument("--limit", type=int, default=25)
    history.set_defaults(func=cmd_history)

    proposal = sub.add_parser("proposal", help="agent proposals")
    proposal_sub = proposal.add_subparsers(dest="proposal_command", required=True)
    proposal_list = proposal_sub.add_parser("list")
    proposal_list.add_argument("--model", default=None)
    proposal_list.set_defaults(func=cmd_proposal_list)
    proposal_create = proposal_sub.add_parser("create")
    proposal_create.add_argument("--model", required=True)
    proposal_create.add_argument("--agent", default="agent:cli")
    proposal_create.add_argument("--goal", required=True)
    proposal_create.add_argument("--hypothesis", default="")
    proposal_create.add_argument("--tensor", required=True)
    proposal_create.add_argument("--op", required=True)
    proposal_create.add_argument("--params", required=True)
    proposal_create.add_argument("--base", default=None)
    proposal_create.set_defaults(func=cmd_proposal_create)
    proposal_approve = proposal_sub.add_parser("approve")
    proposal_approve.add_argument("--proposal", required=True)
    proposal_approve.add_argument("--approver", default="user")
    proposal_approve.set_defaults(func=cmd_proposal_approve)
    proposal_verify = proposal_sub.add_parser("verify")
    proposal_verify.add_argument("--proposal", required=True)
    proposal_verify.add_argument("--suite", default="weight-sanity")
    proposal_verify.set_defaults(func=cmd_proposal_verify)

    rl = sub.add_parser("rl", help="reinforcement learning with verifiable rewards")
    rl_sub = rl.add_subparsers(dest="rl_command", required=True)
    rl_grpo = rl_sub.add_parser("grpo", help="GRPO on the toy rule-bandit environment")
    rl_grpo.add_argument("--steps", type=int, default=200)
    rl_grpo.add_argument("--group-size", type=int, default=8)
    rl_grpo.add_argument("--batch-contexts", type=int, default=16)
    rl_grpo.add_argument("--lr", type=float, default=0.1)
    rl_grpo.add_argument("--kl-beta", type=float, default=0.02)
    rl_grpo.add_argument("--seed", type=int, default=0)
    rl_grpo.add_argument("--context-dim", type=int, default=6)
    rl_grpo.add_argument("--n-actions", type=int, default=4)
    rl_grpo.add_argument("--eval-contexts", type=int, default=256)
    rl_grpo.set_defaults(func=cmd_rl)

    api = sub.add_parser("api", help="local HTTP API")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8765)
    api.add_argument("--token", default=None)
    api.set_defaults(func=cmd_api)

    mcp = sub.add_parser("mcp", help="MCP server over stdio")
    mcp.set_defaults(func=cmd_mcp)

    return parser


GLOBAL_FLAGS = ("--json", "--no-probe")


def _hoist_global_flags(argv: list[str]) -> list[str]:
    """Let ``--json``/``--no-probe``/``--home`` appear anywhere, as users expect on the CLI."""
    globals_found: list[str] = []
    rest: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in GLOBAL_FLAGS:
            globals_found.append(token)
            index += 1
            continue
        if token == "--home" and index + 1 < len(argv):
            globals_found.extend([token, argv[index + 1]])
            index += 2
            continue
        if token.startswith("--home="):
            globals_found.append(token)
            index += 1
            continue
        rest.append(token)
        index += 1
    return globals_found + rest


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):  # pragma: no cover - non-reconfigurable stream
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    raw = _hoist_global_flags(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(raw)
    try:
        return int(args.func(args))
    except FoundryError as exc:
        payload = exc.to_dict()
        if getattr(args, "json", False):
            print(json.dumps({"error": payload}, indent=2))
        else:
            print(f"error [{payload['code']}]: {payload['message']}", file=sys.stderr)
            if payload.get("hint"):
                print(f"hint: {payload['hint']}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
