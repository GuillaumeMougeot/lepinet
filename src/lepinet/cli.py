"""Command-line interface: ``lepinet {train,test,predict,export}``.

Built with `typer`: typed options, generated ``--help``, and shell completion, while staying a thin
wrapper over the Python API (config files remain the source of truth for training/eval). Heavy
imports are done inside each command so ``lepinet --help`` stays instant.

Examples::

    lepinet train  --config configs/20260716_heads_global_independent_muon_5ep_oversample.yaml
    lepinet test   --model 'data/global/models/*-oversample-effnetv2s/*.pt' \\
                   --parquet data/global/....parquet --img-dir data/global/images \\
                   --out-dir data/global/preds --test-set 0
    lepinet predict --model model.pt image.jpg --topk 5
    lepinet export  --model model.pt --out-dir artifact/ --img-size 256
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=True, no_args_is_help=True,
                  help="Hierarchical image classification (lepinet).")


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False,
                                help="YAML training config."),
):
    """Train a model from a YAML config."""
    from .train import train_from_config

    run_dir = train_from_config(str(config))
    typer.echo(f"Run directory: {run_dir}")


@app.command()
def test(
    model: str = typer.Option(..., "--model", "-m", help="Checkpoint path (glob allowed)."),
    parquet: Path = typer.Option(..., "--parquet", "-p", exists=True, help="Evaluation parquet."),
    img_dir: Path = typer.Option(..., "--img-dir", "-i", help="Image root directory."),
    out_dir: Path = typer.Option(..., "--out-dir", "-o", help="Output directory."),
    eval_name: str = typer.Option("eval", help="Sub-folder name for this evaluation."),
    test_set: str = typer.Option("0", help="Fold id to evaluate ('0' = global test)."),
    min_img_per_spc: int = typer.Option(0, help="Minimum images per finest-level class."),
    batch_size: int = typer.Option(64),
    aug_img_size: int = typer.Option(460),
    img_size: int = typer.Option(256),
    num_workers: int | None = typer.Option(None),
):
    """Evaluate a checkpoint on a held-out fold (native metrics + mini_metrics-format predictions)."""
    from .test import evaluate

    evaluate(model_path=model, parquet_path=str(parquet), img_dir=str(img_dir), out_dir=str(out_dir),
             eval_name=eval_name, test_set=test_set, min_img_per_spc=min_img_per_spc,
             batch_size=batch_size, aug_img_size=aug_img_size, img_size=img_size, num_workers=num_workers)


@app.command()
def predict(
    images: list[str] = typer.Argument(..., help="Image file(s) or folder(s)."),
    model: str = typer.Option(..., "--model", "-m", help="Checkpoint path."),
    img_size: int = typer.Option(256),
    topk: int = typer.Option(5),
    batch_size: int = typer.Option(32),
    tta: bool = typer.Option(True, help="Test-time augmentation (4 flips)."),
):
    """Predict taxonomy for an image or folder (JSON to stdout)."""
    from .infer import predict as _predict

    results = _predict(model, images, img_size=img_size, tta=tta, topk=topk, batch_size=batch_size)
    typer.echo(json.dumps([r.as_dict() for r in results], indent=2))


@app.command()
def export(
    model: str = typer.Option(..., "--model", "-m", help="Checkpoint path (glob allowed)."),
    out_dir: Path = typer.Option(..., "--out-dir", "-o", help="Artifact output directory."),
    img_size: int = typer.Option(256),
    opset: int = typer.Option(17),
    dynamo: bool = typer.Option(False, help="Use the dynamo ONNX exporter (default: legacy)."),
    check: bool = typer.Option(True, help="Run the ONNX Runtime parity check."),
):
    """Export a checkpoint to ONNX + taxonomy.json (+ MANIFEST.json)."""
    from .export import export_onnx

    export_onnx(model, str(out_dir), img_size=img_size, opset=opset, check=check, dynamo=dynamo)


def main(argv=None) -> int:
    app(args=argv)
    return 0


if __name__ == "__main__":
    app()
