from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import wandb


@dataclass(frozen=True)
class WandbProjectRef:
    entity: str
    project: str

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}"


def _safe_json(obj: Any) -> Any:
    """
    Make an object JSON-serializable (best-effort).
    """
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run_dir(out_dir: Path, run_id: str) -> Path:
    return out_dir / "runs" / run_id


def list_runs(project_ref: WandbProjectRef, api: wandb.Api, include_finished_only: bool) -> List[wandb.apis.public.Run]:
    runs = api.runs(project_ref.path)
    if include_finished_only:
        runs = [r for r in runs if (r.state or "").lower() == "finished"]
    return runs


def export_run_metadata(run: wandb.apis.public.Run, run_out: Path) -> Dict[str, Any]:
    """
    Save run metadata: config + summary + basic identity fields.
    Returns a flattened dict suitable for aggregation.
    """
    _ensure_dir(run_out)

    identity = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "entity": run.entity,
        "project": run.project,
        "created_at": str(run.created_at),
        "url": run.url,
    }

    config = {k: _safe_json(v) for k, v in dict(run.config).items()}
    summary = {k: _safe_json(v) for k, v in dict(run.summary).items()}

    (run_out / "identity.json").write_text(json.dumps(identity, indent=2, ensure_ascii=False))
    (run_out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (run_out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Flatten for a single-row summary table
    flat: Dict[str, Any] = {}
    flat.update({f"run.{k}": v for k, v in identity.items()})
    flat.update({f"config.{k}": v for k, v in config.items()})
    flat.update({f"summary.{k}": v for k, v in summary.items()})

    # Helpful “convergence time” fields commonly present in W&B
    # _runtime is usually seconds; _step is the last logged step; _timestamp is unix time.
    # These are in summary for many runs.
    for k in ["_runtime", "_step", "_timestamp"]:
        if k in summary:
            flat[f"summary.{k}"] = summary[k]

    return flat


def export_run_history(
    run: wandb.apis.public.Run,
    run_out: Path,
    keys: Optional[List[str]],
    samples: int,
) -> pd.DataFrame:
    """
    Export time-series metrics.
    - If keys is None: export everything W&B returns in history (can be large).
    - If keys provided: export only those columns if present.
    """
    _ensure_dir(run_out)

    # W&B history() returns a pandas DataFrame
    hist = run.history(keys=keys, samples=samples)
    if hist is None or hist.empty:
        (run_out / "history.csv").write_text("")  # create empty file for consistency
        return pd.DataFrame()

    hist.to_csv(run_out / "history.csv", index=False)
    return hist


def download_run_files(run: wandb.apis.public.Run, run_out: Path) -> List[str]:
    """
    Downloads files that exist on W&B for the run.
    Note: if you didn't wandb.save() plots/models in the notebook, older runs may not have these files.
    """
    files_dir = run_out / "files"
    _ensure_dir(files_dir)

    downloaded: List[str] = []
    for f in run.files():
        # Avoid huge / noisy system files unless you explicitly need them
        # You can remove this filter if you want everything.
        if f.name in {"output.log"}:
            continue
        local_path = f.download(root=str(files_dir), replace=True).name
        downloaded.append(local_path)

    (run_out / "downloaded_files.json").write_text(json.dumps(downloaded, indent=2, ensure_ascii=False))
    return downloaded


def export_project(
    project_ref: WandbProjectRef,
    out_dir: Path,
    include_finished_only: bool,
    history_keys: Optional[List[str]],
    history_samples: int,
    download_files: bool,
) -> None:
    _ensure_dir(out_dir)

    api = wandb.Api()  # uses WANDB_API_KEY or wandb login
    runs = list_runs(project_ref, api, include_finished_only)

    # Sort newest first (handy for "latest run" report)
    runs = sorted(runs, key=lambda r: r.created_at, reverse=True)

    aggregated_rows: List[Dict[str, Any]] = []

    for run in runs:
        run_out = _run_dir(out_dir, run.id)
        print(f"[export] {run.name} ({run.id}) state={run.state}")

        flat = export_run_metadata(run, run_out)
        aggregated_rows.append(flat)

        export_run_history(run, run_out, history_keys, history_samples)

        if download_files:
            download_run_files(run, run_out)

    df = pd.DataFrame(aggregated_rows)

    # Save a compact “runs table” (good for report parameter-comparison tables)
    df.to_csv(out_dir / "runs_summary.csv", index=False)

    # Also save a pretty Excel file (often easier for tables in the report)
    try:
        df.to_excel(out_dir / "runs_summary.xlsx", index=False)
    except Exception:
        pass

    print(f"\nDone. Exported {len(runs)} runs to: {out_dir.resolve()}")
    print(f"- runs_summary.csv / .xlsx contain config+summary per run")
    print(f"- each run folder contains config.json, summary.json, history.csv, etc.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="orisin-ben-gurion-university-of-the-negev")
    parser.add_argument("--project", type=str, default="facial-recognition")
    parser.add_argument("--out", type=str, default="./wandb_export")

    parser.add_argument("--finished-only", action="store_true", help="Export only finished runs")
    parser.add_argument("--download-files", action="store_true", help="Download run files stored on W&B")

    # Keys you likely log in your notebook per epoch
    parser.add_argument(
        "--history-keys",
        type=str,
        default="epoch,train_loss,train_acc,val_loss,val_acc,learning_rate",
        help="Comma-separated list; set to 'ALL' to export everything",
    )
    parser.add_argument(
        "--history-samples",
        type=int,
        default=10000,
        help="Max history rows to pull per run (increase if needed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)

    if args.history_keys.strip().upper() == "ALL":
        history_keys = None
    else:
        history_keys = [k.strip() for k in args.history_keys.split(",") if k.strip()]

    project_ref = WandbProjectRef(entity=args.entity, project=args.project)

    export_project(
        project_ref=project_ref,
        out_dir=out_dir,
        include_finished_only=args.finished_only,
        history_keys=history_keys,
        history_samples=args.history_samples,
        download_files=args.download_files,
    )


if __name__ == "__main__":
    main()
