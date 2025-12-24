from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths (set these once)
# =========================
RUNS_SUMMARY_CSV = Path("./wandb_export/runs_summary.csv")
EXPORTS_DIR = Path("./wandb_export")          # your folder name
OUT_DIR = Path("./report_figures")            # figures go here
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CSV columns we rely on (present in your file)
# =========================
REQ_COLS = [
    "run.id",
    "run.name",
    "run.state",
    "summary.val_acc",
    "summary.val_loss",
    "summary.test_acc",
    "summary.test_loss",
    "summary._runtime",
    # config.* columns exist too, but graphs below focus on performance + convergence
    # because that’s what the assignment explicitly requires in Section 4.
]


def _assert_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "runs_summary.csv is missing expected columns:\n"
            + "\n".join(f"- {c}" for c in missing)
        )


def _load_summary() -> pd.DataFrame:
    df = pd.read_csv(RUNS_SUMMARY_CSV)
    _assert_columns(df, REQ_COLS)

    # finished only (consistent comparison)
    df = df[df["run.state"].astype(str).str.lower() == "finished"].copy()

    # numeric conversions
    for c in ["summary.val_acc", "summary.val_loss", "summary.test_acc", "summary.test_loss", "summary._runtime"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["summary.test_acc"])
    return df


def _choose_final_run_id(df: pd.DataFrame) -> str:
    # best finished run by test accuracy
    best_idx = df["summary.test_acc"].idxmax()
    return str(df.loc[best_idx, "run.id"])


def _safe_label(name: str, run_id: str, max_len: int = 28) -> str:
    label = (name or "").strip()
    if not label:
        label = run_id
    return label[:max_len]


def _save(fig_name: str) -> Path:
    out_path = OUT_DIR / fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def _plot_bar(df: pd.DataFrame, y_col: str, title: str, y_label: str, fig_name: str) -> None:
    df_plot = df.sort_values(y_col, ascending=False).copy()

    labels = [
        _safe_label(str(n), str(rid))
        for n, rid in zip(df_plot["run.name"].tolist(), df_plot["run.id"].tolist())
    ]

    plt.figure()
    plt.bar(labels, df_plot[y_col].values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(y_label)
    _save(fig_name)


def _plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, x_label: str, y_label: str, fig_name: str) -> None:
    df_plot = df[[x_col, y_col]].dropna().copy()

    plt.figure()
    plt.scatter(df_plot[x_col].values, df_plot[y_col].values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    _save(fig_name)


def _find_history_csv(run_id: str) -> Path:
    """
    Expected:
      wandb_exports/runs/<RUN_ID>/history.csv

    If not found, we also search under wandb_exports for any matching history.csv
    inside a folder named the run_id.
    """
    p1 = EXPORTS_DIR / "runs" / run_id / "history.csv"
    if p1.exists():
        return p1

    # fallback search
    matches = list(EXPORTS_DIR.glob(f"**/{run_id}/history.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find history.csv for run_id={run_id}\n"
        f"Tried:\n- {p1.resolve()}\n- {EXPORTS_DIR.resolve()}/**/{run_id}/history.csv"
    )


def _load_history(history_path: Path) -> pd.DataFrame:
    h = pd.read_csv(history_path)

    # We expect these columns from your W&B history export
    # If your history uses different names, rename them here once.
    required = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
    missing = [c for c in required if c not in h.columns]
    if missing:
        raise ValueError(
            f"history.csv is missing expected columns: {missing}\n"
            f"Found columns: {list(h.columns)}\n"
            f"File: {history_path}"
        )

    for c in required:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h.dropna(subset=["epoch"])
    return h.sort_values("epoch")


def _plot_training_curves(history: pd.DataFrame, run_id: str) -> None:
    x = history["epoch"].values

    # Loss curves (Figure A.1)
    plt.figure()
    plt.plot(x, history["train_loss"].values, label="Train Loss")
    plt.plot(x, history["val_loss"].values, label="Validation Loss")
    plt.title(f"Loss vs Epoch (Final Model: {run_id})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    _save("Figure_A1_Loss_Curves_Final.png")

    # Accuracy curves (Figure A.2)
    plt.figure()
    plt.plot(x, history["train_acc"].values, label="Train Accuracy")
    plt.plot(x, history["val_acc"].values, label="Validation Accuracy")
    plt.title(f"Accuracy vs Epoch (Final Model: {run_id})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    _save("Figure_A2_Accuracy_Curves_Final.png")


def main() -> None:
    df = _load_summary()
    final_run_id = _choose_final_run_id(df)

    print(f"[info] Finished runs in CSV: {len(df)}")
    print(f"[info] Selected final run_id (best test_acc): {final_run_id}")

    # =========================
    # Section 4a / 4c graphs (from runs_summary.csv)
    # =========================
    _plot_bar(
        df=df,
        y_col="summary.test_acc",
        title="Final Test Accuracy per Run",
        y_label="Test Accuracy",
        fig_name="Figure_4a_TestAccuracy_PerRun.png",
    )

    _plot_bar(
        df=df,
        y_col="summary.val_acc",
        title="Final Validation Accuracy per Run",
        y_label="Validation Accuracy",
        fig_name="Figure_4a_ValAccuracy_PerRun.png",
    )

    _plot_bar(
        df=df,
        y_col="summary._runtime",
        title="Convergence Time per Run (Runtime)",
        y_label="Runtime (seconds)",
        fig_name="Figure_4a_RuntimeSeconds_PerRun.png",
    )

    _plot_bar(
        df=df,
        y_col="summary.test_loss",
        title="Final Test Loss per Run",
        y_label="Test Loss",
        fig_name="Figure_4a_TestLoss_PerRun.png",
    )

    _plot_bar(
        df=df,
        y_col="summary.val_loss",
        title="Final Validation Loss per Run",
        y_label="Validation Loss",
        fig_name="Figure_4a_ValLoss_PerRun.png",
    )

    _plot_scatter(
        df=df,
        x_col="summary._runtime",
        y_col="summary.test_acc",
        title="Runtime vs Test Accuracy (Trade-off)",
        x_label="Runtime (seconds)",
        y_label="Test Accuracy",
        fig_name="Figure_4c_Runtime_vs_TestAcc.png",
    )

    # =========================
    # Section 4b graphs (from history.csv of final run)
    # =========================
    history_path = _find_history_csv(final_run_id)
    print(f"[info] Using history.csv: {history_path.resolve()}")
    history = _load_history(history_path)
    _plot_training_curves(history, final_run_id)

    print(f"\nDone. All figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
