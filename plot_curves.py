"""Plot training losses and metrics from CSV or multi-fold logs."""
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
import re

NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
EPOCH = re.compile(r"\bEpoch\s*(\d+)\b", re.I)
FIELD = re.compile(r"([A-Za-z_][\w]*(?:\([\w]+\))?)\s*=\s*([^\s,|]+)")
FOLD = re.compile(r"\bfold\s*(\d+)\b", re.I)


def finite_number(value: str, context: str) -> float:
    value = value.strip()
    if not NUMBER.fullmatch(value):
        raise ValueError(f"{context}: invalid number {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context}: number must be finite")
    return number


def integer(value: str, context: str) -> int:
    if not re.fullmatch(r"\d+", value.strip()):
        raise ValueError(f"{context}: expected a non-negative integer")
    return int(value)


def validate(folds: dict) -> dict:
    if not folds or not any(folds.values()):
        raise ValueError("No training rows found")
    for fold, rows in folds.items():
        rows.sort(key=lambda row: row['epoch'])
        epochs = [row['epoch'] for row in rows]
        if len(set(epochs)) != len(epochs):
            raise ValueError(f"Fold {fold}: duplicate epochs; specify distinct fold IDs")
        columns = set(rows[0]) - {'epoch'}
        if not columns or not any('loss' in name.lower() or name in {'uni', 'multi', 'mmc'} for name in columns):
            raise ValueError(f"Fold {fold}: at least one loss column is required")
        if any(set(row) - {'epoch'} != columns for row in rows):
            raise ValueError(f"Fold {fold}: metric columns are inconsistent")
    return folds


def parse_csv(text: str) -> dict:
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    names = reader.fieldnames or []
    if len(names) != len(set(names)) or 'epoch' not in names:
        raise ValueError("CSV must have unique column names and an epoch column")
    metrics = [name for name in names if name not in {'epoch', 'fold'}]
    if not metrics:
        raise ValueError("CSV needs a numeric loss column")
    folds = {}
    for line, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV line {line}: column count does not match the header")
        fold = integer(row.get('fold', '0'), f"CSV line {line} fold")
        values = {'epoch': integer(row['epoch'], f"CSV line {line} epoch")}
        values.update({name: finite_number(row[name], f"CSV line {line} {name}") for name in metrics})
        folds.setdefault(fold, []).append(values)
    return validate(folds)


def parse_log(text: str) -> dict:
    folds, pending = {}, []
    current_fold = None

    def flush(fold):
        if pending:
            folds.setdefault(fold, []).extend(pending)
            pending.clear()

    for line_number, line in enumerate(text.splitlines(), 1):
        epoch = EPOCH.search(line)
        marker = FOLD.search(line)
        if epoch:
            fields = {key: finite_number(value, f"Log line {line_number} {key}") for key, value in FIELD.findall(line)}
            if not fields:
                raise ValueError(f"Log line {line_number}: epoch row has no numeric fields")
            pending.append({'epoch': int(epoch[1]), **fields})
        elif marker:
            fold = int(marker[1])
            if re.search(r"test_loss\s*=", line, re.I):
                # Original logs put fold IDs AFTER their epoch rows.
                if current_fold is not None and current_fold != fold:
                    raise ValueError(f"Log line {line_number}: header and trailer fold IDs disagree")
                flush(fold)
                current_fold = None
            else:
                flush(0 if current_fold is None else current_fold)
                current_fold = fold
        elif re.search(r"\bEpoch\b", line, re.I):
            raise ValueError(f"Log line {line_number}: malformed epoch row")
    flush(0 if current_fold is None else current_fold)
    return validate(folds)


def plot_folds(folds: dict, output: Path, overwrite: bool = False) -> list[Path]:
    validate(folds)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    paths = [output / f'fold_{fold}_curves.png' for fold in sorted(folds)]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("Output exists; choose another directory or use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for fold, path in zip(sorted(folds), paths):
        rows = folds[fold]
        columns = [name for name in rows[0] if name != 'epoch']
        losses = [name for name in columns if 'loss' in name.lower() or name in {'uni', 'multi', 'mmc'}]
        metrics = [name for name in columns if name not in losses]
        with plt.style.context('seaborn-v0_8-whitegrid'):
            fig, axes = plt.subplots(1, 2 if metrics else 1, figsize=(12 if metrics else 8, 4.8), squeeze=False)
            try:
                for axis, group, title in zip(axes[0], [losses, metrics], ['Training losses', 'Training metrics']):
                    for name in group:
                        axis.plot([row['epoch'] for row in rows], [row[name] for row in rows], label=name, linewidth=2)
                    axis.set(title=f'Fold {fold} · {title}', xlabel='Epoch', ylabel='Value')
                    axis.legend(frameon=False)
                fig.tight_layout()
                # Exclusive creation protects existing files even if another process races us.
                with path.open('wb' if overwrite else 'xb') as stream:
                    fig.savefig(stream, format='png', dpi=160)
            finally:
                plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--format', choices=['csv', 'log'])
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    try:
        text = args.input.read_text(encoding='utf-8-sig')
        kind = args.format or ('csv' if args.input.suffix.lower() == '.csv' else 'log')
        folds = (parse_csv if kind == 'csv' else parse_log)(text)
        for path in plot_folds(folds, args.output, args.overwrite):
            print(path)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
