#!/usr/bin/env python3
"""
Read Optuna study output and patch the best hyperparameters into a train script.

Usage:
    python src/training/extract_best_hpo.py \
        --yaml   outputs/hpo/hpo_trials.yaml \
        --script slurm/train.sh
"""


import argparse
import re
import sys
from pathlib import Path

import yaml


def load_best_params_from_yaml(yaml_path: Path) -> dict:
    """Load best trial params from HPO YAML output."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Expected structure:
    # best_trial:
    #   params:
    #     model: resnet50
    #     lr: 0.001
    #     momentum: 0.92
    #     weight_decay: 1e-4
    #   value: 0.812   (composite score)
    #   number: 23

    if "best_trial" in data:
        params = data["best_trial"]["params"]
        score  = data["best_trial"].get("value", None)
        trial  = data["best_trial"].get("number", None)
    elif "params" in data:
        # Flat format
        params = data["params"]
        score  = data.get("value", None)
        trial  = data.get("number", None)
    else:
        # Try treating top-level keys as params directly
        params = {k: v for k, v in data.items()
                  if k not in ("value", "number", "trials")}
        score  = data.get("value", None)
        trial  = data.get("number", None)

    print(f"[HPO] Best trial #{trial} | composite score: {score:.4f}" if score else
          f"[HPO] Best trial #{trial}")
    print(f"[HPO] Best params: {params}")
    return params


def patch_slurm_script(script_path: Path, params: dict) -> None:
    """
    Replace hyperparameter values in the training SLURM script.
    Looks for lines like:
        --lr           0.001 \
        --model        resnet50 \
        --momentum     0.9 \
        --weight_decay 1e-4 \
    and replaces the value with the HPO best.
    """
    text = script_path.read_text()
    original = text

    param_map = {
        "model":        ("--model",        str),
        "lr":           ("--lr",           lambda x: f"{float(x):.6g}"),
        "momentum":     ("--momentum",     lambda x: f"{float(x):.6g}"),
        "weight_decay": ("--weight_decay", lambda x: f"{float(x):.2e}"),
    }

    for key, (flag, fmt) in param_map.items():
        if key not in params:
            continue
        val = fmt(params[key])
        # Match: flag followed by whitespace + old_value + optional trailing space/backslash
        pattern = rf'({re.escape(flag)}\s+)(\S+)'
        replacement = rf'\g<1>{val}'
        new_text = re.sub(pattern, replacement, text)
        if new_text != text:
            print(f"[HPO] Patched {flag}: {val}")
            text = new_text
        else:
            print(f"[HPO] WARNING: could not find {flag} in script to patch")

    if text == original:
        print("[HPO] WARNING: No changes made to script — check flag names match")
    else:
        script_path.write_text(text)
        print(f"[HPO] Patched script: {script_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml",   required=True, help="HPO output YAML file")
    ap.add_argument("--script", required=True, help="SLURM train script to patch")
    args = ap.parse_args()

    yaml_path   = Path(args.yaml)
    script_path = Path(args.script)

    if not yaml_path.exists():
        print(f"[HPO] ERROR: YAML not found: {yaml_path}")
        sys.exit(1)

    if not script_path.exists():
        print(f"[HPO] ERROR: Script not found: {script_path}")
        sys.exit(1)

    params = load_best_params_from_yaml(yaml_path)
    patch_slurm_script(script_path, params)
    print("[HPO] Done.")


if __name__ == "__main__":
    main()
