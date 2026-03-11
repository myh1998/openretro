# -*- coding: utf-8 -*-
"""
将 OpenRetro/USPTO-50K 标准 split 数据转换为本项目可直接 load_from_disk 的 DatasetDict。

输出字段（每个 split）：
- reactants: 反应物（可包含试剂信息，取决于原数据格式）
- products: 目标产物 SMILES
- reaction_smiles: "reactants>>products"
- class: 反应类别（若数据中有）
- split: train / validation / test
- source_index: 原文件中的行号

使用示例：
python lora_with_llm/try_this/prepare_openretro_uspto50k_for_screener.py \
  --input_dir /path/to/openretro/uspto50k \
  --output_dir /path/to/uspto50k_hf_disk

随后在筛选脚本中指定：
--eval_dataset_mode uspto50k \
--uspto_local_data_dir /path/to/uspto50k_hf_disk
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict


SPLIT_CANDIDATES: Dict[str, List[str]] = {
    "train": [
        "train.csv", "train.tsv", "train.txt",
        "raw_train.csv", "raw_train.tsv", "raw_train.txt",
    ],
    "validation": [
        "valid.csv", "valid.tsv", "valid.txt",
        "val.csv", "val.tsv", "val.txt",
        "dev.csv", "dev.tsv", "dev.txt",
        "raw_valid.csv", "raw_valid.tsv", "raw_valid.txt",
        "raw_val.csv", "raw_val.tsv", "raw_val.txt",
    ],
    "test": [
        "test.csv", "test.tsv", "test.txt",
        "raw_test.csv", "raw_test.tsv", "raw_test.txt",
    ],
}


def _find_existing_file(input_dir: str, names: List[str]) -> Optional[str]:
    for n in names:
        p = os.path.join(input_dir, n)
        if os.path.isfile(p):
            return p
    return None


def _detect_delimiter(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".tsv"):
        return "\t"
    if lower.endswith(".csv"):
        return ","

    # txt/未知后缀：尝试 sniff
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
        return dialect.delimiter
    except Exception:
        # OpenRetro 常见是 tsv
        return "\t"


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _parse_rxn_smiles(rxn: str) -> Tuple[str, str]:
    rxn = (rxn or "").strip()
    if ">>" in rxn:
        lhs, rhs = rxn.split(">>", 1)
        return lhs.strip(), rhs.strip()
    raise ValueError(f"Invalid reaction SMILES (missing '>>'): {rxn[:120]}")


def _read_rows(path: str) -> List[dict]:
    delim = _detect_delimiter(path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in file: {path}")
    return rows


def _convert_split(path: str, split_name: str) -> Dataset:
    rows = _read_rows(path)
    cols = list(rows[0].keys())

    rxn_col = _pick_col(cols, [
        "reaction_smiles", "rxn_smiles", "canonical_rxn", "rxn", "reaction"
    ])
    src_col = _pick_col(cols, [
        "reactants", "src", "source", "input", "inputs", "reactant"
    ])
    tgt_col = _pick_col(cols, [
        "products", "product", "tgt", "target", "output", "outputs"
    ])
    class_col = _pick_col(cols, ["class", "reaction_class", "rxn_class"])  # optional

    if rxn_col is None and (src_col is None or tgt_col is None):
        raise ValueError(
            f"Cannot parse split file={path}. columns={cols}. "
            "Need either reaction_smiles-like column or (reactants/products)-like columns."
        )

    out = {
        "reactants": [],
        "products": [],
        "reaction_smiles": [],
        "class": [],
        "split": [],
        "source_index": [],
    }

    for i, r in enumerate(rows):
        if rxn_col is not None:
            reactants, products = _parse_rxn_smiles(str(r[rxn_col]))
        else:
            reactants = str(r[src_col]).strip()
            products = str(r[tgt_col]).strip()

        rxn = f"{reactants}>>{products}"
        klass = str(r[class_col]).strip() if class_col is not None else ""

        out["reactants"].append(reactants)
        out["products"].append(products)
        out["reaction_smiles"].append(rxn)
        out["class"].append(klass)
        out["split"].append(split_name)
        out["source_index"].append(i)

    return Dataset.from_dict(out)


def convert_openretro_uspto50k(input_dir: str, output_dir: str) -> None:
    train_file = _find_existing_file(input_dir, SPLIT_CANDIDATES["train"])
    valid_file = _find_existing_file(input_dir, SPLIT_CANDIDATES["validation"])
    test_file = _find_existing_file(input_dir, SPLIT_CANDIDATES["test"])

    if train_file is None:
        raise FileNotFoundError(
            "Could not find train split file in input_dir. "
            f"tried={SPLIT_CANDIDATES['train']}"
        )
    if valid_file is None:
        raise FileNotFoundError(
            "Could not find validation split file in input_dir. "
            f"tried={SPLIT_CANDIDATES['validation']}"
        )
    if test_file is None:
        raise FileNotFoundError(
            "Could not find test split file in input_dir. "
            f"tried={SPLIT_CANDIDATES['test']}"
        )

    dtrain = _convert_split(train_file, "train")
    dvalid = _convert_split(valid_file, "validation")
    dtest = _convert_split(test_file, "test")

    dsd = DatasetDict({"train": dtrain, "validation": dvalid, "test": dtest})
    os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)
    dsd.save_to_disk(output_dir)

    print("[ok] converted openretro uspto50k -> hf dataset dict")
    print({
        "input_dir": input_dir,
        "output_dir": output_dir,
        "train_rows": len(dtrain),
        "valid_rows": len(dvalid),
        "test_rows": len(dtest),
        "train_file": train_file,
        "valid_file": valid_file,
        "test_file": test_file,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="OpenRetro USPTO-50K 标准 split 数据目录")
    ap.add_argument("--output_dir", type=str, required=True, help="输出为 datasets.save_to_disk 目录")
    args = ap.parse_args()

    convert_openretro_uspto50k(args.input_dir, args.output_dir)