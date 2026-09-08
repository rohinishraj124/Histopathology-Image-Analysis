"""
Dataset loading, leakage checks, and train/validation splitting.
Extracted from notebooks/histopathology_analysis.ipynb — see that notebook for the
narrative explanation of each design choice (why no patient-level split is possible,
why the Kaggle test/ folder isn't used for evaluation, etc.).
"""
import hashlib
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def load_labels(labels_csv):
    return pd.read_csv(labels_csv)


def find_exact_duplicates(image_dir, ids, sample_limit=None):
    """MD5-hash based exact duplicate check. O(n) over the file list; pass
    sample_limit for a quick check, or None to scan every file."""
    ids_to_check = ids if sample_limit is None else ids[:sample_limit]
    seen, dupes = {}, []
    for img_id in ids_to_check:
        path = os.path.join(image_dir, img_id + ".tif")
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in seen:
            dupes.append((seen[h], img_id))
        else:
            seen[h] = img_id
    return dupes


def stratified_split(df, val_split=0.10, seed=42, sample_size_per_class=None):
    """Stratified id-level train/val split. NOTE: this dataset has no patient/slide
    identifiers, so this is patch-level, not patient-level, splitting — a known
    limitation documented in the notebook's Data Leakage section."""
    if sample_size_per_class:
        df_0 = df[df["label"] == 0].sample(sample_size_per_class, random_state=seed)
        df_1 = df[df["label"] == 1].sample(sample_size_per_class, random_state=seed)
        df = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)

    df_train, df_val = train_test_split(
        df, test_size=val_split, random_state=seed, stratify=df["label"]
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    assert set(df_train["id"]).isdisjoint(set(df_val["id"])), "Leakage: overlapping ids!"
    return df_train, df_val
