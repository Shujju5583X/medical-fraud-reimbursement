from __future__ import annotations
import math
import hashlib
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

REQUIRED_FIELDS = [
    "person_id","first_name","last_name","date_of_birth","age","email","phone",
    "policy_no","policy_type","policy_start_date","policy_end_date",
    "claim_count","last_claim_date","smoker","claim_date"
]

def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def _age_from_dob(dob: pd.Timestamp, on: pd.Timestamp) -> float | None:
    if pd.isna(dob) or pd.isna(on):
        return None
    return math.floor((on - dob).days / 365.25)

def normalize_name(first: str | None, last: str | None) -> str:
    f = (first or "").strip().lower()
    l = (last or "").strip().lower()
    return f"{f} {l}".strip()

def hash_contact(email: str | None, phone: str | None) -> str:
    raw = f"{(email or '').strip().lower()}|{(phone or '').strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def smoker_to_bin(v) -> int:
    s = str(v).strip().lower()
    return 1 if s in {"yes","true","1","y"} else 0

def compute_features(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    Computes model-ready features & rule hits from raw intake rows.
    Returns:
      X    : pandas DataFrame of features
      hits : per-row list of rules hit (strings)
    """
    df = df_in.copy()

    for c in ["date_of_birth","policy_start_date","policy_end_date","last_claim_date","claim_date"]:
        if c in df.columns:
            df[c] = _to_dt(df[c])

    df["normalized_name"] = (df["first_name"].fillna("") + " " + df["last_name"].fillna("")).str.lower().str.strip()
    df["contact_hash"] = [hash_contact(e, p) for e, p in zip(df.get("email"), df.get("phone"))]

    df["age_calc"] = [_age_from_dob(d, cd) for d, cd in zip(df["date_of_birth"], df["claim_date"])]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_vs_dob_delta"] = (df["age"] - pd.to_numeric(df["age_calc"], errors="coerce")).abs()

    df["is_policy_active"] = (
        (df["claim_date"] >= df["policy_start_date"]) &
        (df["claim_date"] <= df["policy_end_date"])
    ).astype(int)

    df["policy_tenure_days"] = (df["policy_end_date"] - df["policy_start_date"]).dt.days
    df["days_since_last_claim"] = (df["claim_date"] - df["last_claim_date"]).dt.days

    df["smoker_bin"] = [smoker_to_bin(x) for x in df["smoker"]]
    df["claim_count"] = pd.to_numeric(df["claim_count"], errors="coerce").fillna(0)

    tenure_years = (df["policy_tenure_days"].fillna(0) / 365.25).replace(0, np.nan)
    df["claim_count_rate"] = (df["claim_count"] / tenure_years).fillna(df["claim_count"])

    hits: List[List[str]] = []
    for _, r in df.iterrows():
        row_hits: List[str] = []

        if not isinstance(r.get("policy_no"), str) or len((r.get("policy_no") or "").strip()) == 0:
            row_hits.append("MISSING_POLICY_NO (HARD)")

        if r["is_policy_active"] == 0:
            if pd.isna(r["policy_start_date"]) or pd.isna(r["policy_end_date"]) or pd.isna(r["claim_date"]):
                row_hits.append("INVALID_POLICY_DATES (HARD)")
            else:
                if r["claim_date"] < r["policy_start_date"]:
                    row_hits.append("CLAIM_BEFORE_POLICY_START (HARD)")
                if r["claim_date"] > r["policy_end_date"]:
                    row_hits.append("CLAIM_AFTER_POLICY_END (HARD)")

        if pd.notna(r["age_vs_dob_delta"]) and r["age_vs_dob_delta"] > 1.0:
            row_hits.append("AGE_DOB_MISMATCH_GT_1Y (HARD)")

        if r["claim_count_rate"] >= 6:
            row_hits.append("HIGH_CLAIM_VELOCITY (SOFT)")

        if pd.notna(r["last_claim_date"]):
            if (not pd.isna(r["policy_start_date"]) and r["last_claim_date"] < r["policy_start_date"]) or \
               (not pd.isna(r["policy_end_date"]) and r["last_claim_date"] > r["policy_end_date"]):
                row_hits.append("LAST_CLAIM_OUTSIDE_POLICY (SOFT)")

        hits.append(row_hits)

    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"].fillna(df["age"].median())
    X["smoker_bin"] = df["smoker_bin"]
    X["is_policy_active"] = df["is_policy_active"]
    X["claim_count"] = df["claim_count"]
    X["days_since_last_claim"] = df["days_since_last_claim"].fillna(365)
    X["policy_tenure_days"] = df["policy_tenure_days"].fillna(0)
    X["claim_count_rate"] = df["claim_count_rate"]

    pt = pd.get_dummies(df["policy_type"].astype(str).str.lower().str.strip(), prefix="ptype", dummy_na=True)
    X = pd.concat([X, pt], axis=1)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, hits
