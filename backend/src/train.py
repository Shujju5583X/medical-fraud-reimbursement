import argparse, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from .feature_service import compute_features

def main(data_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(data_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    if "claim_date" not in df.columns:
        df["claim_date"] = pd.Timestamp.today().normalize()

    df["is_policy_active_tmp"] = (
        (pd.to_datetime(df["claim_date"]) >= pd.to_datetime(df["policy_start_date"], errors="coerce")) &
        (pd.to_datetime(df["claim_date"]) <= pd.to_datetime(df["policy_end_date"], errors="coerce"))
    ).astype(int)
    df["ready_label"] = ((df["is_policy_active_tmp"] == 1) & (pd.to_numeric(df["claim_count"], errors="coerce").fillna(0) <= 3)).astype(int)

    X, _ = compute_features(df)
    y = df["ready_label"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    base = GradientBoostingClassifier(random_state=42)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    yproba = clf.predict_proba(Xte)[:,1]
    try:
        auc = roc_auc_score(yte, yproba)
    except Exception:
        auc = None

    print("=== Readiness model ===")
    print(classification_report(yte, ypred, digits=3))
    print("ROC-AUC:", auc)

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(X)

    joblib.dump({"clf": clf, "iso": iso, "feature_columns": X.columns.tolist()},
                os.path.join(out_dir, "artifacts.joblib"))
    print(f"Saved {os.path.join(out_dir, 'artifacts.joblib')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV/XLSX with the 12 fields (+ claim_date optional)")
    ap.add_argument("--out", default="models", help="Output folder for artifacts.joblib")
    args = ap.parse_args()
    main(os.path.abspath(args.data), os.path.abspath(args.out))
