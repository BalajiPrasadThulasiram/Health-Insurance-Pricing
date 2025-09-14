import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import GammaRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import joblib

def decile_calibration(y_true, y_pred, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["decile"] = pd.qcut(df["y_pred"], q=n_bins, labels=False, duplicates="drop")
    return df.groupby("decile").agg(
        mean_pred=("y_pred","mean"),
        mean_actual=("y_true","mean"),
        count=("y_true","size")
    ).reset_index()

def main(csv_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    expected = {"age","sex","bmi","children","smoker","region","charges"}
    if not expected.issubset(df.columns):
        raise ValueError("Missing required columns in CSV")

    # Split features and target
    X, y = df.drop(columns=["charges"]), df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing
    num_features = ["age","bmi","children"]
    cat_features = ["sex","smoker","region"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    # Models
    glm = Pipeline([("pre", preprocessor), ("model", GammaRegressor(max_iter=10000))])
    gbr = Pipeline([("pre", preprocessor), ("model", GradientBoostingRegressor(random_state=42))])

    # Train & evaluate
    results = {}
    for name, model in {"GLM": glm, "GBR": gbr}.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(mean_squared_error(y_test, pred, squared=False)),
            "R2": float(r2_score(y_test, pred)),
        }

    # Save metrics
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print("Metrics:", json.dumps(results, indent=2))

    # Calibration table (GBR)
    def decile_table(y_true, y_pred):
        dfc = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        dfc["decile"] = pd.qcut(dfc["y_pred"], 10, labels=False, duplicates="drop")
        return dfc.groupby("decile")[["y_true","y_pred"]].mean().reset_index()

    gbr_pred = gbr.predict(X_test)
    cal = decile_table(y_test.to_numpy(), gbr_pred)
    cal.to_csv(out_dir / "gbr_calibration_by_decile.csv", index=False)

    # Permutation importance (GBR)
    perm = permutation_importance(gbr, X_test, y_test, n_repeats=8, random_state=42, n_jobs=1)
    pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)      .to_csv(out_dir / "gbr_permutation_importance.csv", index=False)

    # Smoker relativities (GLM baseline)
    glm_pred = glm.predict(X_test)
    rel = pd.DataFrame({"smoker": X_test["smoker"].values, "pred": glm_pred})
    rel_tbl = rel.groupby("smoker")["pred"].mean().rename("avg_pred").reset_index()
    if "no" in rel_tbl["smoker"].values:
        base = float(rel_tbl.loc[rel_tbl["smoker"]=="no","avg_pred"].iloc[0])
        rel_tbl["relativity_vs_no"] = rel_tbl["avg_pred"] / base
    rel_tbl.to_csv(out_dir / "glm_smoker_relativities.csv", index=False)

    # Save best model (GBR)
    joblib.dump(gbr, out_dir / "model_gbr.joblib")
    print(f"Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to insurance.csv")
    p.add_argument("--out", default="outputs", help="Output directory")
    args = p.parse_args()
    main(Path(args.csv), Path(args.out))