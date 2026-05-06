"""
BSNL Fibre Fault Prediction System
Inference / Prediction Module
Usage: python src/predict.py
"""

import joblib, os
import numpy as np
import pandas as pd

MODEL_DIR = "outputs"

def load_models():
    models = {}
    for key, fname in [("occurrence", "model_fault_occurrence.pkl"),
                        ("severity",   "model_severity.pkl"),
                        ("type",       "model_fault_type.pkl")]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            print(f"[WARN] {fname} not found — run train.py first")
    return models


def predict_single(record: dict, models: dict) -> dict:
    """
    Predict fault occurrence, severity, and type for a single cable segment.

    Parameters
    ----------
    record : dict  — feature values (see FEAT_COLS in train.py)
    models : dict  — loaded models from load_models()

    Returns
    -------
    dict with keys: will_fault, fault_probability, severity_score, fault_type
    """
    features = [
        "cable_age_years","cable_length_km","cable_type_enc","depth_of_burial_m",
        "num_splices","num_joints","soil_type_enc","zone_enc",
        "weather_enc","temperature_c","humidity_pct","rainfall_mm","wind_speed_kmph","flood_zone_flag",
        "signal_strength_dbm","attenuation_db_per_km","packet_loss_pct","latency_ms",
        "bandwidth_utilization","traffic_load_gbps","peak_hour_flag",
        "last_maintenance_days","maintenance_count","prev_fault_count","days_since_last_fault",
        "distance_from_hub_km","num_customers_served",
        "construction_nearby","road_crossing_count","rodent_activity_index",
    ]
    X = pd.DataFrame([{f: record.get(f, 0) for f in features}])

    result = {}

    # ── 1. Fault occurrence ─────────────────────────────────────────────
    if "occurrence" in models:
        md   = models["occurrence"]
        m, s = md["model"], md["scaler"]
        Xs   = s.transform(X) if md["name"] in ["Logistic Regression","SVM (RBF)"] else X
        prob = m.predict_proba(Xs)[0][1]
        result["fault_probability"] = round(float(prob) * 100, 2)
        result["will_fault_30days"] = "YES ⚠" if prob > 0.5 else "NO  ✓"

    # ── 2. Severity score ───────────────────────────────────────────────
    if "severity" in models:
        md   = models["severity"]
        m, s = md["model"], md["scaler"]
        Xs   = s.transform(X) if md["name"] == "Ridge Regression" else X
        sev  = float(np.clip(m.predict(Xs)[0], 0, 100))
        result["severity_score"]    = round(sev, 2)
        result["severity_category"] = ("Critical" if sev > 75 else "High" if sev > 55 else "Medium" if sev > 35 else "Low")

    # ── 3. Fault type ───────────────────────────────────────────────────
    if "type" in models:
        md = models["type"]
        m, le = md["model"], md["label_encoder"]
        enc = int(m.predict(X)[0])
        result["predicted_fault_type"] = le.inverse_transform([enc])[0]

    return result


def batch_predict(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Run predictions on a DataFrame."""
    out = []
    for _, row in df.iterrows():
        out.append(predict_single(row.to_dict(), models))
    return pd.DataFrame(out)


def risk_report(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Generate a risk summary report."""
    preds = batch_predict(df, models)
    report = df[["zone","cable_age_years","distance_from_hub_km","num_customers_served"]].copy()
    report = pd.concat([report.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    report = report.sort_values("severity_score", ascending=False)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    models = load_models()
    if not models:
        print("No models found. Run: python src/train.py")
        exit(1)

    test_cases = [
        {
            "name": "High Risk — Old cable, storm, construction",
            "cable_age_years": 22, "cable_length_km": 30, "cable_type_enc": 2,
            "depth_of_burial_m": 0.4, "num_splices": 80, "num_joints": 40,
            "soil_type_enc": 1, "zone_enc": 0, "weather_enc": 2,
            "temperature_c": 40, "humidity_pct": 92, "rainfall_mm": 250,
            "wind_speed_kmph": 90, "flood_zone_flag": 1,
            "signal_strength_dbm": -28, "attenuation_db_per_km": 1.1,
            "packet_loss_pct": 12, "latency_ms": 130, "bandwidth_utilization": 95,
            "traffic_load_gbps": 90, "peak_hour_flag": 1,
            "last_maintenance_days": 900, "maintenance_count": 2, "prev_fault_count": 15,
            "days_since_last_fault": 10, "distance_from_hub_km": 25,
            "num_customers_served": 8000, "construction_nearby": 1,
            "road_crossing_count": 18, "rodent_activity_index": 9.5,
        },
        {
            "name": "Medium Risk — Moderate conditions",
            "cable_age_years": 10, "cable_length_km": 15, "cable_type_enc": 0,
            "depth_of_burial_m": 1.5, "num_splices": 35, "num_joints": 18,
            "soil_type_enc": 3, "zone_enc": 3, "weather_enc": 1,
            "temperature_c": 28, "humidity_pct": 65, "rainfall_mm": 40,
            "wind_speed_kmph": 30, "flood_zone_flag": 0,
            "signal_strength_dbm": -15, "attenuation_db_per_km": 0.5,
            "packet_loss_pct": 3, "latency_ms": 40, "bandwidth_utilization": 60,
            "traffic_load_gbps": 45, "peak_hour_flag": 0,
            "last_maintenance_days": 200, "maintenance_count": 10, "prev_fault_count": 5,
            "days_since_last_fault": 90, "distance_from_hub_km": 10,
            "num_customers_served": 2500, "construction_nearby": 0,
            "road_crossing_count": 5, "rodent_activity_index": 3.0,
        },
        {
            "name": "Low Risk — New cable, clear weather",
            "cable_age_years": 1.5, "cable_length_km": 5, "cable_type_enc": 1,
            "depth_of_burial_m": 2.5, "num_splices": 10, "num_joints": 5,
            "soil_type_enc": 3, "zone_enc": 4, "weather_enc": 0,
            "temperature_c": 22, "humidity_pct": 45, "rainfall_mm": 0,
            "wind_speed_kmph": 10, "flood_zone_flag": 0,
            "signal_strength_dbm": -6, "attenuation_db_per_km": 0.15,
            "packet_loss_pct": 0.2, "latency_ms": 5, "bandwidth_utilization": 30,
            "traffic_load_gbps": 10, "peak_hour_flag": 0,
            "last_maintenance_days": 30, "maintenance_count": 20, "prev_fault_count": 0,
            "days_since_last_fault": 365, "distance_from_hub_km": 2,
            "num_customers_served": 400, "construction_nearby": 0,
            "road_crossing_count": 1, "rodent_activity_index": 0.5,
        },
    ]

    print("\n" + "=" * 65)
    print("  BSNL FAULT PREDICTION — INFERENCE REPORT")
    print("=" * 65)

    for case in test_cases:
        name = case.pop("name")
        pred = predict_single(case, models)
        print(f"\n📍 Scenario: {name}")
        print(f"   Fault in 30 days   : {pred.get('will_fault_30days','N/A')}")
        print(f"   Fault Probability  : {pred.get('fault_probability','N/A')}%")
        print(f"   Severity Score     : {pred.get('severity_score','N/A')} / 100")
        print(f"   Severity Category  : {pred.get('severity_category','N/A')}")
        print(f"   Predicted Fault Type: {pred.get('predicted_fault_type','N/A')}")

    print("\n" + "=" * 65 + "\n")
