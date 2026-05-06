"""
BSNL Fibre Fault Prediction — ML Training Pipeline
Three ML tasks:
  1. Fault Occurrence Prediction  (Binary Classification)
  2. Fault Severity Score         (Regression)
  3. Fault Type Classification    (Multiclass)
Author: Bikash (enggbikash15)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve,
                              mean_squared_error, mean_absolute_error, r2_score)
from sklearn.linear_model    import LogisticRegression, Ridge
from sklearn.ensemble        import (RandomForestClassifier, RandomForestRegressor,
                                      GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm             import SVC
from xgboost                 import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

FEAT_COLS = [
    "cable_age_years","cable_length_km","cable_type_enc","depth_of_burial_m",
    "num_splices","num_joints","soil_type_enc","zone_enc",
    "weather_enc","temperature_c","humidity_pct","rainfall_mm",
    "wind_speed_kmph","flood_zone_flag",
    "signal_strength_dbm","attenuation_db_per_km","packet_loss_pct",
    "latency_ms","bandwidth_utilization","traffic_load_gbps","peak_hour_flag",
    "last_maintenance_days","maintenance_count","prev_fault_count",
    "days_since_last_fault","distance_from_hub_km","num_customers_served",
    "construction_nearby","road_crossing_count","rodent_activity_index",
]
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

BG   = "#0a0e1a"; PANEL = "#111827"; BORDER = "#1e2d45"
ACCENT="#00c6ff"; WARN="#ffab00"; DANGER="#ff1744"; SUCCESS="#00e676"; PURPLE="#7c4dff"
PALETTE=[ACCENT, WARN, DANGER, SUCCESS, PURPLE, "#ff4081","#00bcd4","#8bc34a"]

def _style(fig, *axes):
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.tick_params(colors="#607899")
        ax.xaxis.label.set_color("#e8f0fe")
        ax.yaxis.label.set_color("#e8f0fe")
        ax.title.set_color(ACCENT)

# ── Load ──────────────────────────────────────────────────────────────────────
def load_data(path="data/bsnl_fault_dataset.csv"):
    if not os.path.exists(path):
        sys.path.insert(0,"src")
        from generate_data import generate_bsnl_dataset
        os.makedirs("data", exist_ok=True)
        df = generate_bsnl_dataset(5000)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    print(f"[OK]  Dataset loaded: {df.shape}")
    return df

# ── EDA ───────────────────────────────────────────────────────────────────────
def plot_eda(df):
    print("\n[EDA] Generating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    _style(fig, *axes.flatten())

    # Severity distribution
    ax = axes[0,0]
    ax.hist(df["severity_score"], bins=50, color=ACCENT, edgecolor="none", alpha=0.85)
    ax.axvline(df["severity_score"].mean(), color=DANGER, lw=1.8, linestyle="--",
               label=f"Mean={df['severity_score'].mean():.1f}")
    ax.set_title("Fault Severity Distribution"); ax.set_xlabel("Severity Score"); ax.legend(labelcolor="#e8f0fe")

    # Fault type counts
    ax = axes[0,1]
    vc = df["fault_type"].value_counts()
    bars = ax.barh(vc.index, vc.values, color=PALETTE[:len(vc)])
    ax.set_title("Fault Type Frequency"); ax.set_xlabel("Count")
    for b, v in zip(bars, vc.values):
        ax.text(b.get_width()+15, b.get_y()+b.get_height()/2, str(v), va="center", fontsize=8, color="#e8f0fe")

    # Fault probability by zone
    ax = axes[0,2]
    zv = df.groupby("zone")["will_fault_30days"].mean().sort_values(ascending=False)*100
    ax.bar(range(len(zv)), zv.values, color=WARN, edgecolor="none", alpha=0.85)
    ax.set_xticks(range(len(zv))); ax.set_xticklabels(zv.index, rotation=30, ha="right", fontsize=8, color="#607899")
    ax.set_title("Fault Rate (%) by Zone"); ax.set_ylabel("Fault %")

    # Cable age vs severity
    ax = axes[1,0]
    sc = ax.scatter(df["cable_age_years"], df["severity_score"],
                    c=df["will_fault_30days"], cmap="RdYlGn_r", alpha=0.3, s=8)
    plt.colorbar(sc, ax=ax, label="Will Fault")
    ax.set_title("Cable Age vs Severity"); ax.set_xlabel("Cable Age (yrs)"); ax.set_ylabel("Severity Score")

    # Signal strength vs packet loss
    ax = axes[1,1]
    sc2 = ax.scatter(df["signal_strength_dbm"], df["packet_loss_pct"],
                     c=df["severity_score"], cmap="plasma", alpha=0.3, s=8)
    plt.colorbar(sc2, ax=ax, label="Severity")
    ax.set_title("Signal Strength vs Packet Loss"); ax.set_xlabel("Signal (dBm)"); ax.set_ylabel("Packet Loss %")

    # Correlation heatmap
    ax = axes[1,2]
    cols = ["cable_age_years","last_maintenance_days","prev_fault_count",
            "attenuation_db_per_km","packet_loss_pct","rainfall_mm",
            "rodent_activity_index","construction_nearby","severity_score","will_fault_30days"]
    corr = df[cols].corr()
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.3, linecolor=BORDER, annot_kws={"size":6},
                cbar_kws={"shrink":0.7})
    ax.set_title("Feature Correlation Heatmap")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7, color="#607899")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7, color="#607899")

    fig.suptitle("BSNL Fibre Fault Dataset — Exploratory Data Analysis",
                 fontsize=15, fontweight="bold", color=ACCENT, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUT}/eda_analysis.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("[OK]  eda_analysis.png")

# ── Task 1: Binary Classification ────────────────────────────────────────────
def train_fault_occurrence(df):
    print("\n[Task 1] Fault Occurrence — Binary Classification")
    X = df[FEAT_COLS]; y = df["will_fault_30days"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr); Xs_te = sc.transform(X_te)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=500, random_state=42), True),
        "Random Forest":       (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), False),
        "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=150, random_state=42), False),
        "XGBoost":             (XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0), False),
        "SVM (RBF)":           (SVC(probability=True, random_state=42), True),
    }

    results = {}
    for name, (m, use_scaled) in models.items():
        Xtr = Xs_tr if use_scaled else X_tr
        Xte = Xs_te if use_scaled else X_te
        m.fit(Xtr, y_tr)
        yp = m.predict(Xte); ypr = m.predict_proba(Xte)[:,1]
        acc = accuracy_score(y_te, yp); auc = roc_auc_score(y_te, ypr)
        results[name] = {"model":m,"scaler":sc if use_scaled else None,"acc":acc,"auc":auc,"y_pred":yp,"y_prob":ypr,"scaled":use_scaled}
        print(f"   {name:25s}  Acc={acc:.4f}  AUC={auc:.4f}")

    best = max(results, key=lambda k: results[k]["auc"])
    print(f"   ★ Best: {best} (AUC={results[best]['auc']:.4f})")

    # Confusion matrix
    cm = confusion_matrix(y_te, results[best]["y_pred"])
    fig, ax = plt.subplots(figsize=(6,5)); _style(fig, ax)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Fault","Fault"], yticklabels=["No Fault","Fault"],
                linewidths=0.3, linecolor=BORDER, annot_kws={"size":14})
    ax.set_title(f"Confusion Matrix — {best}"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(f"{OUT}/task1_confusion.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(8,6)); _style(fig, ax)
    for (name, res), col in zip(results.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0,1],[0,1],"--", color="#607899", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Fault Occurrence"); ax.legend(fontsize=9, labelcolor="#e8f0fe")
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(f"{OUT}/task1_roc.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close()

    # Feature importance
    rf = results["Random Forest"]["model"]
    imp = rf.feature_importances_; idx = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(10,6)); _style(fig, ax)
    ax.barh([FEAT_COLS[i] for i in idx], imp[idx], color=plt.cm.plasma(np.linspace(0.2,0.9,15)))
    ax.set_title("Feature Importance — Fault Occurrence (Random Forest)")
    plt.tight_layout(); plt.savefig(f"{OUT}/task1_feature_importance.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print("[OK]  task1_confusion.png | task1_roc.png | task1_feature_importance.png")

    joblib.dump({"model":results[best]["model"],"scaler":results[best]["scaler"],
                 "features":FEAT_COLS,"task":"binary","name":best,
                 "scaled":results[best]["scaled"],
                 "metrics":{"accuracy":results[best]["acc"],"auc":results[best]["auc"]}},
                f"{OUT}/model_fault_occurrence.pkl")
    return results, y_te

# ── Task 2: Severity Regression ───────────────────────────────────────────────
def train_severity(df):
    print("\n[Task 2] Fault Severity Score — Regression")
    X = df[FEAT_COLS]; y = df["severity_score"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); Xs_tr = sc.fit_transform(X_tr); Xs_te = sc.transform(X_te)

    models = {
        "Ridge Regression":  (Ridge(alpha=1.0), True),
        "Random Forest":     (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=150, random_state=42), False),
        "XGBoost":           (XGBRegressor(n_estimators=200, random_state=42, verbosity=0), False),
    }

    results = {}
    for name, (m, use_scaled) in models.items():
        Xtr = Xs_tr if use_scaled else X_tr; Xte = Xs_te if use_scaled else X_te
        m.fit(Xtr, y_tr); yp = m.predict(Xte)
        rmse = np.sqrt(mean_squared_error(y_te, yp))
        mae  = mean_absolute_error(y_te, yp); r2 = r2_score(y_te, yp)
        results[name] = {"model":m,"rmse":rmse,"mae":mae,"r2":r2,"y_pred":yp}
        print(f"   {name:25s}  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")

    best = max(results, key=lambda k: results[k]["r2"])
    print(f"   ★ Best: {best} (R²={results[best]['r2']:.4f})")

    # Actual vs predicted
    fig, ax = plt.subplots(figsize=(8,6)); _style(fig, ax)
    ax.scatter(y_te, results[best]["y_pred"], alpha=0.25, s=8, color=ACCENT)
    mn,mx = min(y_te.min(), results[best]["y_pred"].min()), max(y_te.max(), results[best]["y_pred"].max())
    ax.plot([mn,mx],[mn,mx],"--", color=DANGER, lw=1.5, label="Perfect Fit")
    ax.set_xlabel("Actual Severity"); ax.set_ylabel("Predicted Severity")
    ax.set_title(f"Actual vs Predicted — {best}"); ax.legend(labelcolor="#e8f0fe"); ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(f"{OUT}/task2_actual_vs_pred.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close()

    # Model comparison
    names = list(results.keys())
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5)); _style(fig, ax1, ax2)
    ax1.bar(names, [results[n]["rmse"] for n in names], color=PALETTE[:4])
    ax1.set_title("RMSE Comparison (lower=better)"); ax1.set_xticklabels(names, rotation=20, ha="right", color="#607899")
    ax2.bar(names, [results[n]["r2"] for n in names], color=PALETTE[:4]); ax2.set_ylim(0,1)
    ax2.set_title("R² Score Comparison (higher=better)"); ax2.set_xticklabels(names, rotation=20, ha="right", color="#607899")
    plt.tight_layout(); plt.savefig(f"{OUT}/task2_model_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print("[OK]  task2_actual_vs_pred.png | task2_model_comparison.png")

    joblib.dump({"model":results[best]["model"],"scaler":sc,"features":FEAT_COLS,
                 "task":"regression","name":best,
                 "metrics":{"rmse":results[best]["rmse"],"r2":results[best]["r2"]}},
                f"{OUT}/model_severity.pkl")
    return results, y_te

# ── Task 3: Fault Type Multiclass ─────────────────────────────────────────────
def train_fault_type(df):
    print("\n[Task 3] Fault Type Classification — Multiclass")
    X = df[FEAT_COLS]; le = LabelEncoder(); y = le.fit_transform(df["fault_type"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Random Forest":     RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "XGBoost":           XGBClassifier(n_estimators=200, random_state=42, eval_metric="mlogloss", verbosity=0),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr); yp = m.predict(X_te)
        acc = accuracy_score(y_te, yp)
        results[name] = {"model":m,"acc":acc,"y_pred":yp}
        print(f"   {name:25s}  Acc={acc:.4f}")

    best = max(results, key=lambda k: results[k]["acc"])
    print(f"   ★ Best: {best} (Acc={results[best]['acc']:.4f})")
    print(f"\n{classification_report(y_te, results[best]['y_pred'], target_names=le.classes_)}")

    # Confusion matrix
    cm = confusion_matrix(y_te, results[best]["y_pred"])
    fig, ax = plt.subplots(figsize=(10,8)); _style(fig, ax)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.3, linecolor=BORDER, annot_kws={"size":9})
    ax.set_title(f"Confusion Matrix — Fault Type ({best})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8, color="#607899")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, color="#607899")
    plt.tight_layout(); plt.savefig(f"{OUT}/task3_confusion.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close()

    # Feature importance
    rf = results["Random Forest"]["model"]
    imp = rf.feature_importances_; idx = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(10,6)); _style(fig, ax)
    ax.barh([FEAT_COLS[i] for i in idx], imp[idx], color=plt.cm.viridis(np.linspace(0.2,0.9,15)))
    ax.set_title("Feature Importance — Fault Type (Random Forest)")
    plt.tight_layout(); plt.savefig(f"{OUT}/task3_feature_importance.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print("[OK]  task3_confusion.png | task3_feature_importance.png")

    joblib.dump({"model":results[best]["model"],"label_encoder":le,"features":FEAT_COLS,
                 "task":"multiclass","name":best,"classes":list(le.classes_),
                 "metrics":{"accuracy":results[best]["acc"]}},
                f"{OUT}/model_fault_type.pkl")
    return results, y_te, le

# ── Summary Dashboard ─────────────────────────────────────────────────────────
def plot_summary(bin_r, reg_r, cls_r):
    fig, axes = plt.subplots(1,3,figsize=(18,6)); _style(fig, *axes)

    def bar_h(ax, names, vals, title, color, xl, xlim=None):
        bars = ax.barh(names, vals, color=color[:len(names)])
        ax.set_title(title); ax.set_xlabel(xl)
        if xlim: ax.set_xlim(*xlim)
        for b, v in zip(bars, vals):
            ax.text(b.get_width()+0.002, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center", fontsize=9, color="#e8f0fe")

    names1=list(bin_r.keys()); bar_h(axes[0], names1, [bin_r[n]["auc"] for n in names1], "Task 1 — AUC (Fault Occurrence)", PALETTE, "AUC", (0.5,1.0))
    names2=list(reg_r.keys()); bar_h(axes[1], names2, [reg_r[n]["r2"]  for n in names2], "Task 2 — R² (Severity)",           PALETTE, "R²",  (0,1))
    names3=list(cls_r.keys()); bar_h(axes[2], names3, [cls_r[n]["acc"] for n in names3], "Task 3 — Accuracy (Fault Type)",   PALETTE, "Accuracy", (0,1))

    fig.suptitle("BSNL Fault Prediction — Model Summary Dashboard", fontsize=15, fontweight="bold", color=ACCENT, y=1.02)
    plt.tight_layout(); plt.savefig(f"{OUT}/summary_dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print("[OK]  summary_dashboard.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("  BSNL FIBRE FAULT PREDICTION — ML PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)
    df = load_data()
    plot_eda(df)
    bin_r, _  = train_fault_occurrence(df)
    reg_r, _  = train_severity(df)
    cls_r,_,_ = train_fault_type(df)
    plot_summary(bin_r, reg_r, cls_r)
    print("\n"+"="*65)
    print("  PIPELINE COMPLETE — models + plots saved to outputs/")
    print("="*65)
