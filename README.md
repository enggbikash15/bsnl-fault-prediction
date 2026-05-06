# 📡 BSNL Fibre Fault Prediction System

![CI](https://github.com/enggbikash15/bsnl-fault-prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end **Machine Learning system** for predicting optical fibre cable faults in BSNL's telecom network. The system solves three predictive tasks simultaneously using a 5,000-sample synthetic dataset with 30 real-world domain features.

---

## 🎯 Three ML Tasks

| Task | Type | Target | Best Model | Score |
|------|------|--------|-----------|-------|
| **Fault Occurrence** | Binary Classification | Will fault occur in 30 days? | Logistic Regression | AUC = 0.71 |
| **Fault Severity** | Regression | Severity score (0–100) | Ridge Regression | R² = 0.94 |
| **Fault Type** | Multiclass Classification | Type of fault (8 classes) | Gradient Boosting | Acc = 79.4% |

---

## 📁 Project Structure

```
bsnl-fault-prediction/
├── src/
│   ├── generate_data.py    # Synthetic BSNL dataset generator (5000 samples, 30 features)
│   ├── train.py            # Full ML pipeline — EDA + 3 tasks + all plots
│   └── predict.py          # Inference script with 3 test scenarios
├── tests/
│   └── test_pipeline.py    # 11 unit tests (all passing)
├── data/
│   └── bsnl_fault_dataset.csv
├── outputs/
│   ├── model_fault_occurrence.pkl
│   ├── model_severity.pkl
│   ├── model_fault_type.pkl
│   ├── eda_analysis.png
│   ├── task1_confusion.png
│   ├── task1_roc.png
│   ├── task1_feature_importance.png
│   ├── task2_actual_vs_pred.png
│   ├── task2_model_comparison.png
│   ├── task3_confusion.png
│   ├── task3_feature_importance.png
│   └── summary_dashboard.png
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

---

## 🔬 Features Used (30 Features)

### Cable Infrastructure
- `cable_age_years`, `cable_length_km`, `cable_type_enc`, `depth_of_burial_m`
- `num_splices`, `num_joints`, `soil_type_enc`

### Environment
- `weather_enc`, `temperature_c`, `humidity_pct`, `rainfall_mm`
- `wind_speed_kmph`, `flood_zone_flag`

### Network Performance
- `signal_strength_dbm`, `attenuation_db_per_km`, `packet_loss_pct`
- `latency_ms`, `bandwidth_utilization`, `traffic_load_gbps`, `peak_hour_flag`

### Maintenance History
- `last_maintenance_days`, `maintenance_count`, `prev_fault_count`, `days_since_last_fault`

### Geographic / External
- `zone_enc`, `distance_from_hub_km`, `num_customers_served`
- `construction_nearby`, `road_crossing_count`, `rodent_activity_index`

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/enggbikash15/bsnl-fault-prediction.git
cd bsnl-fault-prediction

# 2. Install
pip install -r requirements.txt

# 3. Generate data
python src/generate_data.py

# 4. Train all models
python src/train.py

# 5. Run predictions
python src/predict.py

# 6. Run tests
python -m pytest tests/ -v
```

---

## 📊 Sample Prediction Output

```
📍 Scenario: High Risk — Old cable, storm, construction
   Fault in 30 days    : YES ⚠
   Fault Probability   : 83.2%
   Severity Score      : 78.4 / 100
   Severity Category   : Critical
   Predicted Fault Type: Fiber Cut

📍 Scenario: Low Risk — New cable, clear weather
   Fault in 30 days    : NO  ✓
   Fault Probability   : 18.7%
   Severity Score      : 12.3 / 100
   Severity Category   : Low
   Predicted Fault Type: Equipment Fault
```

---

## 🏗 Models Trained

**Task 1 — Fault Occurrence (Binary):**
Logistic Regression, Random Forest, Gradient Boosting, XGBoost

**Task 2 — Severity (Regression):**
Ridge Regression, Random Forest, Gradient Boosting, XGBoost

**Task 3 — Fault Type (Multiclass, 8 classes):**
Random Forest, Gradient Boosting, XGBoost

---

## 👤 Author

**Bikash** | MTech — Data Mining & Network Security  
GitHub: [@enggbikash15](https://github.com/enggbikash15)

---

## 📄 License

MIT License
