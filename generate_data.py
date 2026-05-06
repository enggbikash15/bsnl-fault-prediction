"""
BSNL Fibre Fault Prediction System
Data Generation Module
Author: Bikash (enggbikash15)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

ZONES       = ["North Delhi","South Delhi","East Delhi","West Delhi","Central Delhi","Noida","Gurugram","Faridabad"]
FAULT_TYPES = ["Fiber Cut","Cable Damage","Signal Loss","Splice Failure","Equipment Fault","Power Failure","Weather Damage","Rodent Damage"]
CABLE_TYPES = ["Single Mode","Multi Mode","Armoured","Aerial","Underground","Submarine"]
SOIL_TYPES  = ["Sandy","Clay","Rocky","Loamy","Mixed"]
WEATHER     = ["Clear","Rain","Storm","Fog","Extreme Heat"]


def generate_bsnl_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)
    now = datetime.now()

    cable_age_years       = np.random.uniform(0.5, 25,   n_samples)
    cable_length_km       = np.random.uniform(0.5, 50,   n_samples)
    cable_type            = np.random.choice(CABLE_TYPES, n_samples)
    soil_type             = np.random.choice(SOIL_TYPES,  n_samples)
    zone                  = np.random.choice(ZONES,       n_samples)
    depth_of_burial_m     = np.random.uniform(0.3, 3.0,  n_samples)
    num_splices           = (cable_length_km * np.random.uniform(1, 3, n_samples)).astype(int)
    num_joints            = (cable_length_km * np.random.uniform(0.5, 1.5, n_samples)).astype(int)

    weather_condition     = np.random.choice(WEATHER, n_samples, p=[0.50,0.20,0.10,0.10,0.10])
    temperature_c         = np.random.uniform(-5, 48,  n_samples)
    humidity_pct          = np.random.uniform(20, 98,  n_samples)
    rainfall_mm           = np.where(weather_condition=="Rain",  np.random.uniform(10, 150, n_samples),
                            np.where(weather_condition=="Storm", np.random.uniform(80, 300, n_samples), 0))
    wind_speed_kmph       = np.random.uniform(0, 120,  n_samples)

    signal_strength_dbm   = np.random.uniform(-30, -3,  n_samples)
    attenuation_db_per_km = np.random.uniform(0.1, 1.2, n_samples)
    packet_loss_pct       = np.random.uniform(0, 15,    n_samples)
    latency_ms            = np.random.uniform(1, 150,   n_samples)
    bandwidth_utilization = np.random.uniform(10, 100,  n_samples)
    traffic_load_gbps     = np.random.uniform(0.1, 100, n_samples)
    peak_hour_flag        = np.random.randint(0, 2,     n_samples)
    num_customers_served  = np.random.randint(50, 10000,n_samples)

    last_maintenance_days = np.random.uniform(1, 1825,  n_samples)
    maintenance_count     = np.random.randint(0, 30,    n_samples)
    prev_fault_count      = np.random.randint(0, 20,    n_samples)
    days_since_last_fault = np.random.uniform(0, 730,   n_samples)

    construction_nearby   = np.random.randint(0, 2,     n_samples)
    road_crossing_count   = np.random.randint(0, 20,    n_samples)
    rodent_activity_index = np.random.uniform(0, 10,    n_samples)
    flood_zone_flag       = np.random.randint(0, 2,     n_samples)
    distance_from_hub_km  = np.random.uniform(0.5, 30,  n_samples)

    cable_type_enc = pd.Categorical(cable_type, categories=CABLE_TYPES).codes
    soil_type_enc  = pd.Categorical(soil_type,  categories=SOIL_TYPES).codes
    zone_enc       = pd.Categorical(zone,        categories=ZONES).codes
    weather_enc    = pd.Categorical(weather_condition, categories=WEATHER).codes

    # Severity score — normalised to 0-100
    raw = (
        cable_age_years        * 1.2  +
        last_maintenance_days  * 0.008 +
        prev_fault_count       * 1.5  +
        attenuation_db_per_km  * 5.0  +
        rainfall_mm            * 0.03 +
        construction_nearby    * 7.0  +
        rodent_activity_index  * 1.8  +
        flood_zone_flag        * 6.0  +
        road_crossing_count    * 0.8  +
        packet_loss_pct        * 1.2  +
        np.abs(signal_strength_dbm + 10) * 0.5 +
        np.where(weather_condition=="Storm", 12, 0) +
        np.where(weather_condition=="Rain",   5, 0) +
        np.random.normal(0, 4, n_samples)
    )
    mn, mx = raw.min(), raw.max()
    severity_score = (raw - mn) / (mx - mn) * 100
    severity_score = np.clip(severity_score, 0, 100)

    # Binary target: will fault occur in 30 days?
    # Use logistic transform centred at median of severity
    med   = np.median(severity_score)
    logit = (severity_score - med) / 15.0
    fault_prob  = 1 / (1 + np.exp(-logit))
    will_fault  = (fault_prob > np.random.uniform(0, 1, n_samples)).astype(int)

    # Fault type
    def assign_fault_type(i):
        if construction_nearby[i] and road_crossing_count[i] > 10: return "Fiber Cut"
        if rodent_activity_index[i] > 7.5:                         return "Rodent Damage"
        if weather_condition[i] == "Storm":                         return "Weather Damage"
        if weather_condition[i] == "Rain":                          return random.choice(["Cable Damage","Weather Damage"])
        if attenuation_db_per_km[i] > 0.95:                        return "Signal Loss"
        if num_splices[i] > 70:                                     return "Splice Failure"
        if cable_age_years[i] > 18:                                 return "Cable Damage"
        return random.choice(FAULT_TYPES)

    fault_type    = np.array([assign_fault_type(i) for i in range(n_samples)])
    fault_type_enc= pd.Categorical(fault_type, categories=FAULT_TYPES).codes

    timestamps = [now - timedelta(days=random.uniform(0, 365*3)) for _ in range(n_samples)]

    df = pd.DataFrame({
        "record_id":              [f"BSNL-{i+1:05d}" for i in range(n_samples)],
        "timestamp":              timestamps,
        "zone":                   zone,
        "zone_enc":               zone_enc,
        "cable_age_years":        cable_age_years.round(2),
        "cable_length_km":        cable_length_km.round(2),
        "cable_type":             cable_type,
        "cable_type_enc":         cable_type_enc,
        "depth_of_burial_m":      depth_of_burial_m.round(2),
        "num_splices":            num_splices,
        "num_joints":             num_joints,
        "soil_type":              soil_type,
        "soil_type_enc":          soil_type_enc,
        "weather_condition":      weather_condition,
        "weather_enc":            weather_enc,
        "temperature_c":          temperature_c.round(1),
        "humidity_pct":           humidity_pct.round(1),
        "rainfall_mm":            rainfall_mm.round(1),
        "wind_speed_kmph":        wind_speed_kmph.round(1),
        "flood_zone_flag":        flood_zone_flag,
        "signal_strength_dbm":    signal_strength_dbm.round(2),
        "attenuation_db_per_km":  attenuation_db_per_km.round(3),
        "packet_loss_pct":        packet_loss_pct.round(2),
        "latency_ms":             latency_ms.round(1),
        "bandwidth_utilization":  bandwidth_utilization.round(1),
        "traffic_load_gbps":      traffic_load_gbps.round(2),
        "peak_hour_flag":         peak_hour_flag,
        "num_customers_served":   num_customers_served,
        "last_maintenance_days":  last_maintenance_days.round(0),
        "maintenance_count":      maintenance_count,
        "prev_fault_count":       prev_fault_count,
        "days_since_last_fault":  days_since_last_fault.round(0),
        "distance_from_hub_km":   distance_from_hub_km.round(2),
        "construction_nearby":    construction_nearby,
        "road_crossing_count":    road_crossing_count,
        "rodent_activity_index":  rodent_activity_index.round(2),
        "severity_score":         severity_score.round(2),
        "fault_type":             fault_type,
        "fault_type_enc":         fault_type_enc,
        "will_fault_30days":      will_fault,
    })
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_bsnl_dataset(5000)
    df.to_csv("data/bsnl_fault_dataset.csv", index=False)
    print(f"[OK] Dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"     Fault rate (30-day): {df['will_fault_30days'].mean()*100:.1f}%")
    print(f"     Severity  mean/std : {df['severity_score'].mean():.1f} / {df['severity_score'].std():.1f}")
    print(f"     Fault types:\n{df['fault_type'].value_counts()}")
