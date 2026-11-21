Python
import os
import numpy as np
import pandas as pd

def generate_driving_data(n_per_class=600, seed=42):
    rng = np.random.default_rng(seed)
    records = []

    # 0: Atsargus (Cautious), 1: Normalus (Normal), 2: Agresyvus (Aggressive)
    profiles = {
        0: {"speed_ratio": (0.88, 0.06), "speed_std": (5.5, 1.5), "acc_m": (0.8, 0.2),
            "acc_std": (0.45, 0.12), "lat_max": (1.5, 0.35), "brake": 0.3, "accel": 0.4, "jerk": (0.6, 0.2)},
        1: {"speed_ratio": (1.01, 0.06), "speed_std": (9.0, 2.0), "acc_m": (1.4, 0.25),
            "acc_std": (0.85, 0.18), "lat_max": (2.4, 0.45), "brake": 1.4, "accel": 1.6, "jerk": (1.3, 0.3)},
        2: {"speed_ratio": (1.20, 0.09), "speed_std": (15.5, 3.2), "acc_m": (2.3, 0.4),
            "acc_std": (1.55, 0.30), "lat_max": (3.6, 0.60), "brake": 4.2, "accel": 4.5, "jerk": (2.6, 0.5)},
    }

    for label, p in profiles.items():
        for _ in range(n_per_class):
            records.append({
                "avg_speed_ratio": np.clip(rng.normal(*p["speed_ratio"]), 0.5, 1.8),
                "speed_std": np.clip(rng.normal(*p["speed_std"]), 1.0, 35.0),
                "accel_long_mean": np.clip(rng.normal(*p["acc_m"]), 0.2, 5.0),
                "accel_long_std": np.clip(rng.normal(*p["acc_std"]), 0.1, 4.0),
                "accel_lat_max": np.clip(rng.normal(*p["lat_max"]), 0.3, 7.0),
                "harsh_brake_count": rng.poisson(p["brake"]),
                "harsh_accel_count": rng.poisson(p["accel"]),
                "jerk_rms": np.clip(rng.normal(*p["jerk"]), 0.1, 6.0),
                "style_label": label
            })

    df = pd.DataFrame(records).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Pridedame 15 jutiklių trikdžių anomalijų (outliers) duomenų valymo demonstravimui
    outlier_idx = rng.choice(df.index, size=15, replace=False)
    df.loc[outlier_idx, "accel_long_std"] = rng.uniform(12.0, 18.0, size=15)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/driving_telemetry.csv", index=False)
    print(f"Sugeneruota {len(df)} įrašų -> data/driving_telemetry.csv")

if __name__ == "__main__":
    generate_driving_data()
