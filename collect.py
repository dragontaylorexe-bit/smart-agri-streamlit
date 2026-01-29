# collect.py
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

BLYNK_TOKEN = "HBV4glfmRrg7jhpsRHFnu2cnRM2TebZK"
BASE_URL = "https://blynk.cloud/external/api/get"

REFRESH_SEC = 5

# ✅ CHỐT THỨ TỰ CHUẨN Ở ĐÂY (DÙNG XUYÊN SUỐT)
FEATURES = ["temp", "humidity", "soil", "ph"]
PINS = {"temp": "V0", "soil": "V1", "ph": "V2", "humidity": "V3"}

OUT_PATH = Path("data/sensors.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_value(pin: str) -> float:
    r = requests.get(BASE_URL, params={"token": BLYNK_TOKEN, pin: ""}, timeout=10)
    r.raise_for_status()
    return float(r.text.strip())

def read_sensors() -> dict:
    # đọc theo FEATURES để đảm bảo đúng thứ tự
    values = {}
    for name in FEATURES:
        values[name] = get_value(PINS[name])
    return values

def main():
    print("Collecting... Ctrl+C to stop")
    while True:
        try:
            row = read_sensors()
            row["ts"] = datetime.now(timezone.utc).isoformat()
            df = pd.DataFrame([row], columns=["ts"] + FEATURES)

            # append vào CSV
            header = not OUT_PATH.exists()
            df.to_csv(OUT_PATH, mode="a", header=header, index=False)

            print(row)
        except Exception as e:
            print("Read error:", e)

        time.sleep(REFRESH_SEC)

if __name__ == "__main__":
    main()
