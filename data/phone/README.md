# Own phone recordings (fan)

Put files here and re-run section 9 of the notebook:

```
data/phone/healthy/   recordings of the fan running normally
data/phone/faulty/    recordings with a fault introduced
```

## How to record (phyphox, free on Android/iOS)
1. Open **phyphox → Acceleration (without g)**. Settings: highest sample rate the phone allows.
2. Lay the phone flat on the fan base/motor housing, always in the same spot (mark it with tape). Same fan speed every time.
3. Record 60 s. Export → **CSV (comma, decimal point)**. Name files like `healthy_speed2_01.csv`.
4. Record at least 3 healthy files (one is used only for testing false alarms).

Fault to introduce safely (fan switched OFF while you change it):
* **Imbalance:** tape a coin (or a small clip) to one blade. Try 1 coin and 2 coins for two severity levels.
* **Looseness:** loosen one base screw by half a turn.

A microphone `.wav` (phone voice recorder held near the motor) also works and has a much higher sample rate.

These are not bearing faults, so they test the **early-warning baseline detector**, not the bearing fault-type classifier.
