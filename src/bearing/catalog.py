"""Case Western Reserve University (CWRU) bearing data: which file is which.

12 kHz drive-end accelerometer, motor loads 0-3 hp (~1797-1730 rpm).
Fault sizes 0.007", 0.014", 0.021" (seeded by EDM). Outer-race faults are the @6:00 position.
Source: https://engineering.case.edu/bearingdatacenter
"""
from __future__ import annotations

BASE_URL = "https://engineering.case.edu/sites/default/files/{id}.mat"
FS_CWRU = 12_000

# (file id, fault type, fault size in inches, motor load hp)
FILES = [
    (97, "Normal", 0.0, 0), (98, "Normal", 0.0, 1), (99, "Normal", 0.0, 2), (100, "Normal", 0.0, 3),
    (105, "Inner race", 0.007, 0), (106, "Inner race", 0.007, 1), (107, "Inner race", 0.007, 2), (108, "Inner race", 0.007, 3),
    (118, "Ball", 0.007, 0), (119, "Ball", 0.007, 1), (120, "Ball", 0.007, 2), (121, "Ball", 0.007, 3),
    (130, "Outer race", 0.007, 0), (131, "Outer race", 0.007, 1), (132, "Outer race", 0.007, 2), (133, "Outer race", 0.007, 3),
    (169, "Inner race", 0.014, 0), (170, "Inner race", 0.014, 1), (171, "Inner race", 0.014, 2), (172, "Inner race", 0.014, 3),
    (185, "Ball", 0.014, 0), (186, "Ball", 0.014, 1), (187, "Ball", 0.014, 2), (188, "Ball", 0.014, 3),
    (197, "Outer race", 0.014, 0), (198, "Outer race", 0.014, 1), (199, "Outer race", 0.014, 2), (200, "Outer race", 0.014, 3),
    (209, "Inner race", 0.021, 0), (210, "Inner race", 0.021, 1), (211, "Inner race", 0.021, 2), (212, "Inner race", 0.021, 3),
    (222, "Ball", 0.021, 0), (223, "Ball", 0.021, 1), (224, "Ball", 0.021, 2), (225, "Ball", 0.021, 3),
    (234, "Outer race", 0.021, 0), (235, "Outer race", 0.021, 1), (236, "Outer race", 0.021, 2), (237, "Outer race", 0.021, 3),
]

CLASSES = ["Normal", "Inner race", "Ball", "Outer race"]

# Drive-end bearing SKF 6205-2RS: characteristic defect frequencies as multiples of shaft speed
DEFECT_ORDERS = {"BPFI (inner race)": 5.4152, "BPFO (outer race)": 3.5848, "BSF (ball)": 4.7135, "FTF (cage)": 0.39828}

ADVICE = {
    "Normal": "No bearing fault signature. Keep the normal inspection schedule.",
    "Inner race": "Inner-race defect signature (impacts at BPFI, modulated by shaft speed). Plan a bearing replacement at the next planned stop and check lubrication and shaft fit.",
    "Ball": "Rolling-element (ball) defect signature. Often follows contamination or poor lubrication. Plan a replacement and inspect the seals and grease.",
    "Outer race": "Outer-race defect signature (impacts at BPFO). Plan a bearing replacement and check housing fit and alignment.",
}
