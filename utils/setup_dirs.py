"""
Setup script to create directory structure for MulFS-CAP
"""

import os

directories = [
    # Data directories
    "./data/test/vis",
    "./data/test/ir",
    "./data/train/vis",
    "./data/train/ir",

    # Output directories
    "./results",
    "./results/ird",
    "./results/fusion",

    # Model directories
    "./pretrain",
]

created = []
already_exist = []

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        created.append(directory)
        print(f"[INFO] created: {directory}")
    else:
        already_exist.append(directory)
        print(f"  already exists: {directory}")

print(f"\nsummary:")
print(f"  created: {len(created)} directories")
print(f"  already existed: {len(already_exist)} directories")

if created:
    print("\n[INFO] new directories created:")
    for d in created:
        print(f"  - {d}")