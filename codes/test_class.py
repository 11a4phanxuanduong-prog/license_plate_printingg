from pathlib import Path
from collections import Counter

LABEL_ROOT = Path(r"C:\Users\asus\OneDrive\Desktop\bao_cao_co_xo\dataset 1\labels")

counter = Counter()

for split in ["train", "val", "test"]:
    label_dir = LABEL_ROOT / split

    for txt_file in label_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    counter[(split, parts[0])] += 1

print("===== CLASS DISTRIBUTION =====")
for key, value in counter.items():
    print(key, value)