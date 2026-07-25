from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path('ocr_train_metrics_by_epoch.csv')
df = pd.read_csv(csv_path)

# Gộp theo epoch để mỗi epoch chỉ còn 1 điểm
epoch_df = df.groupby("epoch", as_index=False).agg({
    "acc": "mean",
    "norm_edit_dis": "mean",
    "loss": "mean"
})

fig, ax1 = plt.subplots(figsize=(10, 6))

# Trục trái
ax1.plot(epoch_df["epoch"], epoch_df["acc"], label="Accuracy", linewidth=2)
ax1.plot(epoch_df["epoch"], epoch_df["norm_edit_dis"], label="Normalized Edit Distance", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy / Normalized Edit Distance")
ax1.grid(True, alpha=0.3)

# Trục phải
ax2 = ax1.twinx()
ax2.plot(epoch_df["epoch"], epoch_df["loss"], label="Loss", linewidth=2)
ax2.set_ylabel("Loss")

# Gộp legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("OCR Training Metrics by Epoch")
plt.tight_layout()
plt.savefig("/mnt/data/ocr_training_outputs/ocr_combined_3curves_epoch.png", dpi=200)
plt.show()