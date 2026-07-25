import matplotlib.pyplot as plt
import pandas as pd

# Ví dụ dữ liệu
df = pd.read_csv("ocr_train_log_parsed.csv")

epochs = df["epoch"]
accuracy = df["acc"]
norm_edit = df["norm_edit_dis"]
loss = df["loss"]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Trục trái: Accuracy + Normalized Edit Distance
line1 = ax1.plot(
    epochs, accuracy,
    color="green",
    linewidth=2.5,
    label="Accuracy"
)

line2 = ax1.plot(
    epochs, norm_edit,
    color="orange",
    linewidth=2.5,
    label="Normalized Edit Distance"
)

ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Accuracy / Normalized Edit Distance", fontsize=12)
ax1.set_title("OCR Training 50 epoch", fontsize=16, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Trục phải: Loss
ax2 = ax1.twinx()
line3 = ax2.plot(
    epochs, loss,
    color="red",
    linestyle="--",
    linewidth=2.5,
    label="Loss"
)
ax2.set_ylabel("Loss", fontsize=12)

# Gộp legend
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center right", fontsize=11)

plt.tight_layout()
plt.show()