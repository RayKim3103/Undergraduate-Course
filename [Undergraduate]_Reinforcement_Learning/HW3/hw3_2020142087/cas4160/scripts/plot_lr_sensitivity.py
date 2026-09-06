import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

# 가장 오래된 순서로 Log 자동 선택
log_dirs = sorted(glob.glob("data/hw3_dqn_dqn_CartPole-v1*"),
                  key=os.path.getmtime,
                  reverse=False)[2:6]

lr_labels = ["lr = 0.0001", "lr = 0.001 (default)", "lr = 0.01", "lr = 0.05"]
colors = ["green", "blue", "orange", "red"]

print("🔍 선택된 Log:")
for i, log_dir in enumerate(log_dirs):
    print(f"  {i+1}. {log_dir}")
    print(f"     → {lr_labels[i]}\n")

plt.figure(figsize=(10, 6))

for i, log_dir in enumerate(log_dirs):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars("eval_return")
    steps = [e.step for e in events]
    values = [e.value for e in events]
    plt.plot(steps, values, label=lr_labels[i], color=colors[i], linewidth=1.5)

plt.title("CartPole-v1: Q-learning Sensitivity to Learning Rate")
plt.xlabel("Train Environment Steps")
plt.ylabel("Eval Return")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_file = "cartpole_lr_sensitivity.png"
plt.savefig(output_file, dpi=400, bbox_inches="tight")
print(f"🎉 Plot Saved → {output_file}")
