import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

# 가장 오래된 2개의 로그 선택 (lr=0.001 → lr=0.05 순서)
log_dirs = sorted(glob.glob("data/hw3_dqn_dqn_CartPole-v1*"),
                  key=os.path.getmtime, reverse=False)[:2]

labels = ["lr = 0.001 (default)", "lr = 0.05"]
colors = ["blue", "red"]

print("Selected Logs:")
for i, d in enumerate(log_dirs):
    print(f"  {i+1}. {d} → {labels[i]}")

# 3개의 개별 plot 생성
keys = ["q_values", "critic_loss", "eval_return"]
titles = ["(a) Predicted Q-values", "(b) Critic Error (Loss)", "(c) Eval Returns"]
ylabels = ["Q-value", "Loss", "Eval Return"]
filenames = ["q_values_comparison.png", "critic_loss_comparison.png", "eval_return_comparison.png"]

for idx, key in enumerate(keys):
    plt.figure(figsize=(8, 5))
    
    for i, log_dir in enumerate(log_dirs):
        ea = EventAccumulator(log_dir)
        ea.Reload()
        events = ea.Scalars(key)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.plot(steps, values, label=labels[i], color=colors[i], linewidth=1.5)
    
    plt.title(titles[idx])
    plt.xlabel("Environment Steps")
    plt.ylabel(ylabels[idx])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(filenames[idx], dpi=400, bbox_inches="tight")
    print(f"✅ Saved: {filenames[idx]}")

print("\n Total 3 images Generated")
print("   → q_values_comparison.png")
print("   → critic_loss_comparison.png")
print("   → eval_return_comparison.png")
