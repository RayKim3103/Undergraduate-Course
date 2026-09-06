import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def plot_dqn_learning_curve(log_dir: str, output_file: str = "dqn_cartpole_learning_curve.png"):
    ea = EventAccumulator(log_dir)
    ea.Reload()

    events = ea.Scalars("eval_return")
    
    steps = [e.step for e in events]          # x축: environment steps
    values = [e.value for e in events]        # y축: eval return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label="DQN CartPole-v1", color="orange", linewidth=1.5)
    
    plt.title("DQN on CartPole-v1: Learning Curve")
    plt.xlabel("Train Environment Steps")
    plt.ylabel("Eval Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Plot saved to: {output_file}")
    print(f"   Final eval return: {values[-1]:.2f} at step {steps[-1]:,}")

if __name__ == "__main__":
    log_dir = "data/hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_06-04-2026_10-06-43"
    plot_dqn_learning_curve(log_dir)