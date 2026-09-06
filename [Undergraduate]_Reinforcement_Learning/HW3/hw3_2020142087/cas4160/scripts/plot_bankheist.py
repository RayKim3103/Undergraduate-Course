import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def plot_bankheist_comparison():
    log_root = "data/hw3_dqn_dqn_ALE"
    all_logs = sorted(glob.glob(f"{log_root}/BankHeist-v5*"))
    
    dqn_logs = []
    ddqn_logs = []
    
    print("🔍 로그 분석 중...\n")
    
    for log_dir in all_logs:
        try:
            ea = EventAccumulator(log_dir)
            ea.Reload()
            tags = ea.Tags()["scalars"]
            
            if "eval_return" not in tags:
                print(f"⏭️  스킵 (eval_return 없음): {log_dir.split('/')[-1]}")
                continue
                
            if "doubleq" in log_dir.lower():
                ddqn_logs.append(log_dir)
                print(f"✅ Double DQN: {log_dir.split('/')[-1]}")
            else:
                dqn_logs.append(log_dir)
                print(f"✅ Vanilla DQN : {log_dir.split('/')[-1]}")
                
        except Exception as e:
            print(f"❌ 오류: {log_dir.split('/')[-1]} → {e}")
    
    print(f"\n🎯 최종 사용: Vanilla DQN {len(dqn_logs)}개 | Double DQN {len(ddqn_logs)}개\n")
    
    if len(dqn_logs) == 0 or len(ddqn_logs) == 0:
        print("❌ 사용할 로그가 부족합니다.")
        return

    plt.figure(figsize=(12, 7))
    
    # Vanilla DQN → 파란색
    for i, log_dir in enumerate(dqn_logs):
        ea = EventAccumulator(log_dir)
        ea.Reload()
        events = ea.Scalars("eval_return")
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.plot(steps, values, color="blue", alpha=0.75, linewidth=1.8,
                 label="DQN" if i == 0 else "")

    # Double DQN → 빨간색
    for i, log_dir in enumerate(ddqn_logs):
        ea = EventAccumulator(log_dir)
        ea.Reload()
        events = ea.Scalars("eval_return")
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.plot(steps, values, color="red", alpha=0.75, linewidth=1.8,
                 label="Double DQN" if i == 0 else "")

    plt.ticklabel_format(axis='x', style='plain')

    plt.title("BankHeist-v5: DQN vs Double DQN (3 seeds each)")
    plt.xlabel("Train Environment Steps")
    plt.ylabel("Eval Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_file = "bankheist_dqn_vs_ddqn.png"
    plt.savefig(output_file, dpi=400, bbox_inches="tight")
    print(f"🎉 Plot 저장 완료 → {output_file}")
    print("   Overleaf에 바로 넣을 수 있습니다!")

if __name__ == "__main__":
    plot_bankheist_comparison()
