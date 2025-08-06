import json
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore

# --- Configuration ---
# These paths should point to the directories containing your final model checkpoints.
GEM_MODEL_DIR = "/data/ananthu/gem_project/results/qwen_1.5b_gem_run/output_dir"
CE_MODEL_DIR = "/data/ananthu/gem_project/results/qwen_1.5b_ce_run/output_dir"

# --- Helper Functions to Parse Results ---

def parse_diversity_log(log_path):
    """Parses the diversity log file to extract the Self-BLEU score."""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            match = re.search(r"BLEU similarity score: ([\d.]+)", content)
            if match:
                bleu_score = float(match.group(1))
                # The paper's diversity score is 100 - Self-BLEU
                return 100 - bleu_score
    except FileNotFoundError:
        print(f"Warning: Diversity log not found at {log_path}")
    return None

def calculate_bon_win_rates(reward_scores_path):
    """
    Parses the reward scores JSON and calculates Best-of-N (BoN) win rates.
    This version correctly handles the actual data structure.
    """
    try:
        with open(reward_scores_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Reward scores file not found at {reward_scores_path}")
        return {}

    sampling_budgets = [1, 2, 4, 8, 16, 32]
    wins_per_budget = {n: 0 for n in sampling_budgets}
    total_prompts_processed = 0

    # Each 'entry' in the data is a full record for one prompt
    for entry in data:
        scores = entry.get('reward', [])
        if not scores or not isinstance(scores, list):
            continue

        total_prompts_processed += 1
        
        for n in sampling_budgets:
            if len(scores) >= n:
                best_of_n_score = max(scores[:n])
                if best_of_n_score > 0:
                    wins_per_budget[n] += 1
    
    win_rates = {}
    if total_prompts_processed > 0:
        for n in sampling_budgets:
            win_rates[n] = (wins_per_budget[n] / total_prompts_processed) * 100

    return win_rates

# --- Main Analysis Logic ---

def main():
    print("--- Starting Analysis ---")

    # 1. Gather data for GEM Model
    gem_diversity_log = os.path.join(GEM_MODEL_DIR, "evaluation_diversity_alpaca/diversity_metrics.log")
    # --- THIS IS THE CORRECTED PATH ---
    gem_reward_json = os.path.join(GEM_MODEL_DIR, "evaluation_chat_alpaca/reward_scores.json")
    
    gem_diversity_score = parse_diversity_log(gem_diversity_log)
    gem_win_rates = calculate_bon_win_rates(gem_reward_json)
    
    # 2. Gather data for CE Model
    ce_diversity_log = os.path.join(CE_MODEL_DIR, "evaluation_diversity_alpaca/diversity_metrics.log")
    # --- THIS IS THE CORRECTED PATH ---
    ce_reward_json = os.path.join(CE_MODEL_DIR, "evaluation_chat_alpaca/reward_scores.json")

    ce_diversity_score = parse_diversity_log(ce_diversity_log)
    ce_win_rates = calculate_bon_win_rates(ce_reward_json)

    # 3. Print Comparison Table
    print("\n--- Results Summary Table (Qwen2-1.5B) ---")
    
    gem_max_win_rate = gem_win_rates.get(32, "N/A")
    ce_max_win_rate = ce_win_rates.get(32, "N/A")

    data = {
        "Metric": ["AlpacaEval Win Rate (BoN@32)", "Output Diversity Score (100-BLEU)"],
        "GEM Model": [f"{gem_max_win_rate:.2f}%" if isinstance(gem_max_win_rate, float) else "N/A", 
                      f"{gem_diversity_score:.2f}" if gem_diversity_score is not None else "N/A"],
        "CE Model": [f"{ce_max_win_rate:.2f}%" if isinstance(ce_max_win_rate, float) else "N/A", 
                     f"{ce_diversity_score:.2f}" if ce_diversity_score is not None else "N/A"]
    }
    df = pd.DataFrame(data)
    print(df.to_markdown(index=False))

    # 4. Generate Plot 1: Performance vs. Diversity
    print("\n--- Generating Plot 1: Performance vs. Diversity ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    if gem_diversity_score is not None and isinstance(gem_max_win_rate, float):
        ax.scatter(gem_diversity_score, gem_max_win_rate, s=250, label='GEM', marker='*', c='darkviolet', zorder=3, edgecolors='black')
    if ce_diversity_score is not None and isinstance(ce_max_win_rate, float):
        ax.scatter(ce_diversity_score, ce_max_win_rate, s=200, label='CE', marker='o', c='orangered', zorder=3, edgecolors='black')

    ax.set_xlabel('Output Diversity Score (100 - Self-BLEU)', fontsize=12)
    ax.set_ylabel('AlpacaEval Win Rate (%)', fontsize=12)
    ax.set_title('Performance vs. Diversity (Qwen2-1.5B)', fontsize=14, weight='bold')
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("performance_vs_diversity.png", dpi=300)
    print("Plot saved as performance_vs_diversity.png")

    # 5. Generate Plot 2: Win Rate vs. Sampling Budget
    print("\n--- Generating Plot 2: Win Rate vs. Sampling Budget ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    if gem_win_rates:
        ax.plot(list(gem_win_rates.keys()), list(gem_win_rates.values()), marker='*', linestyle='-', markersize=10, label='GEM', c='darkviolet')
    if ce_win_rates:
        ax.plot(list(ce_win_rates.keys()), list(ce_win_rates.values()), marker='o', linestyle='-', markersize=8, label='CE', c='orangered')

    ax.set_xlabel('Sampling Budget (N in Best-of-N)', fontsize=12)
    ax.set_ylabel('AlpacaEval Win Rate (%)', fontsize=12)
    ax.set_title('Test-Time Scaling Performance (Qwen2-1.5B)', fontsize=14, weight='bold')
    ax.set_xscale('log', base=2)
    
    all_keys = sorted(list(set(gem_win_rates.keys()) | set(ce_win_rates.keys())))
    if all_keys:
        ax.set_xticks(all_keys)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("win_rate_vs_budget.png", dpi=300)
    print("Plot saved as win_rate_vs_budget.png")


if __name__ == "__main__":
    main()