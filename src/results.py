import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_similarity_scores(similarity_dataset):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    titles = ['Original vs. Original', 'Original vs. Anonymized', 'Anonymized vs. Anonymized']
    scenarios = ['orig_orig_similarity_score', 'orig_anon_similarity_score', 'anon_anon_similarity_score']

    df = pd.DataFrame(similarity_dataset)
    eps = 0.01
    for i, scenario in enumerate(scenarios):
        df_same = df[df['same_speaker'] == True][scenario].dropna()
        df_diff = df[df['same_speaker'] == False][scenario].dropna()

        weights_same = np.ones_like(df_same) / len(df_same) * 100
        weights_diff = np.ones_like(df_diff) / len(df_diff) * 100

        axs[i].hist(df_same, bins=20, weights=weights_same, alpha=0.7, label='Same Speaker', edgecolor='black')
        axs[i].hist(df_diff, bins=20, weights=weights_diff, alpha=0.7, label='Different Speakers', edgecolor='black')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Similarity Score')
        axs[i].set_ylabel('Percentage (%)')
        min_score = min(df_same.min(), df_diff.min())
        axs[i].set_xlim([min_score - eps, 1])
        axs[i].legend()

    plt.tight_layout()
    plt.show()

def plot_wer_scores(orig_wer_stats, anon_wer_stats):
    wer_data = {
        "WER Score": ["Original", "Anonymized"],
        "Mean": [orig_wer_stats['mean'], anon_wer_stats['mean']],
        "CI Lower": [mean - ci[0] for mean, ci in zip([orig_wer_stats['mean'], anon_wer_stats['mean']],
                                                      [orig_wer_stats['ci'], anon_wer_stats['ci']])],
        "CI Upper": [ci[1] - mean for mean, ci in zip([orig_wer_stats['mean'], anon_wer_stats['mean']],
                                                      [orig_wer_stats['ci'], anon_wer_stats['ci']])]
    }
    df = pd.DataFrame(wer_data)

    fig, ax = plt.subplots()
    bars = ax.bar(df["WER Score"], df["Mean"], color=['blue', 'green'],
                  yerr=[df["CI Lower"].values, df["CI Upper"].values], capsize=5)
    
    ax.set_ylabel('WER Score Averages')
    ax.set_title('Comparison of WER Scores: Original vs. Anonymized')
    ax.set_ylim(0, 1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    max_height = max(df["Mean"]) + max(df["CI Upper"])
    for bar, ci_lower, ci_upper in zip(bars, df["CI Lower"], df["CI Upper"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci_upper + 0.01, f'{bar.get_height():.4f}',
                ha='center', va='bottom')

    plt.show()

def plot_eer_scores(eer_scores):
    eer_data = {
        "Comparison": ["Original-Original", "Original-Anonymized", "Anonymized-Anonymized"],
        "EER Score": [eer_scores['orig_orig'][0], eer_scores['orig_anon'][0], eer_scores['anon_anon'][0]],
        "Threshold": [eer_scores['orig_orig'][1], eer_scores['orig_anon'][1], eer_scores['anon_anon'][1]]
    }
    df = pd.DataFrame(eer_data)

    fig, ax = plt.subplots()
    bars = ax.bar(df["Comparison"], df["EER Score"], color=['red', 'orange', 'yellow'])
    ax.set_ylabel('EER Score')
    ax.set_title(f'Equal Error Rate (EER) Scores')
    ax.set_ylim(0, 1)
    max_height = max(df["EER Score"])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, max_height, f'{bar.get_height():.4f}', ha='center', va='bottom')
    plt.show()

def visualize_metrics(metrics):
    plot_similarity_scores(metrics['similarities'])
    plot_wer_scores(metrics['orig_wer_stats'], metrics['anon_wer_stats'])
    plot_eer_scores(metrics['eer_scores'])

# plot of similarity, eer (speaker1=speaker2, speaker1!=speaker2)
# plot of wer (avg, stdv before/after conversion)
# cosine similarity between pairs of embeddings:
    # original vs original
    # original vs anonymized
    # anonymized vs anonymized