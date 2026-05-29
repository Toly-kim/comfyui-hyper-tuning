import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visual_report(csv_path):
    # 1. Загружаем данные
    df = pd.read_csv(csv_path)

    # Настройка стиля графиков
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))

    # --- ГРАФИК 1: Сравнение CLIP Score (Subject vs Style) ---
    plt.subplot(2, 1, 1)

    # Подготавливаем данные для групповой диаграммы
    clip_cols = ['clip_subject_v1', 'clip_subject_v2', 'clip_style_v1', 'clip_style_v2']
    df_melted = df.melt(id_vars='seed', value_vars=clip_cols, var_name='Metric', value_name='Score')

    sns.barplot(data=df_melted, x='seed', y='Score', hue='Metric', palette='viridis')
    plt.title('CLIP Score Comparison: Subject vs Style (V1 vs V2)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- ГРАФИК 2: Тренды Sharpness и SSIM ---
    plt.subplot(2, 1, 2)

    # Используем двойную ось (TwinX), так как шкалы разные
    ax1 = sns.lineplot(data=df, x=range(len(df)), y='sharpness_v2', marker='o', label='Sharpness V2', color='blue')
    plt.ylabel('Sharpness Value')

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x=range(len(df)), y='ssim', marker='s', label='SSIM (Consistency)', color='red', ax=ax2)
    plt.ylabel('SSIM Value')

    plt.title('Sharpness and Structural Similarity (SSIM) across Seeds', fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Сохраняем и показываем
    plt.tight_layout()
    report_name = csv_path.replace('.csv', '.png')
    plt.savefig(report_name)
    print(f"📊 Визуальный отчет сохранен как: {report_name}")
    plt.show()