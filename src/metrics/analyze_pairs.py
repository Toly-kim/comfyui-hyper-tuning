import os
import pandas as pd
from src.utils.paths import paths
from src.utils.config_params import get_run_params2

# Reads metrics_log_csv, processes, creates summary_stats.csv and Stats Summary in console
# clip_sub_v1, clip_sub_v2, clip_sty_v1, clip_sty_v2, lpips, ssim, sharpness_v1, sharpness_v2, color_dist
#
def analyze_results():
    p = get_run_params2()
    csv_filename = p['metrics_log_csv']
    summary_output = p['summary_stats_csv']

    # Пороги из конфига
    delta_mean_threshold = p.get('delta_mean_threshold', 0.01)
    improved_pct_threshold = p.get('improved_pct_threshold', 50.0)

    if not csv_filename or not os.path.exists(csv_filename):
        print(f"❌ File not found error: {csv_filename}")
        return

    df = pd.read_csv(csv_filename)
    if df.empty:
        print(f"⚠️ File is empty")
        return

    # Чистим данные
    df = df.apply(pd.to_numeric, errors='coerce')

    # Настройка полярности метрик (1: больше=лучше, -1: меньше=лучше)
    paired_meta = {
        'clip_sub':  {'label': 'CLIP Subject', 'dir': 1},
        'clip_sty':  {'label': 'CLIP Style',   'dir': 1},
        'sharpness': {'label': 'Sharpness',    'dir': 1}
    }

    single_metrics = {
        'lpips': 'LPIPS',
        'ssim': 'SSIM',
        'color_dist': 'Color Dist'
    }

    summary_data = []
    print(f"\n🚀 Stats Summary (Thresholds: Δ > {delta_mean_threshold}, % > {improved_pct_threshold})")

    for key, meta in paired_meta.items():
        label = meta['label']
        direction = meta['dir']
        v1_col, v2_col = f'{key}_v1', f'{key}_v2'

        if v1_col in df.columns and v2_col in df.columns:
            valid_df = df[[v1_col, v2_col]].dropna()
            v1, v2 = valid_df[v1_col], valid_df[v2_col]

            delta = v2 - v1
            avg_delta = delta.mean()

            # Учитываем направление (direction) для расчета процента улучшений
            # Если dir=1, считаем где delta > 0. Если dir=-1, считаем где delta < 0.
            improved_mask = (delta * direction > 0)
            improved_pct = improved_mask.mean() * 100

            # Условие "Успеха": среднее изменение выше порога И большинство пар стали лучше
            success = (avg_delta * direction > delta_mean_threshold) and (improved_pct > improved_pct_threshold)

            icon = "✅" if success else "⚠️"
            status_text = "Улучшение:" if (improved_pct > improved_pct_threshold) else "Ухудшение:"

            summary_data.append({
                'Metric': label,
                'V1_Mean': v1.mean(),
                'V2_Mean': v2.mean(),
                'Delta_Mean': avg_delta,
                'Improved_Pct': improved_pct
            })

            print(f"{icon} {label:<15}: Δ {avg_delta:+.4f} | {status_text:<11} {improved_pct:>5.1f}%")

    # 2. Анализ одиночных метрик (характеристики разницы)
    print("-" * 50)
    for key, label in single_metrics.items():
        if key in df.columns:
            val = df[key].dropna()
            avg_val = val.mean()

            summary_data.append({
                'Metric': label,
                'V1_Mean': None,
                'V2_Mean': None,
                'Delta_Mean': avg_val,
                'Improved_Pct': None
            })
            # Для LPIPS и SSIM можно добавить пояснения в принты
            note = ""
            if key == 'lpips' and avg_val > 0.5: note = "(Strongly Different)"
            if key == 'ssim' and avg_val > 0.8: note = "(Structurally Similar)"

            print(f"ℹ️ {label:<15}: Среднее значение {avg_val:.4f} {note}")

    # Сохранение
    if summary_data:
        pd.DataFrame(summary_data).to_csv(summary_output, index=False)
        print("=" * 60)
        print(f"💾 Report saved to: {summary_output}")
def test_paired_results():
    p = get_run_params2()
    csv_filename = p['metrics_log_csv']
    # summary_output = p.get('summary_stats_csv', 'summary_stats.csv')
    # summary_output = p.get('summary_stats_csv')
    summary_output = p['test_paired_results']

    if not csv_filename or not os.path.exists(csv_filename):
        print(f"❌ File not found error: {csv_filename}")
        return

    df = pd.read_csv(csv_filename)
    if df.empty:
        print(f"⚠️ File is empty")
        return

    # Чистим данные от NaN один раз для всех колонок сразу
    df = df.apply(pd.to_numeric, errors='coerce')

    paired_metrics = {
        'clip_sub': 'CLIP Subject',
        'clip_sty': 'CLIP Style',
        'sharpness': 'Sharpness'
    }

    summary_data = []
    print(f"\nStats Summary")

    # Анализ парных (v1 vs v2)
    for key, label in paired_metrics.items():
        v1_col, v2_col = f'{key}_v1', f'{key}_v2' # the same for regress testing

        if v1_col in df.columns and v2_col in df.columns:
            # Выбираем только чистые пары
            valid_df = df[[v1_col, v2_col]].dropna()

            v1 = valid_df[v1_col]
            v2 = valid_df[v2_col]
            delta = v2 - v1

            # Больше = лучше для этих метрик
            improved_pct = (delta > 0).mean() * 100

            summary_data.append({
                'Metric': label,
                'V1_Mean': v1.mean(),
                'V2_Mean': v2.mean(),
                'Delta_Mean': delta.mean(),
                'Improved_Pct': improved_pct
            })

            icon = "✅" if delta.mean() > 0 else "⚠️"
            print(f"{icon} {label:<15}: Δ {delta.mean():+.4f} | Улучшение: {improved_pct:>5.1f}%")

    # Анализ одиночных (характеристики пары)
    print("-" * 30)
    for key, label in single_metrics.items():
        if key in df.columns:
            val = df[key].dropna()

            summary_data.append({
                'Metric': label,
                'V1_Mean': None,
                'V2_Mean': None,
                'Delta_Mean': val.mean(),
                'Improved_Pct': None
            })
            print(f"ℹ️ {label:<15}: Среднее значение {val.mean():.4f}")

    if summary_data:
        pd.DataFrame(summary_data).to_csv(summary_output, index=False)
        print("=" * 60)
        print(f"💾 Stats have been saved: {summary_output}")
    else:
        print("❌ Не удалось найти колонки метрик. Проверьте имена в CSV!")


if __name__ == "__main__":
    analyze_results() # paired_results.csv