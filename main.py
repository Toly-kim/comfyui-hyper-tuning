from metrics import selector
from src.core import propmt_run
from src.metrics import calculate_metrics
from src.metrics import analyze_pairs

if __name__ == "__main__":
# For paired experiment
    propmt_run.run_random_seeds() # runs_log.csv
    propmt_run.run_same_seed_another_prompt()  # paired_results.csv
    calculate_metrics.run_metrics()  # metrics_log.csv
    calculate_metrics.run_test_identity_check()  # Regression test1
    analyze_pairs.analyze_results()  # summary_stats_csv
    propmt_run.run_golden_seeds() #

# Best-output selector -> Custom node
    selector.run_multi_seed_for_all_prompts()  # all_prompts_results.csv
    calculate_metrics.compute_metrics()  # all_prompts_score.csv
    calculate_metrics.select_best_output()  # select_best_output.csv
    calculate_metrics.aggregate_best_by_prompt_version()  # prompt_ranking.csv