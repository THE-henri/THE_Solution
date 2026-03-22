import pandas as pd
from pathlib import Path


def append_half_life_result(result_dict, results_file):
    """
    Append a single half-life result to CSV.
    Creates file if it does not exist.
    """

    results_file = Path(results_file)

    df_new = pd.DataFrame([result_dict])

    if results_file.exists():
        df_existing = pd.read_csv(results_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(results_file, index=False)
