import pandas as pd
from pathlib import Path
from core.kinetics import fit_half_life
from core.plotting import plot_half_life_selection

# Folder setup
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "half_life" / "raw"
RESULTS_DIR = BASE_DIR / "data" / "half_life" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

csv_files = list(RAW_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}.")

results_all = []

for csv_file in csv_files:
    print(f"\nProcessing {csv_file.name}")
    df = pd.read_csv(csv_file)
    
    time = df.iloc[:,0].values
    absorbance = df.iloc[:,1].values

    # Show full dataset first
    plot_half_life_selection(time, absorbance, title=f"Raw data: {csv_file.name}")

    # User selects range (simulate for now)
    start_idx, end_idx = 0, len(time)-1  # replace with GUI/mouse selection later

    # Show selected interval
    plot_half_life_selection(time, absorbance, start_idx=start_idx, end_idx=end_idx,
                             title=f"Selection: {csv_file.name}")

    # Compute half-life
    switch_type = "positive"  # or "negative"
    k, A0, t_half = fit_half_life(time[start_idx:end_idx+1], absorbance[start_idx:end_idx+1],
                                  switch=switch_type)

    results_all.append({
        "File": csv_file.name,
        "Switch": switch_type,
        "A0": A0,
        "Rate_k": k,
        "Half_life_s": t_half
    })

# Save all results
df_results = pd.DataFrame(results_all)
output_file = RESULTS_DIR / "half_life_results.csv"
df_results.to_csv(output_file, index=False)
print(f"\nSaved all half-life results to {output_file}")
