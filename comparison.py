import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import argparse
import glob

def compare_results(dataset):
    # Define the modes and their corresponding folder names (Title Case)
    modes = ['none', 'atr', 'sharpe']
    
    # Store data for plotting and table
    comparison_data = {}
    summary_dfs = []

    print(f"Collecting results for dataset: {dataset}...")

    # 1. Load Data
    for mode in modes:
        folder_name = mode.capitalize()
        # Path to asset memory (time series)
        asset_path = f"results/{folder_name}/{dataset}_asset_test.csv"
        
        # FIX: Path to summary stats now includes the dataset name
        summary_path = f"results/{folder_name}/results_summary_{dataset}.csv"

        if os.path.exists(asset_path):
            # Load Asset Data
            df = pd.read_csv(asset_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Calculate Cumulative Reward %: (Current - Initial) / Initial * 100
            initial_asset = df['asset'].iloc[0]
            df['cumulative_reward_pct'] = ((df['asset'] - initial_asset) / initial_asset) * 100
            
            comparison_data[mode] = df
        else:
            print(f"Warning: Asset data for mode '{mode}' not found at {asset_path}")

        if os.path.exists(summary_path):
            # Load Summary Data
            df_sum = pd.read_csv(summary_path)
            summary_dfs.append(df_sum)
        else:
            print(f"Warning: Summary table for mode '{mode}' not found at {summary_path}")

    if not comparison_data:
        print("No results found. Please run test.py for at least one reward mode first.")
        return

    # Create Comparison Folder
    output_dir = "results/Comparison"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==========================================
    # 2. Generate Comparison Plot
    # ==========================================
    plt.figure(figsize=(14, 7))
    
    # colors for consistency
    colors = {'none': 'green', 'atr': 'orange', 'sharpe': '#1f77b4'} # standard matplotlib blue
    
    for mode, df in comparison_data.items():
        plt.plot(df.index, df['cumulative_reward_pct'], 
                 label=mode, 
                 color=colors.get(mode, 'black'),
                 linewidth=1.5)

    # Style formatting (Matching your reference)
    plt.title(f"Cumulative Rewards Comparison ({dataset})", fontsize=16)
    plt.xlabel("Step (Date)", fontsize=12)
    plt.ylabel("Cumulative Reward (%)", fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    
    # Grid and Ticks
    plt.grid(True) # Solid grid
    ax = plt.gca()
    
    # Format Date Axis (YYYY-MM)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    
    # Save Plot
    plot_path = os.path.join(output_dir, f"{dataset}_comparison_plot.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()

    # ==========================================
    # 3. Generate Comparison Table
    # ==========================================
    if summary_dfs:
        final_summary = pd.concat(summary_dfs, ignore_index=True)
        
        # Added Sortino_Ratio to columns
        target_cols = ['Dataset', 'Reward_Mode', 'Initial_Asset', 'Final_Asset', 'Sharpe_Ratio', 'Sortino_Ratio', 'MDD', 'Turnover_Rate']
        
        # Filter to ensure we only select columns that actually exist (backward compatibility)
        cols = [c for c in target_cols if c in final_summary.columns]
        final_summary = final_summary[cols]
        
        # Save CSV
        table_path = os.path.join(output_dir, f"{dataset}_comparison_table.csv")
        final_summary.to_csv(table_path, index=False)
        print(f"Comparison table saved to: {table_path}")

        # Print Table to Terminal
        print("\n" + "="*115)
        print(f"COMPARISON TABLE: {dataset}")
        print("-" * 115)
        # Dynamic formatting including Sortino
        print(f"{'Mode':<10} | {'Init Asset':<12} | {'Final Asset':<12} | {'Sharpe':<8} | {'Sortino':<8} | {'MDD':<8} | {'Turnover':<8}")
        print("-" * 115)
        
        for _, row in final_summary.iterrows():
            # Get values safely using .get() for optional columns
            r_mode = row['Reward_Mode']
            i_asset = row['Initial_Asset']
            f_asset = row['Final_Asset']
            sharpe = row['Sharpe_Ratio']
            sortino = row.get('Sortino_Ratio', '-') # Default to '-' if missing
            mdd = row['MDD']
            turnover = row['Turnover_Rate']
            
            print(f"{r_mode:<10} | {i_asset:<12} | {f_asset:<12} | {sharpe:<8} | {sortino:<8} | {mdd:<8} | {turnover:<8}")
        print("="*115 + "\n")
    else:
        print("No summary files found to build the table.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hightech', help='Dataset name (e.g., hightech, kdd)')
    args = parser.parse_args()
    
    compare_results(args.dataset)