import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import EngFormatter
import seaborn as sns # Ensure seaborn is imported

# --- Configuration ---
NUM_BASE_NODES = 5000
NUM_ENDORSERS_PER_SCENARIO = 50

# New node IDs for the demonstration
NEW_NODE_HIGH_REP = 5000
NEW_NODE_AVG_REP = 5001
NEW_NODE_LOW_REP = 5002

def run_final_demonstration():
    """
    Generates three visual scenarios using a self-contained, synthetic dataset
    with wide ("fat") distributions for clear demonstration.
    """
    
    # --- Step 1: Generate a "fat" base reputation distribution ---
    print("Step 1: Generating a base pool of nodes with a wide reputation distribution...")
    
    # A Beta distribution with a=2, b=2 creates a nice, wide, bell-shaped distribution
    # centered at 0.5. This ensures our source data is not "thin".
    base_reputations = np.random.beta(a=2, b=2, size=NUM_BASE_NODES)
    
    reputations_df = pd.DataFrame({
        'node_id': range(NUM_BASE_NODES),
        'reputation': base_reputations
    })
    print(f"Created a pool of {len(reputations_df)} nodes.\n")

    # --- Step 2: Select Endorser Groups using Recipe-Based Sampling ---
    print("Step 2: Selecting endorser groups using a precise recipe...")
    sorted_reps = reputations_df.sort_values(by='reputation').reset_index(drop=True)
    n_nodes = len(sorted_reps)
    
    # Create 10 decile bins based on reputation
    bins = [sorted_reps.iloc[int(i*n_nodes/10):int((i+1)*n_nodes/10)] for i in range(10)]
    
    # Define the "recipes" for sampling from the 10 bins (from lowest to highest rep)
    # Each recipe sums to 50
    low_rep_recipe  = [20, 15, 8, 4, 3, 0, 0, 0, 0, 0] # Heavily left-skewed
    avg_rep_recipe  = [1, 4, 10, 10, 10, 5, 4, 3, 2, 1] # Bell-shaped
    high_rep_recipe = [0, 0, 0, 0, 0, 3, 4, 8, 15, 20] # Heavily right-skewed

    def sample_from_recipe(recipe):
        endorser_list = []
        for i, count in enumerate(recipe):
            sample_count = min(count, len(bins[i]))
            if sample_count > 0:
                endorser_list.append(bins[i].sample(sample_count, random_state=42))
        return pd.concat(endorser_list)

    low_rep_endorsers = sample_from_recipe(low_rep_recipe)
    avg_rep_endorsers = sample_from_recipe(avg_rep_recipe)
    high_rep_endorsers = sample_from_recipe(high_rep_recipe)

    print("Selected 3 groups of endorsers to fit specific distribution shapes.\n")

    # --- Step 3: Calculate New Node Reputations ---
    print("Step 3: Calculating initial reputation for new nodes...")
    
    # For the demonstration, the new node's score is the average of its endorsers' scores
    final_score_high = high_rep_endorsers['reputation'].mean()
    final_score_avg = avg_rep_endorsers['reputation'].mean()
    final_score_low = low_rep_endorsers['reputation'].mean()

    print("\n--- Final Initial Scores for New Nodes ---")
    print(f"New Node (High-Rep Endorsers): Initial Score = {final_score_high:.4f}")
    print(f"New Node (Avg-Rep Endorsers):  Initial Score = {final_score_avg:.4f}")
    print(f"New Node (Low-Rep Endorsers):  Initial Score = {final_score_low:.4f}\n")
    
    # --- Step 4: Visualize the Results ---
    print("Step 4: Generating figures...")
    
    # --- Plotting Parameters ---
    # SNS_STYLE = "ticks" # Seaborn style name
    FIG_DPI = 300
    FONT_TITLE = FontProperties(weight='bold', size=35)
    FONT_LABEL = FontProperties(weight='bold', size=35)
    FONT_TICK = FontProperties(weight='bold', size=25)
    legend_font = FontProperties(weight='bold', size=25)

    def plot_scenario(endorser_group, new_node_final_score, title, filename):
        plt.figure(figsize=(13, 9))

        # Plot a histogram and a density curve for a nice visual
        plt.hist(endorser_group['reputation'], bins=20, alpha=0.6, edgecolor="black", label='Endorser Reputations', range=(0,1), density=True)
        pd.Series(endorser_group['reputation']).plot.density(label='Distribution Shape', linewidth=3)
        
        plt.axvline(new_node_final_score, color='r', linestyle='--', linewidth=4.5, label=f'New Node Initial Score ({new_node_final_score:.2f})')
        plt.title(title, fontproperties=FONT_TITLE)
        plt.xlabel('Reputation Score', fontproperties=FONT_LABEL)
        plt.ylabel('Density', fontproperties=FONT_LABEL)
            # Apply tick font properties
        ax = plt.gca()
        for label in ax.get_xticklabels(): label.set_fontproperties(FONT_TICK)
        for label in ax.get_yticklabels(): label.set_fontproperties(FONT_TICK)

        plt.legend(prop=legend_font)
        plt.grid(True)
        plt.xlim(0, 1)
        plt.savefig(filename, dpi=FIG_DPI)
        plt.close()

    plot_scenario(high_rep_endorsers, final_score_high, 'Endorsed by a High-Reputation Group', 'final_demo_high_rep.pdf')
    plot_scenario(avg_rep_endorsers, final_score_avg, 'Endorsed by an Average-Reputation Group', 'final_demo_average_rep.pdf')
    plot_scenario(low_rep_endorsers, final_score_low, 'Endorsed by a Low-Reputation Group', 'final_demo_low_rep.pdf')
    
    print("All three demonstration figures have been saved as PDF files.")

if __name__ == '__main__':
    run_final_demonstration()