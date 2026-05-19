import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, Set, Dict, List

def load_bitcoin_interactions(file_path: str, threshold: int = 0) -> Tuple[pd.DataFrame, Set]:
    """
    Loads the Bitcoin OTC interaction dataset.
    
    The dataset is expected to have columns: ["src", "dst", "rating", "timestamp"].
    A new column 'month' is added to categorize entries into monthly groups 
    (with the earliest month as 1, then 2, 3, etc.).
    
    Returns:
        df: DataFrame with an additional 'month' column.
            (The returned DataFrame will contain: "src", "dst", "rating", "timestamp", "datetime", "month".)
        nodes: Set of node IDs encountered in the dataset.
    """
    df = pd.read_csv(file_path, header=None, names=["src", "dst", "rating", "timestamp"])
    df = df[df["rating"].abs() >= threshold]  # Only keep meaningful ratings
    
    # Convert timestamp to datetime (assuming Unix timestamp in seconds)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    start_date = df['datetime'].min()
    
    # Compute month difference (sequential month count)
    df['month'] = df['datetime'].apply(
        lambda d: (d.year - start_date.year) * 12 + (d.month - start_date.month) + 1
    )
    
    nodes = set(df["src"]).union(set(df["dst"]))
    return df, nodes

def load_epinions_endorsements(file_path: str) -> Tuple[pd.DataFrame, Set]:
    """
    Loads the Epinions endorsement dataset.
    
    Returns:
        df: DataFrame with columns ["src", "dst"]
        nodes: Set of node IDs encountered in the dataset.
    """
    df = pd.read_csv(file_path, sep="\t", comment="#", header=None, names=["src", "dst"])
    nodes = set(df["src"]).union(set(df["dst"]))
    return df, nodes

def create_node_mapping(nodes: Set) -> Dict:
    """
    Creates a mapping from node ID to an integer index.
    """
    return {node_id: idx for idx, node_id in enumerate(sorted(nodes))}

def process_bitcoin_interactions(df: pd.DataFrame, node_to_index: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes the Bitcoin OTC interactions to build positive and negative feedback matrices.
    
    The matrices are created as dense NumPy arrays.
    
    Returns:
        pos_matrix: Dense matrix (NumPy array) of positive feedback counts.
        neg_matrix: Dense matrix (NumPy array) of negative feedback counts.
    """
    N = len(node_to_index)
    pos_matrix = np.zeros((N, N), dtype=np.int32)
    neg_matrix = np.zeros((N, N), dtype=np.int32)

    for _, row in df.iterrows():
        i = node_to_index[row["src"]]
        j = node_to_index[row["dst"]]
        rating = row["rating"]

        # Here, positive interactions are counted for rating>=0 and
        # negative interactions for rating<=-1.
        if rating >= 0:
            pos_matrix[i, j] += 1
        elif rating <= -1:
            neg_matrix[i, j] += 1

    return pos_matrix, neg_matrix

def process_epinions_endorsements(df: pd.DataFrame, node_to_index: Dict) -> np.ndarray:
    """
    Processes the Epinions endorsements to build an endorsement matrix.
    
    Only endorsements between nodes that exist in the Bitcoin OTC dataset 
    (as given by node_to_index) are kept.
    
    The matrix is created as a dense NumPy array.
    
    Returns:
        endorsement_matrix: Dense matrix (NumPy array) representing binary endorsements.
    """
    N = len(node_to_index)
    endorsement_matrix = np.zeros((N, N), dtype=np.float32)

    # Only process rows where both src and dst exist in node_to_index.
    df_filtered = df[df['src'].isin(node_to_index.keys()) & df['dst'].isin(node_to_index.keys())]
    
    for _, row in df_filtered.iterrows():
        i = node_to_index[row["src"]]
        j = node_to_index[row["dst"]]
        endorsement_matrix[i, j] = 1.0  # Binary endorsement

    return endorsement_matrix

def save_combined_data(
    nodes: List,
    pos_matrix: np.ndarray,
    neg_matrix: np.ndarray,
    endorsement_matrix: np.ndarray,
    node_to_index: Dict,
    output_prefix: str = "combined_data"
) -> None:
    """
    Saves the processed dense matrices and node mappings to disk.
    
    Files saved:
      - {output_prefix}_positive_feedback.npz (key: pos_matrix)
      - {output_prefix}_negative_feedback.npz (key: neg_matrix)
      - {output_prefix}_endorsement.npz (key: endorsement_matrix)
      - {output_prefix}_nodes.json
      - {output_prefix}_node_to_index.json
    """
    np.savez_compressed(f"{output_prefix}_positive_feedback.npz", pos_matrix=pos_matrix)
    np.savez_compressed(f"{output_prefix}_negative_feedback.npz", neg_matrix=neg_matrix)
    np.savez_compressed(f"{output_prefix}_endorsement.npz", endorsement_matrix=endorsement_matrix)
    
    with open(f"{output_prefix}_nodes.json", "w") as f:
        json.dump(list(nodes), f)
    
    with open(f"{output_prefix}_node_to_index.json", "w") as f:
        json.dump(node_to_index, f)

def save_interaction_data(interactions_df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the full interaction data (with all five desired labels) to a CSV file.
    
    The saved file will contain the following columns: 
      "src", "dst", "rating", "timestamp", and "month".
    Any additional columns (e.g., "datetime") will be excluded.
    
    Args:
        interactions_df (pd.DataFrame): DataFrame containing the interaction data.
        output_path (str): File path where the CSV will be saved.
    """
    # Retain only the necessary columns.
    columns_to_save = ["src", "dst", "rating", "timestamp", "month"]
    # Check if all expected columns exist.
    available_columns = [col for col in columns_to_save if col in interactions_df.columns]
    df_to_save = interactions_df[available_columns]
    df_to_save.to_csv(output_path, index=False)
    print(f"Interaction data saved to {output_path}.")

def main() -> None:
    # File paths for the original datasets.
    bitcoin_file = "../datasets/bitcoin_otc.csv"
    epinions_file = "../datasets/epinions.txt"

    # Load datasets sequentially.
    bitcoin_df, nodes_bitcoin = load_bitcoin_interactions(bitcoin_file, 0)
    epinions_df, nodes_epinions = load_epinions_endorsements(epinions_file)

    # Filter Epinions to use only Bitcoin nodes:
    # This removes any endorsement rows where either source or destination is not in Bitcoin OTC.
    filtered_epinions_df = epinions_df[
        epinions_df['src'].isin(nodes_bitcoin) & epinions_df['dst'].isin(nodes_bitcoin)
    ].reset_index(drop=True)

    # Final node universe = Bitcoin nodes only.
    merged_nodes = sorted(list(nodes_bitcoin))
    node_to_index = create_node_mapping(merged_nodes)
    print(f"Filtered node count (Bitcoin-only): {len(merged_nodes)}")

    # Process both networks sequentially.
    pos_feedback, neg_feedback = process_bitcoin_interactions(bitcoin_df, node_to_index)
    endorsement_matrix = process_epinions_endorsements(filtered_epinions_df, node_to_index)

    # Save combined data files (feedback matrices, endorsement matrix, nodes).
    save_combined_data(
        merged_nodes,
        pos_feedback,
        neg_feedback,
        endorsement_matrix,
        node_to_index,
        output_prefix="combined_data"
    )
    
    # Save the interaction data (with all 5 labels) to a separate CSV file.
    interactions_output_path = "combined_interactions.csv"
    save_interaction_data(bitcoin_df, interactions_output_path)

    print("✅ Data preprocessing completed (dense matrices, month labels, filtered endorsements, and interaction data saved).")

if __name__ == "__main__":
    main()
