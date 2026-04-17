#!/usr/bin/env python3
"""
Plot comparison between predicted and actual Attention prefill latencies.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sqlite3

def load_prediction_data():
    """Load prediction data from CSV."""
    csv_path = "/home/jkim38/dooly/cache/tp1/meta-llama/Llama-3.1-8B/Attention_prefill_902fb90b_predictions.csv"
    df = pd.read_csv(csv_path)

    # Filter for kv_cache_size = 0
    df_filtered = df[df['kv_cache_size'] == 0].copy()

    # Convert prefill_chunk_size_squared back to prefill_chunk_size
    df_filtered['prefill_chunk_size'] = np.sqrt(df_filtered['prefill_chunk_size_squared'])

    return df_filtered[['prefill_chunk_size', 'prediction_ms']]

def load_actual_data():
    """Load actual data from database."""
    db_path = "/home/jkim38/dooly/profiler/database/dooly.db"

    query = """
    SELECT r.prefill_chunk_size, r.median_latency_ms
    FROM results r
    JOIN signatures s ON r.signature_hash = s.signature_hash
    WHERE s.operation_name = 'Attention'
      AND s.signature_hash = '902fb90b69f8c9c122ee5a426c8d978f7a9c7059401ccb0565fa646a898d6ae7'
      AND r.is_prefill = 1
      AND r.kv_cache_size = 0
    ORDER BY r.prefill_chunk_size
    """

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df.rename(columns={'median_latency_ms': 'actual_ms'})

def create_plot():
    """Create comparison plot."""
    # Load data
    pred_df = load_prediction_data()
    actual_df = load_actual_data()

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot actual data
    plt.scatter(actual_df['prefill_chunk_size'], actual_df['actual_ms'],
               alpha=0.6, s=30, color='blue', label='Actual (Database)')

    # Plot predicted data
    plt.plot(pred_df['prefill_chunk_size'], pred_df['prediction_ms'],
             color='red', linewidth=2, label='Predicted (CSV)')

    plt.xlabel('Prefill Chunk Size')
    plt.ylabel('Latency (ms)')
    plt.title('Attention Prefill Latency: Predicted vs Actual (kv_cache_size=0)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Log scale for better visualization
    plt.yscale('log')
    plt.xscale('log')

    # Add statistics
    merged_df = pd.merge(actual_df, pred_df, on='prefill_chunk_size', how='inner')
    if len(merged_df) > 0:
        mape = np.mean(np.abs((merged_df['actual_ms'] - merged_df['prediction_ms']) / merged_df['actual_ms']) * 100)
        plt.text(0.02, 0.98, f'MAPE: {mape:.2f}%', transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/jkim38/dooly/simulator/attention_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print(f"Prediction data points: {len(pred_df)}")
    print(f"Actual data points: {len(actual_df)}")
    print(f"Overlapping points: {len(merged_df)}")

    if len(merged_df) > 0:
        print(f"MAPE: {mape:.2f}%")
        print(f"Max prediction error: {np.max(np.abs(merged_df['actual_ms'] - merged_df['prediction_ms'])):.4f} ms")

if __name__ == "__main__":
    create_plot()