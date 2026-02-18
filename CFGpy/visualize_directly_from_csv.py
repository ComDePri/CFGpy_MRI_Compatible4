import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import simCFG
import ast

# --- IMPORTS FROM YOUR ENVIRONMENT ---
from CFGpy.behavioral._utils import _server_coords_to_binary_shape
from CFGpy.behavioral.PostParser import is_valid_transition

# Load the Global ID-to-Coordinate map
ID2COORD_PATH = "/home/roey/PycharmProjects/CFGpy_MRI_Compatible4/venv/lib/python3.10/site-packages/simCFG/datafiles/grid_coords.npy"
ID2COORD = np.load(ID2COORD_PATH)


def binary_shape_to_id(binary_shape):
    """Translates the 10-int binary row-sums to a Serial ID."""
    pad_width = (0, 10 - len(binary_shape))
    padded_shape = np.pad(binary_shape, pad_width=pad_width)
    shape_id = np.flatnonzero(np.all(ID2COORD == padded_shape, axis=1))
    return shape_id[0] if shape_id.size == 1 else None


def extract_user_id(val):
    """Robustly parses userProvidedId from playerCustomData string."""
    if pd.isna(val) or val == "": return ""
    if isinstance(val, dict): return str(val.get('userProvidedId', '')).strip()
    try:
        data = ast.literal_eval(val)
        if isinstance(data, dict): return str(data.get('userProvidedId', '')).strip()
    except:
        try:
            data = json.loads(val.replace("'", '"'))
            return str(data.get('userProvidedId', '')).strip()
        except:
            pass
    return ""


# --- UPDATED PLOTTING LOGIC WITH ILLEGAL STEP DETECTION ---
def plot_all_steps_chunked(game, player_id, chunk_size=100):
    """Saves high-res plots in chunks. Marks illegal steps with red borders."""
    all_actions = game['actions']  # List of [shape_id, timestamp]
    len_shapes = len(all_actions)
    output_dir = 'ground_truth_plots_TAU'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for page_idx, start_idx in enumerate(range(0, len_shapes, chunk_size)):
        chunk = all_actions[start_idx: start_idx + chunk_size]
        current_page = page_idx + 1
        cols = 10
        rows = int(np.ceil(len(chunk) / cols))

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, rows * 2.5), squeeze=False)
        plt.suptitle(f"Subject {player_id} | Page {current_page} | Red = Illegal Transition", fontsize=20)

        for counter, action in enumerate(chunk):
            shape_id = action[0]
            global_idx = start_idx + counter

            # --- ILLEGAL STEP DETECTION ---
            is_illegal = False
            if global_idx > 0:
                prev_shape_id = all_actions[global_idx - 1][0]
                # If current shape is not a neighbor of the previous shape
                if not is_valid_transition(prev_shape_id, shape_id):
                    is_illegal = True

            # Translate ID to matrix via simCFG
            shape_matrix = simCFG.utils.get_shape_binary_matrix(int(shape_id))
            res = (9, 9)
            shape_image = simCFG.utils.show_binary_matrix(
                shape_matrix, show=False, is_gallery=False, render=True, res=res
            )

            ax = axes[counter // cols, counter % cols]
            ax.imshow(shape_image)
            ax.set_xticks([]);
            ax.set_yticks([])

            if is_illegal:
                # Add thick red border for illegal steps
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(5)
                    spine.set_visible(True)
                ax.set_title(f"idx:{global_idx}", color='red', fontsize=10, fontweight='bold')
            else:
                ax.axis('off')
                ax.set_title(f"idx:{global_idx}", color='black', fontsize=10)

        # Hide unused subplots
        for i in range(counter + 1, rows * cols):
            axes[i // cols, i % cols].axis('off')

        plt.subplots_adjust(wspace=0.1, hspace=0.4)
        save_path = os.path.join(output_dir, f'Subject_{player_id}_page_{current_page}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        plt.close()


# --- MAIN BATCH EXTRACTION ---
def run_batch_extraction(csv_path, player_ids=None):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # 1. Resolve Player IDs
    df['extracted_id'] = df['playerCustomData'].apply(extract_user_id)
    if not player_ids:
        player_ids = [pid for pid in df['extracted_id'].unique() if pid and str(pid).lower() != "nan"]

    for pid in player_ids:
        p_df = df[df['extracted_id'] == str(pid)].copy()
        if p_df.empty:
            print(f"   !!! No data found for {pid}")
            continue

        # 2. Sort by userTimestamp (with fallback to userTime)
        time_col = 'userTimestamp' if 'userTimestamp' in p_df.columns else 'userTime'
        p_df[time_col] = pd.to_datetime(p_df[time_col])
        p_df = p_df.sort_values(time_col).reset_index(drop=True)

        # 3. Windowing (Search Phase Only)
        # Normalize event types to handle 'startsearch', 'startSearch', 'end search', etc.
        p_df['type_norm'] = p_df['type'].astype(str).str.lower().str.replace(" ", "")

        start_indices = p_df.index[p_df['type_norm'] == 'startsearch'].tolist()
        end_indices = p_df.index[p_df['type_norm'] == 'endsearch'].tolist()

        start_cutoff = start_indices[0] if start_indices else 0
        end_cutoff = end_indices[0] if end_indices else len(p_df)

        p_df = p_df.iloc[start_cutoff: end_cutoff + 1]

        # 4. Extract Shape Actions
        p_df['raw_coords_str'] = p_df['customData.newShape'].fillna(p_df['customData.shape'])
        valid_rows = p_df.dropna(subset=['raw_coords_str'])

        translated_actions = []
        for _, row in valid_rows.iterrows():
            try:
                # Use replace to handle Python-style single quotes in coordinates
                coords = json.loads(row['raw_coords_str'].replace("'", '"'))
                s_id = binary_shape_to_id(_server_coords_to_binary_shape(coords))
                if s_id is not None:
                    translated_actions.append([s_id, row[time_col]])
            except:
                continue

        if translated_actions:
            print(f"--- Subject {pid}: Plotting {len(translated_actions)} shapes ---")
            game_obj = {'id': pid, 'actions': translated_actions}
            plot_all_steps_chunked(game_obj, pid, chunk_size=100)


if __name__ == "__main__":
    CSV_FILE = '/home/roey/PycharmProjects/CFGpy_MRI_Compatible4/event_tau.csv'
    # Specify IDs to plot, or leave as None to process everyone
    PLAYERS_TO_PLOT = ['999991','999994']
    run_batch_extraction(CSV_FILE, PLAYERS_TO_PLOT)