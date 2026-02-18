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
ALL_VALID_IDS = np.arange(len(ID2COORD))


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


def find_bridge_shape(id_a, id_c):
    """Searches for a shape B such that A->B and B->C are valid transitions."""
    for b_id in ALL_VALID_IDS:
        if is_valid_transition(id_a, b_id) and is_valid_transition(b_id, id_c):
            return b_id
    return None


# --- UPDATED PLOTTING LOGIC WITH BLUE/RED BORDERS ---
def plot_all_steps_chunked(game, player_id, chunk_size=100):
    """
    Saves plots in chunks.
    Blue border = Imputed bridge.
    Red border = Illegal transition (still a jump).
    """
    all_actions = game['actions']  # List of [shape_id, timestamp, is_imputed]
    len_shapes = len(all_actions)
    output_dir = 'ground_truth_plots_TAU_with_imputation'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for page_idx, start_idx in enumerate(range(0, len_shapes, chunk_size)):
        chunk = all_actions[start_idx: start_idx + chunk_size]
        current_page = page_idx + 1
        cols = 10
        rows = int(np.ceil(len(chunk) / cols))

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, rows * 2.5), squeeze=False)
        plt.suptitle(f"Subject {player_id} | Page {current_page} | Blue=Imputed, Red=Illegal", fontsize=20)

        for counter, action in enumerate(chunk):
            shape_id, _, is_imputed = action
            global_idx = start_idx + counter

            # Detection for Remaining Illegal Jumps
            is_illegal = False
            if global_idx > 0:
                prev_shape_id = all_actions[global_idx - 1][0]
                if not is_valid_transition(prev_shape_id, shape_id):
                    is_illegal = True

            shape_matrix = simCFG.utils.get_shape_binary_matrix(int(shape_id))
            shape_image = simCFG.utils.show_binary_matrix(
                shape_matrix, show=False, is_gallery=False, render=True, res=(9, 9)
            )

            ax = axes[counter // cols, counter % cols]
            ax.imshow(shape_image)
            ax.set_xticks([]);
            ax.set_yticks([])

            if is_imputed:
                color, label = 'blue', 'IMP'
            elif is_illegal:
                color, label = 'red', 'JUMP'
            else:
                color, label = 'black', ''

            if is_imputed or is_illegal:
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(5)
                    spine.set_visible(True)
                ax.set_title(f"{label} idx:{global_idx}", color=color, fontsize=9, fontweight='bold')
            else:
                ax.axis('off')
                ax.set_title(f"idx:{global_idx}", color='black', fontsize=9)

        for i in range(counter + 1, rows * cols):
            axes[i // cols, i % cols].axis('off')

        plt.subplots_adjust(wspace=0.1, hspace=0.4)
        save_path = os.path.join(output_dir, f'Subject_{player_id}_page_{current_page}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        plt.close()


# --- MAIN BATCH EXTRACTION WITH IMPUTATION ---
def run_batch_extraction(csv_path, player_ids=None):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df['extracted_id'] = df['playerCustomData'].apply(extract_user_id)

    if not player_ids:
        player_ids = [pid for pid in df['extracted_id'].unique() if pid and str(pid).lower() != "nan"]

    for pid in player_ids:
        p_df = df[df['extracted_id'] == str(pid)].copy()
        if p_df.empty: continue

        time_col = 'userTimestamp' if 'userTimestamp' in p_df.columns else 'userTime'
        p_df[time_col] = pd.to_datetime(p_df[time_col])
        p_df = p_df.sort_values(time_col).reset_index(drop=True)

        p_df['type_norm'] = p_df['type'].astype(str).str.lower().str.replace(" ", "")
        start_indices = p_df.index[p_df['type_norm'] == 'startsearch'].tolist()
        end_indices = p_df.index[p_df['type_norm'] == 'endsearch'].tolist()
        start_cutoff = start_indices[0] if start_indices else 0
        end_cutoff = end_indices[0] if end_indices else len(p_df)
        p_df = p_df.iloc[start_cutoff: end_cutoff + 1]

        p_df['raw_coords_str'] = p_df['customData.newShape'].fillna(p_df['customData.shape'])
        valid_rows = p_df.dropna(subset=['raw_coords_str'])

        translated = []
        for _, row in valid_rows.iterrows():
            try:
                coords = json.loads(row['raw_coords_str'].replace("'", '"'))
                s_id = binary_shape_to_id(_server_coords_to_binary_shape(coords))
                if s_id is not None:
                    translated.append([s_id, row[time_col]])
            except:
                continue

        if not translated: continue

        # --- RECONSTRUCTION / IMPUTATION LOOP ---
        augmented_actions = [[translated[0][0], translated[0][1], False]]
        for i in range(1, len(translated)):
            prev_id = augmented_actions[-1][0]
            curr_id, curr_time = translated[i]

            if not is_valid_transition(prev_id, curr_id):
                bridge = find_bridge_shape(prev_id, curr_id)
                if bridge is not None:
                    # Add bridge (flagged as imputed)
                    augmented_actions.append([bridge, curr_time, True])

            # Add current actual shape
            augmented_actions.append([curr_id, curr_time, False])

        print(f"--- Subject {pid}: Processed {len(augmented_actions)} shapes (including bridges) ---")
        plot_all_steps_chunked({'actions': augmented_actions}, pid, chunk_size=100)


if __name__ == "__main__":
    CSV_FILE = '/home/roey/PycharmProjects/CFGpy_MRI_Compatible4/event_tau.csv'
    run_batch_extraction(CSV_FILE, ['ZCOnQ5'])
    