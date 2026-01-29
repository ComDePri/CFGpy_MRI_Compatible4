import json

import numpy as np

from CFGpy.behavioral._utils import load_json, CFGPipelineException, segment_explore_exploit, \
    prettify_games_json, segment_explore_exploit_mri
from CFGpy.behavioral._consts import (PARSED_ALL_SHAPES_KEY, PARSED_PLAYER_ID_KEY, EXPLORE_KEY, EXPLOIT_KEY,
                                      INVALID_SHAPE_ERROR, NOT_A_NEIGHBOR_ERROR, POSTPARSER_OUTPUT_FILENAME)
from CFGpy.behavioral import Configuration
from CFGpy.utils import FilesHandler
import pandas as pd
from itertools import groupby

def is_valid_transition(shape1: int, shape2: int) -> bool:
    """
    Checks whether a transition is a valid path in the CFG.
    TODO: assumes empty moves are valid. When empty moves handling is implemented, this function can be replaced with
        shape_network.has_edge(shape1, shape2)
    :param shape1: shape id, after conversion to int by PostParser.convert_shape_ids
    :param shape2: shape id, after conversion to int by PostParser.convert_shape_ids
    :return: True if the transition is valid, False if not
    """
    return shape1 == shape2 or FilesHandler().shape_network.has_edge(shape1, shape2)


class PostParser:
    def __init__(self, *, parsed_data, is_rm1: bool = False,  is_mri: bool = False,
                 config: Configuration = None):
        self.all_players_data = parsed_data
        self.config = config or Configuration.default(is_rm1=is_rm1, is_mri=is_mri)

    @classmethod
    def from_json(cls, path: str, config=None):
        return cls(load_json(path), config)

    def postparse(self):
        self.convert_shape_ids()
        if self.config.SEGMENTATION_ALGORITHM == "MRI":
            self.handle_empty_moves()
        self.add_explore_exploit()
        # TODO: Remove bad games?
        return self.all_players_data

    def convert_shape_ids(self):
        """
        Converts shape ids from their graphical representations to serial numbers.
        Raises an exception if illegal shapes are found.
        """
        from CFGpy.utils import binary_shape_to_id as bin2id

        for player_data in self.all_players_data:
            shapes = player_data[PARSED_ALL_SHAPES_KEY]
            for i, shape in enumerate(shapes):
                shape_binary_repr = shape[self.config.SHAPE_ID_IDX]
                player_id = player_data[PARSED_PLAYER_ID_KEY]
                try:
                    shape_id = bin2id(shape_binary_repr)
                    shape[self.config.SHAPE_ID_IDX] = shape_id
                except ValueError:
                    raise CFGPipelineException(INVALID_SHAPE_ERROR.format(shape_binary_repr, player_id))

                if i > 0 and not is_valid_transition(shapes[i - 1][self.config.SHAPE_ID_IDX], shape_id):
                    print(CFGPipelineException(NOT_A_NEIGHBOR_ERROR.format(i - 1, i, player_id)))
                    # the exception is printed and not raised because many gaps are actually in the source data

    @staticmethod
    def group_consecutive_duplicates(elements):
        """
        Returns a list of group ids such that each group contains consecutive duplicate elements.
        :param elements: iterable
        :return: 1D list with len equal to elements
        """
        group_count = 0
        group_ids = []
        for k, g in groupby(elements):
            group_ids.extend([group_count] * len(list(g)))
            group_count += 1

        return group_ids

    def handle_empty_moves(self):
        """
        Merges consecutive duplicate shapes into single steps, updating their move/save times accordingly.
        """
        shape_id_idx = self.config.SHAPE_ID_IDX
        shape_start_time_idx = self.config.SHAPE_MOVE_TIME_IDX
        shape_save_time_idx = self.config.SHAPE_SAVE_TIME_IDX
        shape_last_move_time_idx = self.config.SHAPE_MAX_MOVE_TIME_IDX

        for player_data in self.all_players_data:
            if not player_data[PARSED_ALL_SHAPES_KEY]:
                continue

            # === DEBUG START: Subject 080 Tracking ===
            p_id = player_data.get(PARSED_PLAYER_ID_KEY, "Unknown")
            is_target = p_id == "080"
            if is_target:
                initial_count = len(player_data[PARSED_ALL_SHAPES_KEY])
                print(f"\n[DEBUG 080] Pre-merge count: {initial_count}")
            # === DEBUG END ===

            shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
            shapes_df[shape_last_move_time_idx] = shapes_df[shape_start_time_idx]

            # Grouping by consecutive duplicate shapes
            shapes_df["group_id"] = self.group_consecutive_duplicates(shapes_df[shape_id_idx])

            shapes_df = (shapes_df
                         .groupby("group_id", as_index=False)
                         .agg({shape_id_idx: lambda x: x.iloc[0],
                               shape_start_time_idx: lambda x: x.iloc[0],
                               shape_save_time_idx: "min",
                               shape_last_move_time_idx: lambda x: x.iloc[-1]})
                         .drop(columns="group_id"))

            # Fixes possible column reordering caused by agg()
            shapes = (shapes_df
                      .reindex(sorted(shapes_df.columns), axis="columns")
                      .values.tolist())

            # === DEBUG START: Check for Post-Merge Integrity ===
            if is_target:
                from CFGpy.utils import is_neighbor  # Adjust import as needed
                final_count = len(shapes)
                print(f"[DEBUG 080] Post-merge count: {final_count} (Dropped {initial_count - final_count} stutters)")

                # Check if the merge created a neighbor violation
                for i in range(len(shapes) - 1):
                    s1, s2 = int(shapes[i][shape_id_idx]), int(shapes[i + 1][shape_id_idx])
                    if s1 != s2 and not is_neighbor(s1, s2):
                        print(
                            f"[DEBUG 080] ALERT: Non-neighboring transition created/detected at index {i}: {s1} -> {s2}")
            # === DEBUG END ===

            if len(shapes) > 0 and len(shapes[0]) <= shape_last_move_time_idx:
                player_id = player_data[PARSED_PLAYER_ID_KEY]
                msg = (f"CRITICAL ERROR: MRI Empty Move handling failed for player {player_id}.\n"
                       f"Expected column index {shape_last_move_time_idx} to exist, but found only {len(shapes[0])} columns.")
                raise CFGPipelineException(msg)

            player_data[PARSED_ALL_SHAPES_KEY] = shapes


    def handle_empty_moves_WITHOUT_DEBUGGING(self):
        """
        Merges consecutive duplicate shapes into single steps, updating their move/save times accordingly.
        This is only called in MRI mode, where empty moves are possible.
        It creates a 4th column in the shapes data, which holds the time of the last move in the merged step.
        If there hasn't been any empty moves, the last move time is equal to the start time.
        This method modifies self.all_players_data in place, and calls a helper method to group consecutive duplicates.
        """
        shape_id_idx = self.config.SHAPE_ID_IDX
        shape_start_time_idx = self.config.SHAPE_MOVE_TIME_IDX
        shape_save_time_idx = self.config.SHAPE_SAVE_TIME_IDX
        shape_last_move_time_idx = self.config.SHAPE_MAX_MOVE_TIME_IDX

        for player_data in self.all_players_data:
            if not player_data[PARSED_ALL_SHAPES_KEY]:
                continue
            shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
            shapes_df[shape_last_move_time_idx] = shapes_df[shape_start_time_idx]
            #  grouping by consecutive duplicate shapes -
            #  two steps are grouped together if they have the same shape id and are consecutive in time
            shapes_df["group_id"] = self.group_consecutive_duplicates(shapes_df[shape_id_idx])

            shapes_df = (shapes_df
                         .groupby("group_id", as_index=False)
                         .agg({shape_id_idx: lambda x: x.iloc[0],
                               shape_start_time_idx: lambda x: x.iloc[0],
                               # if a player clicked "save" at least once during the grouped steps,
                               # take the earliest save time
                               # else, keep NaN
                               shape_save_time_idx: "min",
                               shape_last_move_time_idx: lambda x: x.iloc[-1]})
                         .drop(columns="group_id"))
            # fixes possible column reordering caused by agg()
            shapes = (shapes_df
                      .reindex(sorted(shapes_df.columns), axis="columns")
                      .values.tolist())

            # We check the first row. If it doesn't have enough columns to include shape_last_move_time_idx,
            # it means the column creation failed or was dropped.
            # Example: If index is 3, we need length 4. (0,1,2,3). If len is 3, 3 <= 3 is True -> ERROR.
            if len(shapes) > 0 and len(shapes[0]) <= shape_last_move_time_idx:
                player_id = player_data[PARSED_PLAYER_ID_KEY]
                msg = (f"CRITICAL ERROR: MRI Empty Move handling failed for player {player_id}.\n"
                       f"Expected column index {shape_last_move_time_idx} to exist (min {shape_last_move_time_idx + 1} columns), "
                       f"but found only {len(shapes[0])} columns.\n"
                       "Terminating pipeline to prevent data corruption.")
                raise CFGPipelineException(msg)

            player_data[PARSED_ALL_SHAPES_KEY] = shapes

    def add_explore_exploit(self):
        # 1. Standard Args
        shape_move_idx = self.config.SHAPE_MOVE_TIME_IDX
        shape_save_idx = self.config.SHAPE_SAVE_TIME_IDX
        #  this constant has remained the same in both MRI and standard logic
        # keeping it = 3.
        min_save = self.config.MIN_SAVE_FOR_EXPLOIT

        # 2. Check for MRI Mode
        is_mri_mode = self.config.SEGMENTATION_ALGORITHM == "MRI"

        for player_data in self.all_players_data:

            # --- PATH A: MRI Logic ---
            if is_mri_mode:
                # Retrieve MRI specific params
                min_efficiency = self.config.MIN_EFFICIENCY_FOR_EXPLOIT
                max_pace = self.config.MAX_PACE_FOR_MERGE

                # Call the dedicated MRI function
                explore, exploit, robust_median, max_pace_val = segment_explore_exploit_mri(
                    shapes=player_data[PARSED_ALL_SHAPES_KEY],
                    min_save_for_exploit=min_save,
                    min_efficiency=min_efficiency,
                    max_pace=max_pace,
                    shape_save_time_idx=shape_save_idx,
                    shape_move_time_idx=shape_move_idx,
                    shape_max_move_time_idx=self.config.SHAPE_MAX_MOVE_TIME_IDX,
                    shape_id_index=self.config.SHAPE_ID_IDX
                )

                # Save MRI stats
                player_data['robust_median_exploit_pace'] = robust_median
                player_data['robust_threshold_exploit_pace'] = max_pace_val

            else:
                # --- PATH B: Standard Logic (Untouched) ---
                explore, exploit = segment_explore_exploit(
                    shapes=player_data[PARSED_ALL_SHAPES_KEY],
                    shape_move_time_idx=shape_move_idx,
                    shape_save_time_idx=shape_save_idx,
                    min_save_for_exploit=min_save
                )

            # Common assignment
            player_data[EXPLORE_KEY] = explore
            player_data[EXPLOIT_KEY] = exploit

    def dump(self, path=POSTPARSER_OUTPUT_FILENAME, pretty=False):
        # dump post-parsed
        json_str = prettify_games_json(self.all_players_data) if pretty else json.dumps(self.all_players_data)
        with open(path, "w") as out_file:
            out_file.write(json_str)

        # dump config
        self.config.to_yaml(path)
