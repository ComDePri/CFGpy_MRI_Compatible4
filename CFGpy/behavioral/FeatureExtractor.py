import numpy as np
import pandas as pd
from datetime import datetime
from CFGpy.behavioral.data_interfaces import PostparsedDataset
from CFGpy.behavioral._consts import (FEATURES_ID_KEY, FEATURES_START_TIME_KEY, N_CLUSTERS_KEY, GAME_DURATION_KEY,
                                      N_MOVES_KEY, LONGEST_PAUSE_KEY, MEDIAN_EXPLORE_LENGTH_KEY, N_GALLERIES_KEY,
                                      SELF_AVOIDANCE_KEY, EXPLORE_EFFICIENCY_KEY, EXPLOIT_EFFICIENCY_KEY,
                                      MEDIAN_EXPLOIT_LENGTH_KEY, AVERAGE_SPEED_KEY, FRACTION_GALLERY_IN_EXPLORE_KEY,
                                      FRACTION_TIME_IN_EXPLORE_KEY, EFFICIENCY_RATIO_KEY, EXPLORE_SPEED_KEY,
                                      EXPLOIT_SPEED_KEY, DEFAULT_FINAL_OUTPUT_FILENAME, EXCLUSION_REASON_KEY,
                                      STEP_ORIG_KEY, FRACTION_STEPS_UNIQUELY_COVERED_KEY, GALLERY_ORIG_KEY,
                                      GALLERY_ORIG_EXPLORE_KEY, GALLERY_ORIG_EXPLOIT_KEY,
                                      FRACTION_GALLERIES_UNIQUELY_COVERED_KEY, FRACTION_CLUSTERS_IN_GC_KEY,
                                      FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLORE_KEY,
                                      FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLOIT_KEY, N_CLUSTERS_IN_GC_KEY,
                                      ABSOLUTE_FEATURES_MESSAGE, RELATIVE_FEATURES_MESSAGE, EXPLORE_OUTLIER_REASON,
                                      EXPLOIT_OUTLIER_REASON, NO_EXPLOIT_EXCLUSION_REASON, MANUAL_EXCLUSION_REASON,
                                      GAME_LENGTH_EXCLUSION_REASON, GAME_DURATION_EXCLUSION_REASON,
                                      PAUSE_EXCLUSION_REASON, SAMPLE_RELATIVE_FEATURES_LABEL,
                                      ROBUST_MEDIAN_PACE_KEY, ROBUST_THRESHOLD_KEY)
from CFGpy.behavioral import Configuration
from CFGpy.behavioral._utils import load_json, is_semantic_connection
from functools import reduce
from scipy.stats import zscore
from CFGpy.utils import get_vanilla_stats, step_orig_map_factory, gallery_orig_map_factory
from tqdm import tqdm


def _get_frac_uniquely_covered(player_objects, objects_not_uniquely_covered):
    set_player_objects = set(player_objects)
    n_unique_player_objects = len(set_player_objects)
    if not n_unique_player_objects:
        return None

    n_not_uniquely_covered = len(set_player_objects & set(objects_not_uniquely_covered))
    frac_not_uniquely_covered = n_not_uniquely_covered / n_unique_player_objects
    frac_uniquely_covered = 1 - frac_not_uniquely_covered
    return frac_uniquely_covered


class FeatureExtractor:
    def __init__(self, *, preprocessed_data,  is_rm1: bool = False,  is_mri: bool = False,
                 config: Configuration = None):
        self.config = config if config is not None else Configuration.default(is_rm1=is_rm1, is_mri=is_mri)
        self.input_data = PostparsedDataset(input_data=preprocessed_data, config=config)
        self.all_absolute_features = None
        self.output_df = None
        self.exclusions = pd.DataFrame(columns=[FEATURES_ID_KEY, EXCLUSION_REASON_KEY])
    
    @classmethod
    def from_json(cls, path: str, config=Configuration.default()):
        return cls(preprocessed_data=load_json(path), config=config)

    def _drop_nonfirst_games_from_list(self):
        """
        Keeps the session with the best data quality for each unique Subject ID.
        Prioritizes Gallery Count first, then Move Count.
        """
        best_games = {}

        for player in self.input_data.players_data:
            # Identity: Fallback between .id and .player_id
            pid = getattr(player, 'id', getattr(player, 'player_id', None))
            subject_id = str(pid).strip()

            # Quality Score: Prioritize Galleries, then Moves
            g_score = len(getattr(player, 'galleries', []))
            m_score = len(getattr(player, 'delta_move_times', []))
            current_quality = (g_score, m_score)

            if subject_id not in best_games:
                best_games[subject_id] = (player, current_quality)
            else:
                _, existing_quality = best_games[subject_id]
                # Tuple comparison: (10, 100) > (0, 200) -> Correctly picks the game with galleries
                if current_quality > existing_quality:
                    best_games[subject_id] = (player, current_quality)

        self.input_data.players_data = [val[0] for val in best_games.values()]
        print(f"Filter complete: {len(self.input_data.players_data)} unique subjects remain.")


    def extract(self, verbose=False):
        # 1. Sync ID access (MRI JSON uses 'id', Standard uses 'player_id')
        for p in self.input_data.players_data:
            if not hasattr(p, 'id') and hasattr(p, 'player_id'): p.id = p.player_id
            if not hasattr(p, 'player_id') and hasattr(p, 'id'): p.player_id = p.id

        # 2. Selection & Initial Audit
        self._drop_nonfirst_games_from_list()

        # 3. Absolute Pass
        self.all_absolute_features = self._extract_absolute_features(verbose)
        self.output_df = self.all_absolute_features.copy()

        # 4. Vanilla Relative
        vanilla_rel = self._extract_relative_features(get_vanilla_stats(), verbose=verbose)
        self.output_df = self.output_df.merge(vanilla_rel, on=FEATURES_ID_KEY)

        # 5. Pruning & Alignment (The Gemini Sync Fix)
        self._apply_soft_filters()
        surviving_ids = set(self.output_df[FEATURES_ID_KEY].astype(str))

        # Ensure the list of objects matches the survivors in the table
        self.input_data.players_data = [p for p in self.input_data.players_data if str(p.id) in surviving_ids]

        # Enforce
        # Ensure the objects list and the table are 100% identical in size and ID
        surviving_ids = self.output_df[FEATURES_ID_KEY].astype(str).tolist()
        self.input_data.players_data = [
            p for p in self.input_data.players_data if str(p.id) in surviving_ids
        ]

        print(f"[SYNC] Objects: {len(self.input_data.players_data)} | Table: {len(self.output_df)}")

        # 6. Sample-based Relative
        stats = self.input_data.get_stats()
        sample_rel = self._extract_relative_features(stats, verbose=verbose, label=SAMPLE_RELATIVE_FEATURES_LABEL)

        # Final Merge (Force string IDs for safety)
        sample_rel[FEATURES_ID_KEY] = sample_rel[FEATURES_ID_KEY].astype(str)
        self.output_df[FEATURES_ID_KEY] = self.output_df[FEATURES_ID_KEY].astype(str)
        self.output_df = self.output_df.merge(sample_rel, on=FEATURES_ID_KEY, how="left")

        print("\n" + "=" * 50)
        print("FINAL SAMPLE-RELATIVE BIOPSY")
        print("=" * 50)

        # 1. Check if the Sample Results Table is actually empty
        print(f"Sample Results Table Length: {len(sample_rel)}")

        # 2. Check the ID types in both tables
        table_id_sample = self.output_df[FEATURES_ID_KEY].iloc[0]
        result_id_sample = sample_rel[FEATURES_ID_KEY].iloc[0]
        print(f"Table ID Type: {type(table_id_sample)} (Value: {table_id_sample})")
        print(f"Result ID Type: {type(result_id_sample)} (Value: {result_id_sample})")

        # 3. Check the "Overlap" - Do any IDs actually match?
        overlap = set(self.output_df[FEATURES_ID_KEY].astype(str)) & set(sample_rel[FEATURES_ID_KEY].astype(str))
        print(f"ID Overlap Count: {len(overlap)} / {len(self.output_df)}")

        # 4. Check the "Stats" - Is the probability map actually empty?
        # stats is a tuple: (steps_not_uniquely_covered, step_counter, ...)
        _, step_counter, _, gallery_counter, _ = stats
        print(f"Total Unique Steps in Sample Map: {len(step_counter)}")
        print(f"Total Unique Galleries in Sample Map: {len(gallery_counter)}")

        if len(step_counter) == 0:
            print("[CRITICAL] The Sample Map is EMPTY. The extractor isn't seeing any moves!")
        print("=" * 50 + "\n")

        # Check a few actual values from the result table before the merge
        valid_values = sample_rel.iloc[:, 1:].notna().sum().sum()
        print(f"Total non-NaN values in Sample Results: {valid_values}")
        print(f"Sample Results Columns: {sample_rel.columns.tolist()}")

        return self.output_df

    def extract_BEFORE_ROEYS_CHANGES(self, verbose=False):
        self._drop_nonfirst_games() # TODO: ROEY MOVED IT HERE BC I THINK IT MAKES MORE SENSE BEFORE CALCULATING ANYTHING
        self.all_absolute_features = self._extract_absolute_features(verbose)
        self.output_df = self.all_absolute_features.copy()
        #self._drop_nonfirst_games()
        vanilla_relative_features = self._extract_relative_features(get_vanilla_stats(), verbose=verbose)
        self.output_df = self.output_df.merge(vanilla_relative_features, on=FEATURES_ID_KEY)
        self._apply_soft_filters()
        sample_relative_features = self._extract_relative_features(self.input_data.get_stats(), verbose=verbose,
                                                                   label=SAMPLE_RELATIVE_FEATURES_LABEL)
        self.output_df = self.output_df.merge(sample_relative_features, on=FEATURES_ID_KEY, how="left")
        return self.output_df

    def dump(self, path=DEFAULT_FINAL_OUTPUT_FILENAME):
        self.output_df.to_csv(path, index=False)  # reorder columns
        self.exclusions.to_csv(f"{path}_exclusions.csv", index=False)
        self.config.to_yaml(path)

        # TODO: document all filtered ids and filtering criteria
        # TODO: write html with dashboards to inspect data quality and some summary stats

    def is_cluster_in_GC(self, cluster, GC):
        for GC_cluster in GC:
            if is_semantic_connection(cluster, GC_cluster, self.config.MIN_OVERLAP_FOR_SEMANTIC_CONNECTION):
                return True

        return False

    def get_all_absolute_features(self):
        return self.all_absolute_features

    def _drop_nonfirst_games_ROEY_IS_TRYING_THIS_FIX_FROM_GEMINI(self, min_duration_seconds=600):
        """
        Keeps the 'best' valid game for each player.
        Prioritizes the first game that meets the duration threshold.
        """
        # 1. Update the DataFrame: Sort by time, but prioritize games that meet the threshold
        # We create a temporary 'is_valid' helper for sorting
        self.output_df['is_long_enough'] = self.output_df[GAME_DURATION_KEY] >= min_duration_seconds

        self.output_df = (self.output_df
                          .sort_values(by=['is_long_enough', FEATURES_START_TIME_KEY],
                                       ascending=[False, True])  # Valid games first, then by time
                          .drop_duplicates(subset=[FEATURES_ID_KEY], keep="first")
                          .drop(columns=['is_long_enough'])
                          .reset_index(drop=True))

        # 2. Sync the Input Data (the objects)
        # We must tell input_data to keep ONLY the indices that survived in output_df
        surviving_ids = set(self.output_df[FEATURES_ID_KEY].unique())

        # This is where the 225 vs 165 usually happens.
        # Ensure the underlying objects are pruned to match the Table EXACTLY.
        self.input_data.keep_only_indices(self.output_df[FEATURES_INDEX_KEY].tolist())

    #def _drop_nonfirst_games_BEFORE_ROEY_ADDED_A_TIMING_TEST(self):
    def _drop_nonfirst_games(self):
        """
        Keeps only the first game from each player. Allows functions downstream to assume unique IDs.
        """
        # TODO: this is the versino that Roey removed in favor of the one above, which leeps non first games if they were long enough (more than 10 minutes)
        self.input_data.drop_non_first_games()
        self.output_df = (self.output_df.
                          sort_values(by=[FEATURES_START_TIME_KEY], ascending=True).
                          drop_duplicates(subset=[FEATURES_ID_KEY], keep="first").
                          reset_index(drop=True))

    def _apply_soft_filters(self):
        """
        Applies absolute filters first, then sample-relative filters with the remaining sample.
        """
        for filter_getter in (self._get_absolute_filters, self._get_sample_relative_filters):
            masks, reasons = filter_getter()
            self._update_exclusion_info(masks, reasons)

            # Combine masks into a single boolean array
            is_excluded = reduce(np.logical_or, masks)

            # 1. Capture surviving indices BEFORE we filter output_df
            # This ensures we know exactly which Player Objects in players_data
            # correspond to the 'False' values in is_excluded.
            surviving_indices = self.output_df.index[~is_excluded].tolist()

            # 2. Filter the Table (Standard original logic)
            self.output_df = self.output_df.loc[~is_excluded].reset_index(drop=True)

            # 3. Filter the Objects (The manual sync that prevents the IndexingError)
            # This replaces 'self.input_data.filter(~is_excluded)'
            self.input_data.players_data = [self.input_data.players_data[i] for i in surviving_indices]

        print(f"Soft filters complete. Final survivors: {len(self.output_df)}")


    def _get_absolute_filters(self):
        """
        Absolute filters are based on absolute features, can be applied independently of each other. Each filter is
        represented by a textual description and a mask with **True for players to exclude**, False for players to keep.
        :return: masks, reasons.
        """
        reasons = (MANUAL_EXCLUSION_REASON, NO_EXPLOIT_EXCLUSION_REASON, GAME_LENGTH_EXCLUSION_REASON,
                   GAME_DURATION_EXCLUSION_REASON, PAUSE_EXCLUSION_REASON)
        masks = (self.output_df[FEATURES_ID_KEY].isin(self.config.MANUALLY_EXCLUDED_IDS),
                 self.output_df[N_CLUSTERS_KEY] < self.config.MIN_N_CLUSTERS,
                 self.output_df[N_MOVES_KEY] < self.config.MIN_N_MOVES,
                 self.output_df[GAME_DURATION_KEY] < self.config.MIN_GAME_DURATION_SEC,
                 self.output_df[LONGEST_PAUSE_KEY] > self.config.MAX_PAUSE_DURATION_SEC)

        return masks, reasons

    def _get_sample_relative_filters(self):
        """
        Each filter is represented by a textual description and a mask with **True for players to exclude**, False for
        players to keep.
        :return: masks, reasons.
        """
        reasons = (EXPLORE_OUTLIER_REASON, EXPLOIT_OUTLIER_REASON)
        zscores = self.output_df[[MEDIAN_EXPLORE_LENGTH_KEY, MEDIAN_EXPLOIT_LENGTH_KEY]].apply(zscore)
        masks = (abs(zscores[MEDIAN_EXPLORE_LENGTH_KEY]) > self.config.MAX_ZSCORE_FOR_OUTLIERS,
                 abs(zscores[MEDIAN_EXPLOIT_LENGTH_KEY]) > self.config.MAX_ZSCORE_FOR_OUTLIERS)

        return masks, reasons

    def _update_exclusion_info(self, masks, reasons):
        """
        Updates self.to_exclude based on filters results.
        :param masks: a collection of masks, each has **True for players to exclude**, false for players to keep.
        :param reasons: a collection of strings describing exclusion reasons for the masks.
        """
        for is_excluded, reason in zip(masks, reasons):
            ids_to_exclude = self.output_df.loc[is_excluded, FEATURES_ID_KEY]
            current_exclusion = pd.DataFrame({
                FEATURES_ID_KEY: ids_to_exclude,
                EXCLUSION_REASON_KEY: [reason] * len(ids_to_exclude)
            })
            self.exclusions = pd.concat((self.exclusions, current_exclusion))

    def _extract_absolute_features(self, verbose=True):
        n_galleries_in_explore = []
        total_explore_times = []
        total_exploit_times = []
        total_explore_lengths = []
        total_exploit_lengths = []

        iterator = self.input_data
        if verbose:
            print(ABSOLUTE_FEATURES_MESSAGE)
            iterator = tqdm(iterator)

        absolute_features = []
        for player_data in iterator:
            # IDENTITY FIX: Ensure ID is accessible
            p_id = getattr(player_data, 'id', getattr(player_data, 'player_id', "UNKNOWN"))

            # 1. Pre-calculations
            explore_lengths = [end - start for start, end in player_data.explore_slices]
            exploit_lengths = [end - start for start, end in player_data.exploit_slices]
            is_gallery = player_data.get_gallery_mask()
            n_galleries = sum(is_gallery)

            # 2. NO-SKIP LOGIC: Instead of 'continue', we handle 0 galleries gracefully
            if n_galleries == 0:
                print(f"[DEBUG] Subject {p_id} has 0 galleries. Providing NaN absolute features.")
                # We append a placeholder so the merge indices stay aligned
                absolute_features.append({FEATURES_ID_KEY: p_id})
                n_galleries_in_explore.append(np.nan)
                total_explore_times.append(np.nan)
                total_exploit_times.append(np.nan)
                total_explore_lengths.append(np.nan)
                total_exploit_lengths.append(np.nan)
                continue

            is_explore = player_data.get_explore_mask()

            # Data collection for later vectorized operations
            n_galleries_in_explore.append(sum(is_gallery & is_explore))
            total_explore_times.append(player_data.total_explore_time())
            total_exploit_times.append(player_data.total_exploit_time())
            total_explore_lengths.append(sum(explore_lengths))
            total_exploit_lengths.append(sum(exploit_lengths))

            # MRI-specific robust metrics
            robust_median = getattr(player_data, ROBUST_MEDIAN_PACE_KEY, np.nan)
            robust_threshold = getattr(player_data, ROBUST_THRESHOLD_KEY, np.nan)

            # Player-wise calculations
            explore_efficiency, exploit_efficiency = player_data.get_efficiency()
            absolute_features.append({
                FEATURES_ID_KEY: p_id,
                FEATURES_START_TIME_KEY: datetime.fromtimestamp(player_data.start_time).isoformat() if hasattr(
                    player_data, 'start_time') else None,
                GAME_DURATION_KEY: player_data.get_last_action_time(),
                N_MOVES_KEY: len(player_data),
                N_GALLERIES_KEY: n_galleries,
                SELF_AVOIDANCE_KEY: player_data.get_self_avoidance(),
                N_CLUSTERS_KEY: len(player_data.exploit_slices),
                EXPLORE_EFFICIENCY_KEY: explore_efficiency,
                EXPLOIT_EFFICIENCY_KEY: exploit_efficiency,
                MEDIAN_EXPLORE_LENGTH_KEY: np.median(explore_lengths) if explore_lengths else np.nan,
                MEDIAN_EXPLOIT_LENGTH_KEY: np.median(exploit_lengths) if exploit_lengths else np.nan,
                LONGEST_PAUSE_KEY: player_data.get_max_pause_duration(),
                ROBUST_MEDIAN_PACE_KEY: robust_median,
                ROBUST_THRESHOLD_KEY: robust_threshold,
            })

        # Vectorized operations
        features_df = pd.DataFrame(absolute_features)

        # Safe division to prevent crashes on empty subjects
        with np.errstate(divide='ignore', invalid='ignore'):
            features_df[AVERAGE_SPEED_KEY] = features_df[N_MOVES_KEY] / features_df[GAME_DURATION_KEY]
            features_df[FRACTION_GALLERY_IN_EXPLORE_KEY] = pd.Series(n_galleries_in_explore) / features_df[
                N_GALLERIES_KEY]
            features_df[FRACTION_TIME_IN_EXPLORE_KEY] = pd.Series(total_explore_times) / features_df[GAME_DURATION_KEY]
            features_df[EFFICIENCY_RATIO_KEY] = features_df[EXPLORE_EFFICIENCY_KEY] / features_df[
                EXPLOIT_EFFICIENCY_KEY]
            features_df[EXPLORE_SPEED_KEY] = pd.Series(total_explore_lengths) / pd.Series(total_explore_times)
            features_df[EXPLOIT_SPEED_KEY] = pd.Series(total_exploit_lengths) / pd.Series(total_exploit_times)

        return features_df

    def _extract_relative_features(self, stats, label=None, verbose=False):
        steps_not_uniquely_covered, step_counter, galleries_not_uniquely_covered, gallery_counter, GC = stats
        label_ext = f" ({label})" if label else ""

        step_orig_map = step_orig_map_factory(step_counter, alpha=self.config.STEP_ORIG_PSEUDOCOUNT,
                                              d=self.config.STEP_ORIG_N_CATEGORIES)
        gallery_orig_map = gallery_orig_map_factory(gallery_counter, alpha=self.config.GALLERY_ORIG_PSEUDOCOUNT,
                                                    d=self.config.GALLERY_ORIG_N_CATEGORIES)

        iterator = self.input_data
        if verbose:
            print(RELATIVE_FEATURES_MESSAGE.format(label_ext))
            iterator = tqdm(iterator)

        relative_features = []
        for player_data in iterator:
            p_id = getattr(player_data, 'id', getattr(player_data, 'player_id', "UNKNOWN"))

            # Initialize empty results for this player
            p_results = {FEATURES_ID_KEY: p_id}

            try:
                # 1. Step Logic
                steps = player_data.get_steps()
                step_orig = [step_orig_map[step] for step in steps]
                p_results[f"{STEP_ORIG_KEY}{label_ext}"] = np.mean(step_orig) if step_orig else np.nan
                p_results[f"{FRACTION_STEPS_UNIQUELY_COVERED_KEY}{label_ext}"] = _get_frac_uniquely_covered(steps,
                                                                                                            steps_not_uniquely_covered)

                # 2. Gallery Logic (MRI and Standard)
                gallery_ids = player_data.get_gallery_ids()
                gallery_orig = np.array([gallery_orig_map[shape_id] for shape_id in gallery_ids])

                is_gallery = player_data.get_gallery_mask()
                explore_mask = player_data.get_explore_mask()

                # Check if we have galleries to calculate gallery-relative stats
                if len(gallery_ids) > 0:
                    is_explore_given_gallery = explore_mask[is_gallery]
                    is_exploit_given_gallery = ~is_explore_given_gallery

                    p_results[f"{GALLERY_ORIG_KEY}{label_ext}"] = np.mean(gallery_orig)

                    # Safe Slice Means
                    p_results[f"{GALLERY_ORIG_EXPLORE_KEY}{label_ext}"] = np.mean(
                        gallery_orig[is_explore_given_gallery]) if any(is_explore_given_gallery) else np.nan
                    p_results[f"{GALLERY_ORIG_EXPLOIT_KEY}{label_ext}"] = np.mean(
                        gallery_orig[is_exploit_given_gallery]) if any(is_exploit_given_gallery) else np.nan

                    # Uniqueness
                    p_results[f"{FRACTION_GALLERIES_UNIQUELY_COVERED_KEY}{label_ext}"] = _get_frac_uniquely_covered(
                        gallery_ids, galleries_not_uniquely_covered)
                    p_results[
                        f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLORE_KEY}{label_ext}"] = _get_frac_uniquely_covered(
                        gallery_ids[is_explore_given_gallery], galleries_not_uniquely_covered) if any(
                        is_explore_given_gallery) else np.nan
                    p_results[
                        f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLOIT_KEY}{label_ext}"] = _get_frac_uniquely_covered(
                        gallery_ids[is_exploit_given_gallery], galleries_not_uniquely_covered) if any(
                        is_exploit_given_gallery) else np.nan

                # 3. Cluster Logic
                exploit_clusters = player_data.get_exploit_clusters()
                n_clusters_in_GC = sum([self.is_cluster_in_GC(cluster, GC) for cluster in exploit_clusters])

                p_results[f"{N_CLUSTERS_IN_GC_KEY}{label_ext}"] = n_clusters_in_GC
                p_results[f"{FRACTION_CLUSTERS_IN_GC_KEY}{label_ext}"] = (n_clusters_in_GC / len(
                    player_data.exploit_slices)) if player_data.exploit_slices else np.nan

            except Exception as e:
                print(f"[ERROR] Failed relative extraction for {p_id}: {e}")
                # We still append the dict with only the ID to keep the merge healthy

            relative_features.append(p_results)

        return pd.DataFrame(relative_features)
