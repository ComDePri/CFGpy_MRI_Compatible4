import numpy as np
import pandas as pd
import json
import re
from datetime import datetime, timezone
from CFGpy.behavioral._utils import server_coords_to_binary_shape, prettify_games_json, CFGPipelineException
from CFGpy.behavioral._consts import (PARSED_PLAYER_ID_KEY, PARSED_TIME_KEY, PARSED_ALL_SHAPES_KEY,
                                      PARSED_CHOSEN_SHAPES_KEY, MERGED_ID_KEY, DEFAULT_ID, PARSER_OUTPUT_FILENAME)
from CFGpy.behavioral import Configuration


class Parser:
    old_date_format_with_placeholder = 'DateObject<{%Y, %m, %d, %H, %M, %S.%f}, "Instant", "Gregorian", 2.>'  # The actual format has '[' instead of '<' but it makes everything easier this way
    datetime_re = '"(DateObject\[\{\d+, \d+, \d+, \d+, \d+, \d+(?:\.\d+)?}, "Instant", "Gregorian", \d+\.\])"'  # Used to remove quotes from game strings
    parse_datetime_re_day = 'DateObject\[\{(\d+), (\d+), (\d+)}, "Day", "Gregorian", \d+\.\]'
    parse_datetime_re_second = 'DateObject\[\{(\d+), (\d+), (\d+), (\d+), (\d+), (\d+)}, "Second", "Gregorian", \d+\.\]'
    parse_datetime_re_millisecond = 'DateObject\[\{(\d+), (\d+), (\d+), (\d+), (\d+), (\d+)(.\d+)?}, "Instant", "Gregorian", \d+\.\]'
    datetime_sub_expressions = [
        parse_datetime_re_day,
        parse_datetime_re_second,
        parse_datetime_re_millisecond,
    ]

    def __init__(self, *, raw_data: pd.DataFrame, is_rm1: bool = False, config: Configuration = None):
        self.raw_data = raw_data
        self.config = config or Configuration.default(is_rm1=is_rm1)
        self.parsed_data = None

        self.include_in_id = list(self.config.INCLUDE_IN_PARSER_ID)
        self.parser_relevant_columns = [
            MERGED_ID_KEY,
            self.config.EVENT_TYPE,
            self.config.RAW_NEW_SHAPE,
            self.config.RAW_SHAPE,
            self.config.RAW_USER_TIME,
        ]
        self.shape_relevant_event_types = [
            self.config.SHAPE_MOVE_EVENT_TYPE,
            self.config.GALLERY_SAVE_EVENT_TYPE,
        ]

    @classmethod
    def from_file(cls, raw_data_filename: str, config=None):
        raw_data = pd.read_csv(raw_data_filename)
        return cls(raw_data, config)

    def parse(self):
        prepared_data = self._prepare_data()
        games_grouped_by_unique_id = prepared_data.groupby(self.config.UNIQUE_INTERNAL_ID_COLUMN)
        hard_filtered_games = games_grouped_by_unique_id.filter(self._apply_hard_filters)
        self.parsed_data = self._parse_all_player_games(hard_filtered_games)
        return self.parsed_data

    def dump(self, path=PARSER_OUTPUT_FILENAME, pretty=False):
        # dump parsed
        json_str = prettify_games_json(self.parsed_data) if pretty else json.dumps(self.parsed_data)
        with open(path, "w") as out_file:
            out_file.write(json_str)

        # dump config
        self.config.to_yaml(path)

    def _prepare_data(self):
        data = self.raw_data.copy()

        # 1. Standardize column names based on config
        # If the CSV has 'type', map it to what the parser expects
        if self.config.EVENT_TYPE in data.columns and self.config.EVENT_TYPE != "eventType":
            data['eventType'] = data[self.config.EVENT_TYPE]

        # 2. Backfill IDs (This is where Subject 057b is fixed)
        data = self.merge_id_columns(data)

        # 3. Expand JSON (MRI uses playerCustomData)
        json_col = self.config.PARSER_JSON_COLUMN
        if json_col in data.columns:
            data[json_col] = data[json_col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            # Flatten keys (like userProvidedId) into top-level columns
            all_keys = self.get_all_json_keys_from_csv_data(data)
            for key in all_keys:
                data[key] = data[json_col].apply(lambda d: d.get(key) if isinstance(d, dict) else np.nan)

        # 4. Final cleaning and sorting
        data[self.config.PARSER_TIME_COLUMN] = pd.to_datetime(data[self.config.PARSER_TIME_COLUMN], errors='coerce')
        data = data.sort_values(by=self.config.PARSER_TIME_COLUMN).reset_index(drop=True)

        return data

    def patchfix_csv_data(self, data):
        '''Small patchy bugfix for temporary problems'''
        # Bug no.1 sometimes player external id is this instead of a random number
        if 'playerExternalId' in data.columns: # For rm2 this column does not exist
            data.loc[data['playerExternalId'] == '${rand://int/100000:10000000}', 'playerExternalId'] = None

        # Bug no.2 sometimes the endPosition and shape columns switch places
        switched_column_indices = np.flatnonzero(
            data['customData.endPosition'].apply(lambda x: len(json.loads(x)) == 10 if (isinstance(x, str) and x.strip()) else False))
        data.loc[switched_column_indices, 'customData.shape'] = data.loc[
            switched_column_indices, 'customData.endPosition']
        data['customData.shape'] = data['customData.shape'].apply(
            lambda x: x if isinstance(x, list) 
            else json.loads(x) if isinstance(x, str) 
            else []).apply(lambda x: str(x) if len(x) == 10 else np.nan
        )

        return data

    def get_all_json_keys_from_csv_data(self, data):
        all_json_keys = np.concatenate(data[self.config.PARSER_JSON_COLUMN].apply(lambda x: tuple(x.keys())).unique())

        return set(all_json_keys)

    def merge_id_columns(self, data):
        data[MERGED_ID_KEY] = None

        # 1. THE UNIVERSAL SEARCH
        # We look for each key in the hierarchy defined in your config
        for id_key in self.config.PARSER_ID_COLUMNS:
            # Only work on rows that still need an ID
            missing = data[MERGED_ID_KEY].isna()
            if not missing.any():
                break

            # --- Strategy A: Flat Columns (Common in RM1/MRI CSVs) ---
            if id_key in data.columns:
                # Fill only the missing slots
                data.loc[missing, MERGED_ID_KEY] = data.loc[missing, id_key]

            # Refresh missing mask for Strategy B
            missing = data[MERGED_ID_KEY].isna()
            if not missing.any():
                break

            # --- Strategy B: Nested JSON (Common in RM2/MRI JSON blobs) ---
            json_col = self.config.PARSER_JSON_COLUMN
            if json_col in data.columns:
                # Extract values from dicts, ensuring we don't pick up 'nan' strings
                extracted = data.loc[missing, json_col].apply(
                    lambda d: d.get(id_key) if isinstance(d, dict) else None
                )
                data.loc[missing, MERGED_ID_KEY] = extracted

        # 2. SANITIZATION
        # Convert everything to string but map all variants of 'null' back to real NaNs
        # This prevents the 'nan' string from blocking the backfill
        data[MERGED_ID_KEY] = data[MERGED_ID_KEY].astype(str).replace(
            ['None', 'nan', 'null', '', 'NaN', 'nan', 'None'], np.nan
        )

        # 3. THE BACKFILL (Crucial for both RM1 MRI and RM2)
        # Propagates the ID from 'Save' rows to 'Move' rows within the same session
        session_col = self.config.UNIQUE_INTERNAL_ID_COLUMN
        if session_col in data.columns:
            data[MERGED_ID_KEY] = data.groupby(session_col)[MERGED_ID_KEY].transform(
                lambda x: x.ffill().bfill()
            )

        # 4. FINAL FALLBACK
        # Use DEFAULT_ID (usually 'No ID Found' or 'None') if everything failed
        data.loc[data[MERGED_ID_KEY].isna(), MERGED_ID_KEY] = DEFAULT_ID

        return data

    def merge_id_columns_GOOD_FOR_RM2(self, data):
        data[MERGED_ID_KEY] = None

        # 1. THE UNIVERSAL LOOP: Works for RM1 columns and RM2 dicts
        json_col = self.config.PARSER_JSON_COLUMN  # 'playerCustomData'

        for id_key in self.config.PARSER_ID_COLUMNS:
            missing = data[MERGED_ID_KEY].isna()
            if not missing.any():
                break

            # Strategy A: Check for flat columns (RM1 / RedMetrics 1)
            if id_key in data.columns:
                data.loc[missing, MERGED_ID_KEY] = data.loc[missing, id_key].astype(str)
                missing = data[MERGED_ID_KEY].isna()  # Refresh missing mask

            # Strategy B: Check inside the JSON blob (RM2 / RedMetrics 2)
            if missing.any() and json_col in data.columns:
                # We extract only the missing rows to save time
                extracted = data.loc[missing, json_col].apply(
                    lambda d: d.get(id_key) if isinstance(d, dict) else None
                )
                data.loc[missing, MERGED_ID_KEY] = extracted.astype(str)

        # 2. SANITIZATION: Remove 'nan' strings that Pandas .astype(str) creates
        data[MERGED_ID_KEY] = data[MERGED_ID_KEY].replace(['None', 'nan', 'null', '', 'NaN'], np.nan)

        # 3. THE BACKFILL: Crucial for MRI and RM2 where labels are only on 'Save' rows
        internal_session_col = self.config.UNIQUE_INTERNAL_ID_COLUMN
        if internal_session_col in data.columns:
            data[MERGED_ID_KEY] = data.groupby(internal_session_col)[MERGED_ID_KEY].transform(
                lambda x: x.ffill().bfill()
            )

        # 4. FINAL SAFETY
        data.loc[data[MERGED_ID_KEY].isna(), MERGED_ID_KEY] = "No ID Found"

        return data

    def merge_id_columns_WORKED_WELL_FOR_RM1(self, data):
        data[MERGED_ID_KEY] = None

        # 1. Pull the IDs from your configured columns (userProvidedId)
        for id_column in self.config.PARSER_ID_COLUMNS:
            if id_column in data.columns:
                missing_indices = data[MERGED_ID_KEY].isna()
                data.loc[missing_indices, MERGED_ID_KEY] = data[id_column].loc[missing_indices].astype(str)

        # 2. THE FIX: Group by the internal database ID and propagate the label
        # This takes "057b" from the Save row and gives it to all the Move rows in that session
        internal_session_col = self.config.UNIQUE_INTERNAL_ID_COLUMN
        if internal_session_col in data.columns:
            # We replace common string nulls with actual NaNs so fillna works
            data[MERGED_ID_KEY] = data[MERGED_ID_KEY].replace(['None', 'nan', ''], np.nan)

            # Forward-fill and Back-fill within each internal session group
            data[MERGED_ID_KEY] = data.groupby(internal_session_col)[MERGED_ID_KEY].transform(
                lambda x: x.ffill().bfill()
            )

        # 3. Final Fallback for rows that truly have no ID info
        missing_indices = data[MERGED_ID_KEY].isna()
        data.loc[missing_indices, MERGED_ID_KEY] = DEFAULT_ID

        return data

    def _apply_hard_filters(self, game):
        # Get the player ID from the current group (game)
        # We use iloc[0] because all rows in this group belong to the same player
        player_id = str(game[self.config.UNIQUE_INTERNAL_ID_COLUMN].iloc[0])

        # Check against the exclusion list from config
        if player_id in self.config.MANUALLY_EXCLUDED_IDS:
            return False
        return self.is_game_started(game)

    def is_game_started(self, game):
        return game[self.config.EVENT_TYPE].str.contains(self.config.TUTORIAL_END_EVENT_TYPE).sum() > 0

    def _parse_all_player_games(self, games):
        all_parsed_games = []
        for _, game in games.groupby(self.config.UNIQUE_INTERNAL_ID_COLUMN):
            parsed_game = self.parse_single_game(game)
            all_parsed_games.append(parsed_game)

        return all_parsed_games


    def parse_single_game(self, game_data):
        # TODO: Note that this version of the function truncates the games at 720 seconds (12 minutes) to match the Vanilla games
        parser_relevant_columns = self.parser_relevant_columns + self.include_in_id
        game_data = game_data[parser_relevant_columns]

        assert len(game_data[MERGED_ID_KEY].unique()) == 1
        player_id_field = game_data[MERGED_ID_KEY].iloc[0]

        # Identify the start search time (Tutorial End)
        game_start_time = game_data[game_data[self.config.EVENT_TYPE] == self.config.TUTORIAL_END_EVENT_TYPE].iloc[0][
            self.config.PARSER_TIME_COLUMN]

        # --- TIME CUTOFF LOGIC ---
        # Define 12 minutes (720 seconds) from the start
        CUTOFF_SECONDS = 720
        game_cutoff_time = game_start_time + pd.Timedelta(seconds=CUTOFF_SECONDS)

        # Filter: Keep only rows between start and the 12-minute mark
        game_data = game_data[
            (game_data[self.config.PARSER_TIME_COLUMN] >= game_start_time) &
            (game_data[self.config.PARSER_TIME_COLUMN] <= game_cutoff_time)
            ]
        # -------------------------

        # Filter for relevant move/save events within that 12-minute window
        game_data = game_data[game_data[self.config.EVENT_TYPE].isin(self.shape_relevant_event_types)]

        # Create the initial entry (starting shape) at t=0
        first_row = [player_id_field, self.config.SHAPE_MOVE_EVENT_TYPE,
                     self.config.FIRST_SHAPE_SERVER_COORDS, np.nan, game_start_time]

        # Ensure we include any additional ID fields in the placeholder row
        if self.include_in_id:
            for extra_col in self.include_in_id:
                # Add value from the first available row for these extra ID columns
                first_row.append(game_data[extra_col].iloc[0] if not game_data.empty else np.nan)

        first_row_df = pd.DataFrame([first_row], columns=game_data.columns)
        game_data = pd.concat([first_row_df, game_data], ignore_index=True)

        # Convert datetime objects to "seconds since start" (0 to 720)
        game_data[self.config.PARSER_TIME_COLUMN] = (game_data[self.config.PARSER_TIME_COLUMN] - game_start_time).apply(
            lambda time_delta: time_delta.total_seconds())

        # Binary conversion for shapes
        game_data[self.config.SHAPE_MOVE_COLUMN] = game_data[self.config.SHAPE_MOVE_COLUMN].apply(
            server_coords_to_binary_shape)

        # Gallery Save Logic: link saves to the preceding move
        game_data[self.config.GALLERY_SAVE_TIME_COLUMN] = None
        gallery_save_indices = game_data[game_data[self.config.SHAPE_MOVE_COLUMN].isna()].index

        # Ensure we don't try to index out of bounds if a save is the very first row
        valid_save_indices = gallery_save_indices[gallery_save_indices > 0]
        game_data.loc[valid_save_indices - 1, self.config.GALLERY_SAVE_TIME_COLUMN] = game_data.loc[
            valid_save_indices, self.config.PARSER_TIME_COLUMN].values

        # Clean up: remove the actual 'save' rows now that their timestamps are mapped to moves
        game_data = game_data[game_data[self.config.EVENT_TYPE].isin([self.config.SHAPE_MOVE_EVENT_TYPE])]

        actions = game_data.loc[:, self.config.PARSED_GAME_HEADERS]

        if self.include_in_id:
            player_id_field = [game_data[MERGED_ID_KEY].iloc[0]] + [game_data[col].iloc[0] for col in
                                                                    self.include_in_id]

        parsed_game = {
            PARSED_PLAYER_ID_KEY: player_id_field,
            PARSED_TIME_KEY: game_start_time.timestamp(),
            PARSED_ALL_SHAPES_KEY: actions.values.tolist(),
        }

        return parsed_game

    def parse_single_game_ORIG_WITHOUT_TIME_CUTOFF(self, game_data):
        parser_relevant_columns = self.parser_relevant_columns + self.include_in_id
        game_data = game_data[parser_relevant_columns]

        assert len(game_data[MERGED_ID_KEY].unique()) == 1
        player_id_field = game_data[MERGED_ID_KEY].iloc[0]
        game_start_time = game_data[game_data[self.config.EVENT_TYPE] == self.config.TUTORIAL_END_EVENT_TYPE].iloc[0][
            self.config.PARSER_TIME_COLUMN]

        game_data = game_data[game_data[self.config.PARSER_TIME_COLUMN] >= game_start_time]
        game_data = game_data[game_data[self.config.EVENT_TYPE].isin(self.shape_relevant_event_types)]
        first_row = [player_id_field, self.config.SHAPE_MOVE_EVENT_TYPE,
                     self.config.FIRST_SHAPE_SERVER_COORDS, np.nan, game_start_time]
        first_row_df = pd.DataFrame([first_row], columns=game_data.columns)
        game_data = pd.concat([first_row_df, game_data], ignore_index=True)

        game_data[self.config.PARSER_TIME_COLUMN] = (game_data[self.config.PARSER_TIME_COLUMN] - game_start_time).apply(
            lambda time_delta: time_delta.total_seconds())
        game_data[self.config.SHAPE_MOVE_COLUMN] = game_data[self.config.SHAPE_MOVE_COLUMN].apply(
            server_coords_to_binary_shape)

        game_data[self.config.GALLERY_SAVE_TIME_COLUMN] = None

        gallery_save_indices = game_data[self.config.SHAPE_MOVE_COLUMN].isna()[
            game_data[self.config.SHAPE_MOVE_COLUMN].isna()].index # TODO: retrieves indices of all the None values
        game_data.loc[gallery_save_indices - 1, self.config.GALLERY_SAVE_TIME_COLUMN] = game_data.loc[
            gallery_save_indices, self.config.PARSER_TIME_COLUMN].values
        # Now that we have the save time in all move rows, we can get rid of save rows:
        game_data = game_data[game_data[self.config.EVENT_TYPE].isin([self.config.SHAPE_MOVE_EVENT_TYPE])]
        actions = game_data.loc[:, self.config.PARSED_GAME_HEADERS]
        if self.include_in_id:
            player_id_field = [game_data[MERGED_ID_KEY].iloc[0]] + self.include_in_id
        parsed_game = {
            PARSED_PLAYER_ID_KEY: player_id_field,
            PARSED_TIME_KEY: game_start_time.timestamp(),
            PARSED_ALL_SHAPES_KEY: actions.values.tolist(),
        }

        return parsed_game

    @classmethod
    def translate_parsed_results_to_mathematica(cls, json_format):
        games_in_old_format = []
        for entry in json_format:
            player_id = entry[PARSED_PLAYER_ID_KEY]
            player_start_time = entry[PARSED_TIME_KEY]
            player_actions = entry[PARSED_ALL_SHAPES_KEY]

            old_format_actions = [
                [list(map(str, action[0])), action[1]] if action[2] is None else [list(map(str, action[0])), action[1],
                                                                                  action[2]]
                for action in player_actions
            ]
            old_format_entry = [
                player_id,
                datetime.strftime(datetime.fromtimestamp(player_start_time, tz=timezone.utc),
                                  cls.old_date_format_with_placeholder),
                old_format_actions,
                "",
            ]
            games_in_old_format.append(old_format_entry)

        parsed_data_in_old_format = '\n'.join(
            [cls.replace_chars_to_old_format(json.dumps(game_in_old_format)) for game_in_old_format in
             games_in_old_format])

        return parsed_data_in_old_format

    @classmethod
    def replace_chars_to_old_format(cls, old_format_json_string):
        brackets_replaced = old_format_json_string.replace('[', '{').replace(']', '}')
        brackets_replaced = brackets_replaced.replace('<', '[').replace('>', ']').replace('\\', '')

        quotes_removed = re.sub(pattern=cls.datetime_re, repl='\\g<1>', string=brackets_replaced)

        return quotes_removed

    @classmethod
    def translate_mathematica_to_python(cls, mathematica_path):
        with open(mathematica_path, 'r') as f:
            data = f.read()

        games = data.split('\n')

        game_timestamps = [cls.parse_date_from_game_string(game).timestamp() for game in games]
        games = [cls.replace_datetime_with_timestamp(game_string=games[i], timestamp=game_timestamps[i]) for i in
                 range(len(games))]
        games = [re.sub(pattern='(\d\.)([\[\]\{\},])', repl='\\g<1>0\\g<2>', string=game) for game in
                 games]  # There's a bug here if we have a user with the string "0.[" in its id
        games = [game.replace('{', '[').replace('}', ']').replace('$Failed', '"$Failed"') for game in games]
        games = [json.loads(game) for game in games]

        json_format_games = []
        for game in games:
            game_id = game[0].replace('[', '{').replace(']', '}')  # { and } sometimes appear in Mathematica-parsed ids
            absolute_start_time = game[1]
            actions = [
                [list(map(int, action[0])), action[1], action[2]] if len(action) == 3 else
                [list(map(int, action[0])), action[1], None]
                for action in game[2]
            ]

            chosen_shapes = []
            if len(game) == 4 and game[3] != "":
                chosen_shapes = game[3]
                if type(chosen_shapes) is not list:
                    chosen_shapes = [chosen_shapes]

            json_format_game = {
                PARSED_PLAYER_ID_KEY: game_id,
                PARSED_TIME_KEY: absolute_start_time,
                PARSED_CHOSEN_SHAPES_KEY: chosen_shapes,
                PARSED_ALL_SHAPES_KEY: actions,
            }

            json_format_games.append(json_format_game)

        return json_format_games

    @classmethod
    def parse_date_from_game_string(cls, game):
        game_date_string = re.findall(cls.parse_datetime_re_millisecond, game)
        if game_date_string == []:
            game_date_string = re.findall(cls.parse_datetime_re_second, game)
            if game_date_string == []:
                game_date_string = re.findall(cls.parse_datetime_re_day, game)
                if game_date_string == []:
                    raise CFGPipelineException('Was not able to parse the date in the following game:', game)
        else:
            year = int(game_date_string[0][0])
            month = int(game_date_string[0][1])
            day = int(game_date_string[0][2])
            hour = int(game_date_string[0][3])
            minute = int(game_date_string[0][4])
            second = int(game_date_string[0][5])
            microsecond = int(game_date_string[0][6][1:7])

            return datetime(year, month, day, hour, minute, second, microsecond)

        return datetime(*map(int, game_date_string[0]))

    @classmethod
    def replace_datetime_with_timestamp(cls, game_string, timestamp):
        if type(timestamp) is not str:
            timestamp = str(timestamp)

        for datetime_sub_expression in cls.datetime_sub_expressions:
            replaced_string = re.sub(pattern=datetime_sub_expression, repl=timestamp, string=game_string)
            if replaced_string != game_string:
                return replaced_string

        raise CFGPipelineException('Was not able to replace the DateObject with a timestamp in the following game:',
                                   game_string)
