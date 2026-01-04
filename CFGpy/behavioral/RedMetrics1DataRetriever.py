import json
import os
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests

from CFGpy.behavioral import Configuration, DataRetriever
from CFGpy.behavioral._consts import (DATA_RETRIEVER_OUTPUT_FILENAME, CONFIG_URL_MISMATCH_ERROR)

try:
    from CFGpy.utils._nas_path import get_nas_path
except ImportError:
    # Fallback for users who don't have the internal NAS utility
    def get_nas_path():
        return ""


class RedMetrics1DataRetriever(DataRetriever):
    def __init__(self, *, game_name: str | None = None, game_id: str | None = None,
                 game_version_ids: list[str] | None = None,
                 output_filename: str = DATA_RETRIEVER_OUTPUT_FILENAME,
                 config: Configuration = None, csv_directory: str = None,
                 input_url: str = None) -> None:
        """
        :param game_name: The game name (optional metadata).
        :param game_id: The game id (optional metadata).
        :param output_filename: filename for output.
        :param config: a Configuration file.
        :param csv_directory: Directory containing the CSV files (events.csv, players.csv).
        :param input_url: URL input to get the data from the RM1 API.
        """
        super().__init__(game_name=game_name, game_id=game_id, output_filename=output_filename,
                         config=config if config is not None else Configuration.default(is_rm1=True))

        self._game_version_ids = self._config.GAME_VERSION_IDS or game_version_ids

        # Mode A: URL provided
        if input_url:
            print(f"Initializing retrieval from URL: {input_url}...")
            # Download data and get metadata
            downloaded_meta = self._download_and_cache(input_url)
            self._csv_directory = Path(downloaded_meta['csv_directory'])

            # If game_id/name were not provided in CLI, use what we found in the URL/CSV
            if not self._game_id:
                self._game_id = downloaded_meta['game_id']
                self._config.GAME_ID = self._game_id
            if not self._game_name:
                self._game_name = downloaded_meta['game_name']
                self._config.GAME_NAME = self._game_name

        # Mode B: Folder provided explicitly
        elif csv_directory:
            self._csv_directory = Path(csv_directory)

        # Mode C: Legacy NAS (Construct path dynamically)
        else:
            try:
                self.nas_path = get_nas_path()
                self.csv_path = os.path.join(self.nas_path, "Projects", "CFG", "all_data_from_aws",
                                             "redmetrics")
                self._csv_directory = Path(self.csv_path)
            except Exception as e:
                # Fail gracefully if neither is available
                raise FileNotFoundError(
                    f"Could not determine NAS path and no csv_directory provided. Error: {e}")

        if not self._csv_directory.exists():
            raise FileNotFoundError(f"The CSV directory does not exist: {self._csv_directory}")

        self._validate_input(input=[game_id, game_name, self._config.GAME_ID, self._config.GAME_NAME])
        self._validate_config()
        self._load_csv_files()

    def _download_and_cache(self, url: str, target_directory: str = "downloaded_data_cache") -> dict:
        """
        Downloads content from URL and synthesizes necessary CSV files (events, games, players, versions).
        Returns dictionary with extracted game_id, game_name, and the directory path.
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        os.makedirs(target_directory, exist_ok=True)

        # A. Download
        print(f"Downloading data...")
        response = requests.get(url)
        response.raise_for_status()
        csv_content = response.text

        # B. Save events.csv
        events_path = os.path.join(target_directory, "events.csv")
        with open(events_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        # C. Load for processing
        df_events = pd.read_csv(StringIO(csv_content))

        # D. Extract ID/Name
        game_id = query_params.get('game', [None])[0]
        if not game_id and 'game_id' in df_events.columns and not df_events.empty:
            game_id = str(df_events['game_id'].iloc[0])

        game_name = f"Game_{game_id}" if game_id else "Unknown_Game"

        # E. Synthesize games.csv
        df_games = pd.DataFrame([{'id': game_id, 'name': game_name}])
        df_games.to_csv(os.path.join(target_directory, "games.csv"), index=False)

        # F. Synthesize players.csv
        user_col = next((col for col in ['playerId', 'player', 'user_id', 'player_id', 'user'] if
                         col in df_events.columns), None)
        if user_col:
            unique_players = df_events[user_col].unique()
            df_players = pd.DataFrame({'id': unique_players})
            df_players['name'] = df_players['id'].apply(lambda x: f"Player_{x}")
            df_players.to_csv(os.path.join(target_directory, "players.csv"), index=False)
        else:
            pd.DataFrame(columns=['id', 'name']).to_csv(os.path.join(target_directory, "players.csv"),
                                                        index=False)

        # G. Synthesize game_versions.csv (Crucial for Legacy Support)
        version_col = next((col for col in ['version', 'gameVersion', 'game_version']
                            if col in df_events.columns), None)

        if version_col:
            unique_versions = df_events[version_col].unique()
            df_versions = pd.DataFrame({'id': unique_versions})
        else:
            df_versions = pd.DataFrame({'id': ['1.0']})  # Dummy version

        df_versions['game_id'] = game_id
        df_versions.to_csv(os.path.join(target_directory, "game_versions.csv"), index=False)

        print(f"Successfully cached data in '{target_directory}'")

        return {
            "game_name": game_name,
            "game_id": game_id,
            "csv_directory": target_directory
        }

    def _load_csv_files(self):
        """Load all CSV files and normalize columns (Hybrid Approach)"""
        try:
            # 1. Load ALL files (Required for Legacy Compatibility)
            self._events_df = pd.read_csv(self._csv_directory / "events.csv")
            self._players_df = pd.read_csv(self._csv_directory / "players.csv")
            self._games_df = pd.read_csv(self._csv_directory / "games.csv")
            self._game_versions_df = pd.read_csv(self._csv_directory / "game_versions.csv")

            # 2. Normalize Player ID (Required for URL Data)
            # Nas data has 'player_id', rm1 URL data has 'playerId'
            if 'player_id' not in self._events_df.columns:
                found_col = next((col for col in ['playerId', 'player', 'user_id', 'user', 'subject_id']
                                  if col in self._events_df.columns), None)

                if found_col:
                    if self._config:
                        print(f"Renaming column '{found_col}' to 'player_id'...")
                    self._events_df.rename(columns={found_col: 'player_id'}, inplace=True)

            # 3. Normalize Game Version ID (Required for URL Data)
            # Legacy data has 'gameVersion_id', URL data has 'gameVersion'
            if 'gameVersion_id' not in self._events_df.columns and 'gameVersion' in self._events_df.columns:
                self._events_df.rename(columns={'gameVersion': 'gameVersion_id'}, inplace=True)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing CSV file in {self._csv_directory}: {e}")

    def retrieve_data(self, *, verbose: bool = False, after: str = None, before: str = None,
                      event_type: str = None, section: str = None) -> pd.DataFrame:

        if verbose:
            print("Fetching all data from events.csv...")

        # process the loaded DF
        self._retrieved_df = self.fetch_all_data(verbose=verbose, after=after, before=before,
                                                 event_type=event_type, section=section)
        return self._format_df(verbose=verbose)

    def _validate_config(self) -> None:
        if not self._config.is_rm1:
            raise ValueError(CONFIG_URL_MISMATCH_ERROR)
        return None

    def get_game_version_ids(self):
        """Determine game version IDs based on provided inputs"""
        if self._game_version_ids:
            return self._game_version_ids

        elif self._game_id:
            matching_versions = self._game_versions_df[self._game_versions_df['game_id'] == self._game_id]
            if not matching_versions.empty:
                return matching_versions['id'].tolist()
            else:
                raise ValueError(f"No game_versions found for game_id: {self._game_id}")

        else:
            merged_df = self._game_versions_df.merge(
                self._games_df,
                left_on='game_id',
                right_on='id',
                suffixes=('', '_game')
            )
            matching_versions = merged_df[merged_df['name_game'] == self._game_name]
            if not matching_versions.empty:
                return matching_versions['id'].tolist()
            else:
                raise ValueError(f"No game_versions found for game name: {self._game_name}")

    def parse_json_column(self, *, df: pd.DataFrame, column_name: str, prefix: str):
        """Parse JSON columns in the DataFrame"""

        def try_parse(val):
            if pd.isna(val):
                return {}
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return {}

        parsed_df = df[column_name].apply(try_parse).apply(pd.Series)
        parsed_df.columns = [f"{prefix}.{col}" for col in parsed_df.columns]

        return pd.concat([df.drop(columns=[column_name]), parsed_df], axis=1)

    def convert_to_iso8601_millis(self, *, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Convert specified datetime columns to ISO 8601 format with milliseconds"""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce") \
                              .dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ') \
                              .str.slice(stop=-4) + 'Z'
        return df

    def _format_df(self, *, verbose: Optional[bool] = False) -> pd.DataFrame:
        """Format the retrieved DataFrame to match expected output format"""
        if not self._retrieved_df.empty:

            if verbose:
                print("Formatting dataframe...")

            self._retrieved_df.rename(columns={
                "gameVersion_id": "gameVersion",  # We assume this col exists in csv or is ignored
                "player_id": "playerId",
                "birthDate": "playerBirthdate",
                "region": "playerRegion",
                "country": "playerCountry",
                "gender": "playerGender",
                "externalId": "playerExternalId",
            }, inplace=True)

            if "eventCustomData" in self._retrieved_df.columns:
                self._retrieved_df = self.parse_json_column(df=self._retrieved_df,
                                                            column_name="eventCustomData",
                                                            prefix="customData")

            self._retrieved_df = self.convert_to_iso8601_millis(df=self._retrieved_df,
                                                                columns=["serverTime", "userTime"])

        # This reindex step is CRITICAL. It ensures that even if 'gameVersion'
        # is missing from the CSV, the column is created (with NaNs)
        # so the downstream Parser doesn't crash.
        self._extra_fields = set(self._retrieved_df.columns) - set(self._config.DOWNLOADER_FIELD_ORDER)
        all_fields = self._config.DOWNLOADER_FIELD_ORDER + tuple(self._extra_fields)
        self._retrieved_df = self._retrieved_df.reindex(columns=all_fields)

        return self._retrieved_df

    def _create_df(self, *, game_version_id: str, after: str = None, before: str = None,
                   event_type: str = None, section: str = None) -> pd.DataFrame:

        # 1. Standard Filtering
        filtered_df = self._events_df[self._events_df['gameVersion_id'] == game_version_id].copy()
        if after: filtered_df = filtered_df[filtered_df['serverTime'] >= after]
        if before: filtered_df = filtered_df[filtered_df['serverTime'] <= before]
        if event_type: filtered_df = filtered_df[filtered_df['type'] == event_type]
        if section: filtered_df = filtered_df[filtered_df['section'].str.contains(section, na=False)]

        result_df = filtered_df.merge(self._players_df, left_on='player_id', right_on='id',
                                      suffixes=('', '_player'))

        # 2. UNMASKING LOGIC (Recover Real IDs)
        # Determine column names dynamically to handle both old and new data structures
        custom_data_col = 'customData_player' if 'customData_player' in result_df.columns else 'playerCustomData'
        external_id_col = 'externalId_player' if 'externalId_player' in result_df.columns else 'playerExternalId'
        if 'playerExternalId' not in result_df.columns and 'externalId' in result_df.columns:
            external_id_col = 'externalId'

        # Helper function to extract ID
        def extract_real_id(row):
            current_id = str(row.get(external_id_col, ''))
            # Check for masking
            if current_id.lower() == 'xxxxxx' or current_id == '' or current_id == 'nan' or current_id == 'None':
                try:
                    custom_data = json.loads(row.get(custom_data_col, '{}'))
                    return custom_data.get('userProvidedId', current_id)
                except:
                    return current_id
            return current_id

        # Apply Unmasking
        if external_id_col in result_df.columns and custom_data_col in result_df.columns:
            if not result_df.empty and 'xxxxxx' in result_df[external_id_col].astype(str).str.lower().values:
                print("Attempting to recover masked IDs from playerCustomData...")
                # We overwrite the column with the REAL ID
                result_df[external_id_col] = result_df.apply(extract_real_id, axis=1)

        # Now that IDs are unmasked, we filter them out immediately using the config, if needed
        if self._config and self._config.MANUALLY_EXCLUDED_IDS:
            # Ensure we are looking at the correct column for filtering
            target_col = external_id_col

            # Count before dropping
            initial_count = len(result_df)

            # Filter: Keep rows where ID is NOT in the excluded list
            # We convert to string to ensure "190" matches 190
            result_df = result_df[~result_df[target_col].astype(str).isin(self._config.MANUALLY_EXCLUDED_IDS)]

            dropped_count = initial_count - len(result_df)
            if dropped_count > 0:
                print(f"Exclusion applied: Dropped {dropped_count} rows matching excluded IDs.")

        # 4. Standard Cleanup
        if 'id_player' in result_df.columns: result_df = result_df.drop(columns=['id_player'])
        if 'customData' in result_df.columns: result_df = result_df.rename(
            columns={'customData': 'eventCustomData'})
        if 'customData_player' in result_df.columns: result_df = result_df.rename(
            columns={'customData_player': 'playerCustomData'})

        # Normalize the ID column name to what _format_df expects ('playerExternalId')
        if external_id_col != 'playerExternalId' and external_id_col in result_df.columns:
            result_df.rename(columns={external_id_col: 'playerExternalId'}, inplace=True)

        return result_df

    def fetch_all_data(self, *, verbose: bool = False, after: str = None, before: str = None,
                       event_type: str = None, section: str = None) -> pd.DataFrame:
        """Fetch all data from local CSV files (Iterates through versions)"""
        all_dfs = []
        game_version_ids = self.get_game_version_ids()

        for game_version_id in game_version_ids:
            if verbose:
                print(f"Fetching game_version_id={game_version_id}...")

            df = self._create_df(
                game_version_id=game_version_id,
                after=after,
                before=before,
                event_type=event_type,
                section=section
            )

            if not df.empty:
                all_dfs.append(df)

        result_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        if verbose:
            print(f"Fetched {len(result_df)} rows total from {len(game_version_ids)} game version(s).")

        return result_df
