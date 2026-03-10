import json
import os
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from tqdm import tqdm

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

    import os
    import pandas as pd
    import requests
    from io import StringIO
    from urllib.parse import urlparse, parse_qs
    from datetime import datetime

    def _download_and_cache(self, url: str, target_directory: str = "downloaded_data_cache") -> dict:
        """
        Optimized paginated download. Replaces entry-by-entry duplicate checking
        with vectorized pandas processing for speed and row-count integrity.
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        os.makedirs(target_directory, exist_ok=True)
        events_path = os.path.join(target_directory, "events.csv")

        # 1. Setup URL & Game ID extraction
        json_url = url.replace("/event.csv", "/event.json")
        base_params = {"perPage": 500}

        # Extract the real Game ID from the URL parameters
        game_id_from_url = query_params.get('game', [None])[0]

        # 2. Sequential Page Download
        print("Downloading all pages from RedMetrics...")
        response = requests.get(json_url, params={"page": 1, **base_params})
        response.raise_for_status()
        page_count = int(response.headers.get('x-page-count', 1))

        all_events = []
        for page in tqdm(range(1, page_count + 1), desc="Fetching Pages"):
            r = requests.get(json_url, params={"page": page, **base_params})
            r.raise_for_status()
            all_events.extend(r.json())

        # 3. Vectorized Processing
        if not all_events:
            raise ValueError(f"No events found for game {game_id_from_url}. Check your 'after' date filter.")

        print(f"Downloaded {len(all_events)} events. Processing...")
        df_events = pd.json_normalize(all_events, sep='.')

        # 4. DYNAMIC COLUMN MAPPING (Fixes the 'player' KeyError)
        player_col = next((col for col in ['player', 'player.id', 'playerId'] if col in df_events.columns), None)
        if not player_col:
            raise KeyError(f"Could not find player ID column. Found columns: {df_events.columns.tolist()}")

        # 5. BULK PLAYER ENRICHMENT
        unique_player_ids = df_events[player_col].unique()
        player_cache = {}

        print(f"Enriching {len(unique_player_ids)} unique players...")
        for p_id in tqdm(unique_player_ids, desc="Players"):
            try:
                p_r = requests.get(f"https://api.creativeforagingtask.com/v1/player/{p_id}")
                p_r.raise_for_status()
                player_cache[p_id] = p_r.json()
            except:
                player_cache[p_id] = {'id': p_id}

                # 6. CONSTRUCT PLAYER DATAFRAME & MERGE
        player_df = pd.DataFrame.from_dict(player_cache, orient='index').reset_index()
        player_df.rename(columns={'index': player_col}, inplace=True)

        rename_map = {
            'birthDate': 'playerBirthdate', 'region': 'playerRegion',
            'country': 'playerCountry', 'gender': 'playerGender',
            'externalId': 'playerExternalId', 'customData': 'playerCustomData'
        }
        player_df.rename(columns={k: v for k, v in rename_map.items() if k in player_df.columns}, inplace=True)

        final_df = df_events.merge(player_df, on=player_col, how='left')

        # 7. FINAL CLEANUP (Fixes the 'id' KeyError)
        if player_col != 'playerId':
            final_df.rename(columns={player_col: 'playerId'}, inplace=True)

        id_col = next((col for col in ['id', 'id_x', '_id'] if col in final_df.columns), None)
        if id_col:
            final_df.drop_duplicates(subset=[id_col], keep='first', inplace=True)
            if id_col != 'id':
                final_df.rename(columns={id_col: 'id'}, inplace=True)

        if 'userTime' in final_df.columns:
            final_df.sort_values('userTime', inplace=True)

        # 8. SYNTESIZE META FILES (Prevents "No game_versions found" error)
        # This part ensures the tables get built so get_game_version_ids() works
        final_df.to_csv(events_path, index=False)

        # Use the ID we found in the URL
        actual_game_id = game_id_from_url or (
            str(final_df['game_id'].iloc[0]) if 'game_id' in final_df.columns else "Unknown")

        # Create the small supporting CSVs the pipeline expects
        pd.DataFrame([{'id': actual_game_id, 'name': f"Game_{actual_game_id}"}]).to_csv(
            os.path.join(target_directory, "games.csv"), index=False)

        # If the data has version info, use it; otherwise, create a dummy 1.0
        v_col = next((c for c in ['gameVersion', 'version'] if c in final_df.columns), None)
        v_ids = final_df[v_col].unique() if v_col else ['1.0']
        pd.DataFrame({'id': v_ids, 'game_id': actual_game_id}).to_csv(
            os.path.join(target_directory, "game_versions.csv"), index=False)

        print(f"Successfully cached {len(final_df)} events in '{target_directory}'")

        return {
            "game_name": f"Game_{actual_game_id}",
            "game_id": actual_game_id,
            "csv_directory": target_directory
        }
    def _download_and_cache_I_THOUGHT_THIS_WORKS_BUT_IT_MISSES_SOME_DATA_ROWS(self, url: str, target_directory: str = "downloaded_data_cache") -> dict:
        """
        Downloads data in 30-day chunks to prevent server timeouts and synthesizes
        necessary CSV files (events, games, players, versions).
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        os.makedirs(target_directory, exist_ok=True)
        events_path = os.path.join(target_directory, "events.csv")

        # 1. Determine Time Range for Chunking
        after_str = query_params.get('after', ["2021-01-01T00:00:00.000Z"])[0]
        current_after_dt = pd.to_datetime(after_str.replace('Z', ''))
        end_dt = datetime.now()

        base_url_no_params = url.split('?')[0]
        game_id_from_url = query_params.get('game', [None])[0]

        print(f"Starting chunked download from {after_str} to present...")
        chunk_dfs = []

        # 2. The Chunking Loop (Handles timeouts)
        while current_after_dt < end_dt:
            next_before_dt = current_after_dt + pd.Timedelta(days=30)
            if next_before_dt > end_dt:
                next_before_dt = end_dt

            after_val = current_after_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            before_val = next_before_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

            chunk_url = f"{base_url_no_params}?game={game_id_from_url}&entityType=event&after={after_val}&before={before_val}"

            try:
                # Using a long read timeout for heavy CSV generation on the server
                response = requests.get(chunk_url, timeout=(15, 300))
                response.raise_for_status()

                chunk_df = pd.read_csv(StringIO(response.text), on_bad_lines='skip')
                if not chunk_df.empty:
                    print(f"  [OK]    {after_val[:10]} to {before_val[:10]} ({len(chunk_df)} rows)")
                    chunk_dfs.append(chunk_df)
                else:
                    print(f"  [EMPTY] {after_val[:10]} to {before_val[:10]}")
            except Exception as e:
                print(f"  [FAILED] Chunk {after_val[:10]} failed: {e}")

            current_after_dt = next_before_dt

        # 3. Combine and Save events.csv
        if not chunk_dfs:
            raise ValueError("Download failed: No data retrieved from any chunk.")

        df_events = pd.concat(chunk_dfs, ignore_index=True).drop_duplicates()
        df_events.to_csv(events_path, index=False)

        # 4. Extract Game ID and Name (Mirroring your original logic)
        game_id = game_id_from_url
        if not game_id and 'game_id' in df_events.columns and not df_events.empty:
            game_id = str(df_events['game_id'].iloc[0])

        game_name = f"Game_{game_id}" if game_id else "Unknown_Game"

        # 5. Synthesize games.csv
        pd.DataFrame([{'id': game_id, 'name': game_name}]).to_csv(
            os.path.join(target_directory, "games.csv"), index=False
        )

        # 6. Synthesize players.csv (Crucial to fix your FileNotFoundError)
        user_col = next((col for col in ['playerId', 'player', 'user_id', 'player_id', 'user']
                         if col in df_events.columns), None)
        if user_col:
            unique_players = df_events[user_col].unique()
            df_players = pd.DataFrame({'id': unique_players})
            df_players['name'] = df_players['id'].apply(lambda x: f"Player_{x}")
            df_players.to_csv(os.path.join(target_directory, "players.csv"), index=False)
        else:
            # Create empty but existing file if no user column found
            pd.DataFrame(columns=['id', 'name']).to_csv(
                os.path.join(target_directory, "players.csv"), index=False
            )

        # 7. Synthesize game_versions.csv (For Pipeline Compatibility)
        version_col = next((col for col in ['version', 'gameVersion', 'game_version']
                            if col in df_events.columns), None)

        if version_col:
            unique_versions = df_events[version_col].unique()
            df_versions = pd.DataFrame({'id': unique_versions})
        else:
            df_versions = pd.DataFrame({'id': ['1.0']})  # Default version

        df_versions['game_id'] = game_id
        df_versions.to_csv(os.path.join(target_directory, "game_versions.csv"), index=False)

        print(f"Successfully cached {len(df_events)} events in '{target_directory}'")

        return {
            "game_name": game_name,
            "game_id": game_id,
            "csv_directory": target_directory
        }


    def _download_and_cache_BEFORE_ROEYS_CHANEGS_TO_HANDLE_RM1_TIMEOUT(self, url: str, target_directory: str = "downloaded_data_cache") -> dict:
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

        # Merge with player metadata
        result_df = filtered_df.merge(self._players_df, left_on='player_id', right_on='id',
                                      suffixes=('', '_player'))

        # 2. STRICT ID EXTRACTION LOGIC
        # Identify which columns exist (names vary between RM1 and RM2 exports)
        custom_data_col = 'customData_player' if 'customData_player' in result_df.columns else 'playerCustomData'
        external_id_col = 'externalId_player' if 'externalId_player' in result_df.columns else 'playerExternalId'
        if 'externalId' in result_df.columns and external_id_col not in result_df.columns:
            external_id_col = 'externalId'

        def extract_real_id(row):
            """Prioritizes manual input from JSON, ignores system noise."""
            # A. Priority 1: The manual 'userProvidedId' inside the JSON blob
            try:
                cdata_raw = row.get(custom_data_col, '{}')
                cdata = json.loads(cdata_raw) if isinstance(cdata_raw, str) else cdata_raw
                up_id = cdata.get('userProvidedId')
                if up_id and str(up_id).strip() and str(up_id).lower() not in ['nan', 'none', 'null']:
                    return str(up_id).strip()
            except:
                pass

            # B. Priority 2: Use the external ID column ONLY if it's not masked or empty
            ext_val = str(row.get(external_id_col, '')).strip()
            if ext_val and ext_val.lower() not in ['xxxxxx', 'nan', 'none', 'null', '']:
                return ext_val

            # C. Fallback: If no ID is found, we use a placeholder to flag it for exclusion
            return "UNIDENTIFIED_SESSION"

        # Apply to EVERY row to ensure UserProvidedID is captured globally
        if not result_df.empty:
            result_df['playerExternalId'] = result_df.apply(extract_real_id, axis=1)

        # 3. Filtering using your String-based Exclusion List
        if self._config and self._config.MANUALLY_EXCLUDED_IDS:
            initial_count = len(result_df)
            # Filter based on the 'playerExternalId' we just synthesized/recovered
            result_df = result_df[~result_df['playerExternalId'].astype(str).isin(self._config.MANUALLY_EXCLUDED_IDS)]

            dropped = initial_count - len(result_df)
            if dropped > 0:
                print(f"ID Policy applied: Dropped {dropped} rows (Excluded or Missing UserProvidedID).")

        # 4. Standard Cleanup
        if 'id_player' in result_df.columns: result_df.drop(columns=['id_player'], inplace=True)
        if 'customData' in result_df.columns: result_df.rename(columns={'customData': 'eventCustomData'}, inplace=True)
        if 'customData_player' in result_df.columns: result_df.rename(columns={'customData_player': 'playerCustomData'},
                                                                      inplace=True)

        return result_df

    def _create_df_BAD_vERSION_WHICH_OVERRIDES_THE_USERPROVIDEDID(self, *, game_version_id: str, after: str = None, before: str = None,
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
