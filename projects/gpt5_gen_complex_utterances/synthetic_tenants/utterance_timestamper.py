#!/usr/bin/env python3
"""
Utterance Timestamper Module

This module adds timestamps to utterances for testing Seval.
1. For file-based utterances, it reads the files in a folder and finds the latest
file modification time and generates timestamps for utterances with Segment=file, setting them to
a few hours after the latest file modification time.
2. For email-based utterances, it reads LLM-generated timestamps from files in a folder and
assigns them to the corresponding utterances. The input files of LLM-generated timestamps
may contain duplicated utterances.

"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import fire
import random
import os
import csv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FileUtteranceTimestamper:
    """Handles timestamping of file-related utterances based on file access analysis results."""

    def __init__(self):
        self.latest_file_time = None
        self.timestamp_start = None
        self.unused_timestamp_utterances = []

    def _detect_file_format(self, file_path: str) -> str:
        """Detect if file is JSON or TSV based on extension.

        Args:
            file_path: Path to the file

        Returns:
            'json' or 'tsv'
        """
        path = Path(file_path)
        if path.suffix.lower() in [".tsv", ".csv"]:
            return "tsv"
        else:
            return "json"

    def _load_file_access_results(
        self, file_access_results_path: str
    ) -> Dict[str, Any]:
        """Load the file access analysis results.

        Args:
            file_access_results_path: Path to the file access results JSON

        Returns:
            Dict containing the file access analysis results

        Raises:
            FileNotFoundError: If the file access results file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
        """
        try:
            file_path = Path(file_access_results_path)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"File access results not found: {file_access_results_path}"
                )

            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file access results: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading file access results: {e}")
            raise

    def _scan_files_folder(self, files_folder: str = "data/files") -> str:
        """Scan the files folder and find the latest modification time.

        Args:
            files_folder: Path to the folder containing files to scan

        Returns:
            Latest modification time in ISO format

        Raises:
            FileNotFoundError: If the files folder doesn't exist
            ValueError: If no files are found in the folder
        """
        try:
            folder_path = Path(files_folder)
            if not folder_path.exists():
                raise FileNotFoundError(f"Files folder not found: {files_folder}")

            if not folder_path.is_dir():
                raise ValueError(f"Path is not a directory: {files_folder}")

            # Find all files recursively
            all_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = Path(root) / file
                    all_files.append(file_path)

            if not all_files:
                raise ValueError(f"No files found in folder: {files_folder}")

            logger.info(f"📁 Found {len(all_files)} files in {files_folder}")

            # Find the latest modification time
            latest_mod_time = None
            latest_file = None

            for file_path in all_files:
                try:
                    mod_time = file_path.stat().st_mtime
                    mod_datetime = datetime.fromtimestamp(mod_time)

                    if latest_mod_time is None or mod_datetime > latest_mod_time:
                        latest_mod_time = mod_datetime
                        latest_file = file_path

                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not access file {file_path}: {e}")
                    continue

            if latest_mod_time is None:
                raise ValueError("Could not determine modification time for any files")

            self.latest_file_time = latest_mod_time.isoformat()
            logger.info(f"📅 Latest file modification time: {self.latest_file_time}")
            logger.info(f"📄 Latest modified file: {latest_file}")

            return self.latest_file_time

        except Exception as e:
            logger.error(f"Error scanning files folder: {e}")
            raise

    def _extract_latest_file_time(self, file_access_results: Dict[str, Any]) -> str:
        """Extract the latest file modification time from the analysis results.

        Args:
            file_access_results: The loaded file access analysis results

        Returns:
            Latest modification time in ISO format

        Raises:
            KeyError: If the required timestamp fields are missing
        """
        try:
            summary_stats = file_access_results["summary_statistics"]
            latest_modification = summary_stats["latest_modification_time"]
            latest_creation = summary_stats["latest_creation_time"]

            # Use the later of modification or creation time
            if latest_modification and latest_creation:
                self.latest_file_time = max(latest_modification, latest_creation)
            elif latest_modification:
                self.latest_file_time = latest_modification
            elif latest_creation:
                self.latest_file_time = latest_creation
            else:
                raise KeyError("No timestamp information found in file access results")

            logger.info(f"📅 Latest file time extracted: {self.latest_file_time}")
            return self.latest_file_time

        except KeyError as e:
            logger.error(
                f"Missing required timestamp fields in file access results: {e}"
            )
            raise

    def _calculate_timestamp_start(self, hours_after: float = 2.0) -> datetime:
        """Calculate the starting timestamp for file utterances.

        Args:
            hours_after: Number of hours after the latest file time to start

        Returns:
            Starting datetime for file utterance timestamps
        """
        if not self.latest_file_time:
            raise ValueError(
                "Latest file time not set. Call _extract_latest_file_time first."
            )

        # Parse the ISO timestamp
        file_datetime = datetime.fromisoformat(self.latest_file_time)

        # Add the specified hours
        self.timestamp_start = file_datetime + timedelta(hours=hours_after)

        logger.info(
            f"⏰ File utterance timestamps will start at: {self.timestamp_start.isoformat()}"
        )
        return self.timestamp_start

    def _load_utterances(self, utterances_path: str) -> List[Dict[str, Any]]:
        """Load the utterances from JSON or TSV file.

        Args:
            utterances_path: Path to the utterances file (JSON or TSV)

        Returns:
            List of utterance dictionaries

        Raises:
            FileNotFoundError: If the utterances file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
        """
        file_format = self._detect_file_format(utterances_path)

        if file_format == "tsv":
            return self._load_utterances_from_tsv(utterances_path)
        else:
            return self._load_utterances_from_json(utterances_path)

    def _load_utterances_from_json(self, utterances_path: str) -> List[Dict[str, Any]]:
        """Load the utterances from JSON file.

        Args:
            utterances_path: Path to the utterances JSON file

        Returns:
            List of utterance dictionaries

        Raises:
            FileNotFoundError: If the utterances file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
        """
        try:
            file_path = Path(utterances_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Utterances file not found: {utterances_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both list format and dict with 'utterances' key
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "utterances" in data:
                return data["utterances"]
            else:
                logger.warning(
                    "Unexpected utterances file format, treating as empty list"
                )
                return []

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in utterances file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading utterances: {e}")
            raise

    def _load_utterances_from_tsv(self, utterances_path: str) -> List[Dict[str, Any]]:
        """Load utterances from TSV file.

        Args:
            utterances_path: Path to the utterances TSV file

        Returns:
            List of utterance dictionaries

        Raises:
            FileNotFoundError: If the utterances file doesn't exist
        """
        try:
            file_path = Path(utterances_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Utterances file not found: {utterances_path}")

            utterances = []
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    # Convert row to dict and clean up any None values
                    utterance = {
                        k: v for k, v in row.items() if v is not None and v != ""
                    }
                    utterances.append(utterance)

            logger.info(f"✅ Loaded {len(utterances)} utterances from TSV file")
            return utterances

        except Exception as e:
            logger.error(f"Error loading utterances from TSV: {e}")
            raise

    def _generate_file_utterance_timestamp(
        self, index: int, total_file_utterances: int, time_window_hours: float = 8.0
    ) -> str:
        """Generate a timestamp for a file utterance.

        Args:
            index: Index of the current file utterance (0-based)
            total_file_utterances: Total number of file utterances
            time_window_hours: Time window in hours to spread the utterances

        Returns:
            ISO formatted timestamp string
        """
        if not self.timestamp_start:
            raise ValueError(
                "Timestamp start not set. Call _calculate_timestamp_start first."
            )

        # Spread utterances across the time window with some randomization
        if total_file_utterances > 1:
            # Calculate base offset
            base_offset_hours = (
                index / (total_file_utterances - 1)
            ) * time_window_hours

            # Add some randomization (±30 minutes)
            random_offset_minutes = random.uniform(-30, 30)
            total_offset = timedelta(
                hours=base_offset_hours, minutes=random_offset_minutes
            )
        else:
            # Single utterance gets a small random offset
            random_offset_minutes = random.uniform(10, 60)
            total_offset = timedelta(minutes=random_offset_minutes)

        utterance_time = self.timestamp_start + total_offset
        return utterance_time.isoformat()

    def _process_utterances(
        self, utterances: List[Dict[str, Any]], time_window_hours: float = 8.0
    ) -> List[Dict[str, Any]]:
        """Process utterances and add timestamps to file-related ones.

        Args:
            utterances: List of utterance dictionaries
            time_window_hours: Time window in hours to spread file utterances

        Returns:
            List of processed utterances with timestamps added
        """
        # Find all file-related utterances (check both 'segment' and 'Segment')
        file_utterances = []
        for i, utterance in enumerate(utterances):
            segment = utterance.get("segment") or utterance.get("Segment", "")
            if segment and segment.lower() == "file":
                file_utterances.append((i, utterance))

        logger.info(
            f"📁 Found {len(file_utterances)} file-related utterances to timestamp"
        )

        if not file_utterances:
            logger.warning("No file-related utterances found (segment='file')")
            return utterances

        # Process file utterances
        processed_utterances = utterances.copy()

        for file_index, (utterance_index, utterance) in enumerate(file_utterances):
            timestamp = self._generate_file_utterance_timestamp(
                file_index, len(file_utterances), time_window_hours
            )

            # Add or update the timestamp
            processed_utterances[utterance_index]["timestamp"] = timestamp

            logger.debug(f"📝 Utterance {utterance_index}: {timestamp}")

        return processed_utterances

    def _save_utterances(
        self, utterances: List[Dict[str, Any]], output_path: str
    ) -> None:
        """Save utterances to JSON or TSV file based on the output path extension.

        Args:
            utterances: List of utterance dictionaries to save
            output_path: Path where to save the utterances
        """
        file_format = self._detect_file_format(output_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if file_format == "tsv":
            self._save_utterances_to_tsv(utterances, output_path)
        else:
            self._save_utterances_to_json(utterances, output_path)

    def _save_utterances_to_json(
        self, utterances: List[Dict[str, Any]], output_path: str
    ) -> None:
        """Save utterances to JSON file.

        Args:
            utterances: List of utterance dictionaries to save
            output_path: Path where to save the JSON file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(utterances, f, indent=2, ensure_ascii=False)

    def _save_utterances_to_tsv(
        self, utterances: List[Dict[str, Any]], output_path: str
    ) -> None:
        """Save utterances to TSV file.

        Args:
            utterances: List of utterance dictionaries to save
            output_path: Path where to save the TSV file
        """
        if not utterances:
            logger.warning("No utterances to save")
            return

        # Get all field names that exist in the utterances
        all_fields = set()
        for utterance in utterances:
            all_fields.update(utterance.keys())

        # Use a consistent field order: start with the original order from first utterance,
        # then add any new fields (like timestamp) at the end
        if utterances:
            first_utterance_fields = list(utterances[0].keys())
            # Add any additional fields that might have been added (like timestamp)
            additional_fields = [
                field for field in all_fields if field not in first_utterance_fields
            ]
            fieldnames = first_utterance_fields + additional_fields
        else:
            fieldnames = list(all_fields)

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(utterances)

    def _output_untimestamped_utterances_to_console(
        self, untimestamped_utterances: List[Dict[str, Any]]
    ) -> None:
        """Output untimestamped utterances to console for user review.

        Args:
            untimestamped_utterances: List of utterances that couldn't be timestamped
        """
        if not untimestamped_utterances:
            return

        print("\n" + "=" * 80)
        print(f"📋 UNTIMESTAMPED UTTERANCES ({len(untimestamped_utterances)} total)")
        print("=" * 80)

        # Group by user_id for better organization
        utterances_by_user = {}
        for utterance in untimestamped_utterances:
            user_id = utterance.get("user_id", "Unknown")
            if user_id not in utterances_by_user:
                utterances_by_user[user_id] = []
            utterances_by_user[user_id].append(utterance)

        for user_id, utterances in utterances_by_user.items():
            print(f"\n👤 User: {user_id} ({len(utterances)} utterances)")
            print("-" * 60)

            for i, utterance in enumerate(utterances[:5], 1):  # Show first 5 per user
                utterance_text = utterance.get("Utterance", "No utterance text")
                # Truncate long utterances for console display
                if len(utterance_text) > 100:
                    utterance_text = utterance_text[:97] + "..."
                print(f"  {i}. {utterance_text}")

            if len(utterances) > 5:
                print(f"  ... and {len(utterances) - 5} more utterances")

        print("\n" + "=" * 80)
        print("💡 These utterances need timestamps generated by LLM processing.")
        print("=" * 80 + "\n")

    def timestamp_file_utterances(
        self,
        utterances_path: str,
        output_path: str,
        files_folder: str = "data/files",
        hours_after_files: float = 2.0,
        time_window_hours: float = 8.0,
    ) -> None:
        """Main function to timestamp file-related utterances.

        Args:
            utterances_path: Path to the input utterances JSON file
            output_path: Path where to save the timestamped utterances
            files_folder: Path to the folder containing files to scan (default: "data/files")
            hours_after_files: Hours after latest file time to start utterance timestamps
            time_window_hours: Time window in hours to spread the file utterances
        """
        try:
            logger.info("🚀 Starting file utterance timestamping...")
            logger.info(f"� Files folder: {files_folder}")
            logger.info(f"📝 Input utterances: {utterances_path}")
            logger.info(f"💾 Output file: {output_path}")

            # Scan files folder and find latest modification time
            logger.info("📖 Scanning files folder for latest modification time...")
            self._scan_files_folder(files_folder)

            # Calculate timestamp starting point
            logger.info(
                f"⏱️  Calculating timestamps starting {hours_after_files} hours after latest file time..."
            )
            self._calculate_timestamp_start(hours_after_files)

            # Load utterances
            logger.info("📚 Loading utterances...")
            utterances = self._load_utterances(utterances_path)
            logger.info(f"✅ Loaded {len(utterances)} utterances")

            # Process utterances
            logger.info(
                f"🔄 Processing utterances with {time_window_hours}-hour time window..."
            )
            processed_utterances = self._process_utterances(
                utterances, time_window_hours
            )

            # Save results
            logger.info("💾 Saving timestamped utterances...")
            self._save_utterances(processed_utterances, output_path)

            logger.info(f"🎉 File utterance timestamping complete!")
            logger.info(f"📁 Output saved to: {output_path}")

            # Summary (use same logic as processing for consistency)
            file_utterance_count = 0
            timestamped_count = 0

            for utterance in processed_utterances:
                segment = utterance.get("segment") or utterance.get("Segment", "")
                if segment and segment.lower() == "file":
                    file_utterance_count += 1
                    if utterance.get("timestamp"):
                        timestamped_count += 1

            logger.info(f"📊 Summary:")
            logger.info(f"   📝 Total utterances: {len(processed_utterances)}")
            logger.info(f"   📁 File utterances: {file_utterance_count}")
            logger.info(f"   ⏰ Timestamped file utterances: {timestamped_count}")
            logger.info(
                f"   🕐 Timestamp range: {hours_after_files}h to {hours_after_files + time_window_hours}h after {self.latest_file_time}"
            )

        except Exception as e:
            logger.error(f"❌ Error during file utterance timestamping: {e}")
            raise

    def timestamp_email_utterances(
        self,
        utterances_path: str,
        output_path: str,
        timestamp_results_folder: str = "results/timestamp_results",
        untimestamped_output_path: Optional[str] = None,
    ) -> None:
        """Main function to timestamp email-related utterances from timestamp results files.

        Args:
            utterances_path: Path to the input utterances TSV file
            output_path: Path where to save the timestamped utterances
            timestamp_results_folder: Path to folder containing timestamp results files
            untimestamped_output_path: Optional path to save utterances without timestamps
        """
        try:
            logger.info("🚀 Starting email utterance timestamping...")
            logger.info(f"📁 Timestamp results folder: {timestamp_results_folder}")
            logger.info(f"📝 Input utterances: {utterances_path}")
            logger.info(f"💾 Output file: {output_path}")
            if untimestamped_output_path:
                logger.info(
                    f"📋 Untimestamped output file: {untimestamped_output_path}"
                )

            # Load utterances
            logger.info("📚 Loading utterances...")
            utterances = self._load_utterances(utterances_path)
            logger.info(f"✅ Loaded {len(utterances)} utterances")

            # Load timestamp results from all files
            logger.info("📊 Loading timestamp results...")
            timestamp_data = self._load_timestamp_results(timestamp_results_folder)
            logger.info(f"✅ Loaded timestamp data for {len(timestamp_data)} accounts")

            # Process utterances
            logger.info("🔄 Processing email utterances with timestamps...")
            processed_utterances = self._process_email_utterances(
                utterances, timestamp_data
            )

            # Save results
            logger.info("💾 Saving timestamped utterances...")
            self._save_utterances(processed_utterances, output_path)

            # Extract and save untimestamped email utterances if requested
            if untimestamped_output_path:
                untimestamped_utterances = self._extract_untimestamped_email_utterances(
                    processed_utterances
                )
                if untimestamped_utterances:
                    logger.info(
                        f"📋 Saving {len(untimestamped_utterances)} untimestamped email utterances..."
                    )
                    self._save_utterances(
                        untimestamped_utterances, untimestamped_output_path
                    )
                    logger.info(
                        f"📁 Untimestamped utterances saved to: {untimestamped_output_path}"
                    )

                    # Output untimestamped utterances to console
                    self._output_untimestamped_utterances_to_console(
                        untimestamped_utterances
                    )
                else:
                    logger.info("✅ No untimestamped email utterances to save!")

            logger.info(f"🎉 Email utterance timestamping complete!")
            logger.info(f"📁 Output saved to: {output_path}")

            # Summary
            email_utterance_count = 0
            timestamped_count = 0

            for utterance in processed_utterances:
                segment = utterance.get("segment") or utterance.get("Segment", "")
                if segment and segment.lower() == "email":
                    email_utterance_count += 1
                    if utterance.get("timestamp"):
                        timestamped_count += 1

            logger.info(f"📊 Summary:")
            logger.info(f"   📝 Total utterances: {len(processed_utterances)}")
            logger.info(f"   📧 Email utterances: {email_utterance_count}")
            logger.info(f"   ⏰ Timestamped email utterances: {timestamped_count}")

        except Exception as e:
            logger.error(f"❌ Error during email utterance timestamping: {e}")
            raise

    def _load_timestamp_results(
        self, timestamp_results_folder: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load timestamp results from all TSV files in the specified folder.
        Also analyzes and reports duplicates for each file.

        Args:
            timestamp_results_folder: Path to folder containing timestamp results

        Returns:
            Dictionary mapping email accounts to their timestamp data
        """
        timestamp_data = {}
        results_path = Path(timestamp_results_folder)

        if not results_path.exists():
            raise FileNotFoundError(
                f"Timestamp results folder not found: {timestamp_results_folder}"
            )

        tsv_files = list(results_path.glob("*.tsv"))
        logger.info(f"📂 Found {len(tsv_files)} timestamp result files")

        files_with_duplicates = 0
        files_without_duplicates = 0

        logger.info("🔍 Analyzing duplicates in each file...")

        for tsv_file in sorted(tsv_files):
            try:
                # Extract account name from filename (remove _results.tsv suffix)
                account_name = tsv_file.stem.replace("_results", "")

                # Load the TSV file
                with open(tsv_file, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    file_data = list(reader)

                # Analyze duplicates in this file
                file_analysis = self._analyze_file_duplicates(tsv_file.name, file_data)

                if file_analysis["has_duplicates"]:
                    files_with_duplicates += 1
                    logger.info(
                        f"📄 {tsv_file.name}: {len(file_data)} records ({file_analysis['total_duplicates']} duplicates)"
                    )
                    for count, num_utterances in sorted(
                        file_analysis["duplicate_counts"].items()
                    ):
                        logger.info(
                            f"   🔄 Duplicated {count} times: {num_utterances} utterances"
                        )
                    logger.info(
                        f"   📊 Same timestamp: {file_analysis['same_timestamp_duplicates']}"
                    )
                    logger.info(
                        f"   ⚠️ Different timestamp: {file_analysis['different_timestamp_duplicates']}"
                    )
                else:
                    files_without_duplicates += 1
                    logger.info(
                        f"✅ {tsv_file.name}: {len(file_data)} records (no duplicates)"
                    )

                timestamp_data[account_name] = file_data

            except Exception as e:
                logger.warning(f"⚠️ Failed to load {tsv_file.name}: {e}")
                continue

        logger.info(f"📊 File duplicate summary:")
        logger.info(f"   📁 Files with duplicates: {files_with_duplicates}")
        logger.info(f"   ✅ Files without duplicates: {files_without_duplicates}")

        return timestamp_data

    def _analyze_file_duplicates(
        self, filename: str, file_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze duplicates in a single file's data.

        Args:
            filename: Name of the file being analyzed
            file_data: List of records from the file

        Returns:
            Dictionary with duplicate analysis results
        """
        from collections import defaultdict, Counter

        # Track utterances and their timestamps
        utterance_data = defaultdict(list)

        for row in file_data:
            utterance_text = row.get("Utterance", "").strip()
            timestamp = row.get("Timestamp", "").strip()

            if utterance_text:
                utterance_data[utterance_text].append(timestamp)

        # Analyze duplicates
        total_utterances = len(file_data)
        unique_utterances = len(utterance_data)
        duplicate_counts = Counter()
        same_timestamp_duplicates = 0
        different_timestamp_duplicates = 0

        for utterance_text, timestamps in utterance_data.items():
            count = len(timestamps)
            if count > 1:
                duplicate_counts[count] += 1

                # Check if timestamps are the same or different
                unique_timestamps = set(timestamps)
                if len(unique_timestamps) == 1:
                    same_timestamp_duplicates += 1
                else:
                    different_timestamp_duplicates += 1

        return {
            "file_name": filename,
            "total_utterances": total_utterances,
            "unique_utterances": unique_utterances,
            "total_duplicates": total_utterances - unique_utterances,
            "duplicate_counts": dict(duplicate_counts),
            "same_timestamp_duplicates": same_timestamp_duplicates,
            "different_timestamp_duplicates": different_timestamp_duplicates,
            "has_duplicates": len(duplicate_counts) > 0,
        }

    def _process_email_utterances(
        self,
        utterances: List[Dict[str, Any]],
        timestamp_data: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Process utterances and add timestamps to email-related ones.

        Args:
            utterances: List of utterance dictionaries
            timestamp_data: Dictionary mapping accounts to timestamp data

        Returns:
            List of processed utterances with timestamps added to email utterances
        """
        processed_utterances = []

        # Find email-related utterances
        email_utterances = []
        for i, utterance in enumerate(utterances):
            segment = utterance.get("segment") or utterance.get("Segment", "")
            if segment and segment.lower() == "email":
                email_utterances.append((i, utterance))

        logger.info(
            f"📧 Found {len(email_utterances)} email-related utterances to timestamp"
        )

        if not email_utterances:
            logger.warning("No email-related utterances found (segment='email')")
            return utterances

        # Create lookup for timestamp data
        timestamp_lookup = self._create_timestamp_lookup(timestamp_data)

        # Process each utterance
        for i, utterance in enumerate(utterances):
            # Create a copy to avoid modifying the original
            processed_utterance = utterance.copy()

            # Check if this is an email utterance
            segment = utterance.get("segment") or utterance.get("Segment", "")
            if segment and segment.lower() == "email":
                # Try to find timestamp for this utterance
                timestamp = self._find_utterance_timestamp(
                    processed_utterance, timestamp_lookup
                )
                if timestamp:
                    processed_utterance["timestamp"] = timestamp
                    logger.debug(
                        f"📅 Added timestamp {timestamp} to email utterance {i+1}"
                    )
                else:
                    logger.debug(f"⚠️ No timestamp found for email utterance {i+1}")

            processed_utterances.append(processed_utterance)

        return processed_utterances

    def _create_timestamp_lookup(
        self, timestamp_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Create a lookup dictionary for finding timestamps by utterance text and Email_Account.
        Skips duplicate utterances entirely, keeping only the first occurrence.

        Args:
            timestamp_data: Dictionary mapping account file names to timestamp data

        Returns:
            Dictionary for quick timestamp lookup keyed by Email_Account::utterance_text
        """
        lookup = {}
        duplicate_stats = {
            "same_timestamp": 0,
            "different_timestamp": 0,
            "total_duplicates_skipped": 0,
        }

        # Store unused utterances for reporting
        self.unused_timestamp_utterances = []

        for account_file, records in timestamp_data.items():
            for record in records:
                utterance_text = record.get("Utterance", "").strip()
                timestamp = record.get("Timestamp", "")
                email_account = record.get("Email_Account", "").strip()

                if not utterance_text or not email_account:
                    continue

                # Create lookup key using Email_Account field, not file name
                key = f"{email_account}::{utterance_text}"

                # Check for duplicates - skip if already exists
                if key in lookup:
                    duplicate_stats["total_duplicates_skipped"] += 1
                    existing_timestamp = lookup[key].get("timestamp", "")

                    if existing_timestamp != timestamp:
                        duplicate_stats["different_timestamp"] += 1
                        logger.debug(
                            f"🔄 Skipping duplicate with different timestamp for {email_account}: {utterance_text[:50]}..."
                        )
                    else:
                        duplicate_stats["same_timestamp"] += 1
                        logger.debug(
                            f"🔄 Skipping duplicate with identical timestamp for {email_account}: {utterance_text[:50]}..."
                        )

                    # Skip this duplicate - don't add to lookup
                    continue

                # Store the record (first occurrence only)
                lookup[key] = {
                    "timestamp": timestamp,
                    "email_account": email_account,
                    "utterance": utterance_text,
                    "record": record,
                    "used": False,  # Track if this timestamp gets used
                }

                # Also store for unused tracking
                self.unused_timestamp_utterances.append(
                    {
                        "key": key,
                        "email_account": email_account,
                        "utterance": utterance_text,
                        "timestamp": timestamp,
                        "used": False,
                    }
                )

        logger.info(
            f"📊 Created timestamp lookup with {len(lookup)} unique utterance-account combinations"
        )
        logger.info(
            f"🔄 Skipped {duplicate_stats['total_duplicates_skipped']} duplicate utterances:"
        )
        logger.info(f"   📊 Same timestamp: {duplicate_stats['same_timestamp']}")
        logger.info(
            f"   ⚠️ Different timestamp: {duplicate_stats['different_timestamp']}"
        )
        return lookup

    def _find_utterance_timestamp(
        self, utterance: Dict[str, Any], timestamp_lookup: Dict[str, Any]
    ) -> Optional[str]:
        """Find timestamp for a given utterance using exact Email_Account matching.

        Args:
            utterance: Utterance dictionary
            timestamp_lookup: Lookup dictionary for timestamps

        Returns:
            Timestamp string if found, None otherwise
        """
        utterance_text = utterance.get("Utterance", "").strip()
        user_id = utterance.get("user_id", "").strip()

        if not utterance_text or not user_id:
            return None

        # Use exact user_id matching against Email_Account field
        # No account name extraction - match the full email address exactly
        key = f"{user_id}::{utterance_text}"

        if key in timestamp_lookup:
            logger.debug(f"✅ Found exact timestamp match for {user_id}")
            # Mark this timestamp as used
            timestamp_lookup[key]["used"] = True
            # Also mark in unused tracking list
            for unused_item in self.unused_timestamp_utterances:
                if unused_item["key"] == key:
                    unused_item["used"] = True
                    break
            return timestamp_lookup[key]["timestamp"]

        # Log missing timestamp for debugging
        logger.debug(
            f"❌ No timestamp found for utterance from {user_id}: {utterance_text[:100]}..."
        )
        return None

    def _report_unmatched_utterances(
        self,
        processed_utterances: List[Dict[str, Any]],
        timestamp_data: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Report on unmatched utterances - both unused timestamps and untimestamped email utterances.

        Args:
            processed_utterances: The processed utterances list
            timestamp_data: Original timestamp data
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔍 UNMATCHED UTTERANCES ANALYSIS")
        logger.info("=" * 80)

        # 1. Report unused timestamp utterances (should be 20 items: 934-914)
        unused_timestamps = [
            item for item in self.unused_timestamp_utterances if not item["used"]
        ]
        logger.info(
            f"\n1️⃣ UNUSED TIMESTAMP UTTERANCES ({len(unused_timestamps)} items):"
        )
        logger.info("-" * 60)

        if unused_timestamps:
            # Group by account for cleaner reporting
            by_account = {}
            for item in unused_timestamps:
                account = item["email_account"]
                if account not in by_account:
                    by_account[account] = []
                by_account[account].append(item)

            for account, items in sorted(by_account.items()):
                logger.info(f"📧 {account} ({len(items)} unused):")
                for item in items[:3]:  # Show first 3
                    logger.info(f"   • {item['utterance'][:100]}...")
                if len(items) > 3:
                    logger.info(f"   ... and {len(items) - 3} more")
                logger.info("")
        else:
            logger.info("✅ All timestamp utterances were used!")

        # 2. Report email utterances without timestamps
        untimestamped_email_utterances = []
        for utterance in processed_utterances:
            segment = utterance.get("segment") or utterance.get("Segment", "")
            if (
                segment
                and segment.lower() == "email"
                and not utterance.get("timestamp")
            ):
                untimestamped_email_utterances.append(utterance)

        logger.info(
            f"\n2️⃣ EMAIL UTTERANCES WITHOUT TIMESTAMPS ({len(untimestamped_email_utterances)} items):"
        )
        logger.info("-" * 60)

        if untimestamped_email_utterances:
            # Group by user_id for account-level summary
            by_account = {}
            for utterance in untimestamped_email_utterances:
                user_id = utterance.get("user_id", "unknown")
                if user_id not in by_account:
                    by_account[user_id] = []
                by_account[user_id].append(utterance)

            for user_id, utterances in sorted(by_account.items()):
                logger.info(
                    f"📧 {user_id}: {len(utterances)} utterances without timestamps"
                )

                # Show first few utterances
                for utterance in utterances[:2]:
                    utterance_text = utterance.get("Utterance", "")
                    logger.info(f"   • {utterance_text[:100]}...")

                if len(utterances) > 2:
                    logger.info(f"   ... and {len(utterances) - 2} more")
                logger.info("")
        else:
            logger.info("✅ All email utterances received timestamps!")

        logger.info("=" * 80)

    def _extract_untimestamped_email_utterances(
        self, processed_utterances: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract email utterances that don't have timestamps.

        Args:
            processed_utterances: The processed utterances list

        Returns:
            List of email utterances without timestamps
        """
        untimestamped_utterances = []

        for utterance in processed_utterances:
            segment = utterance.get("segment") or utterance.get("Segment", "")
            if (
                segment
                and segment.lower() == "email"
                and not utterance.get("timestamp")
            ):
                # Create a copy of the utterance without the timestamp field
                clean_utterance = utterance.copy()
                # Remove timestamp field if it exists but is empty
                if "timestamp" in clean_utterance and not clean_utterance["timestamp"]:
                    del clean_utterance["timestamp"]
                untimestamped_utterances.append(clean_utterance)

        return untimestamped_utterances


def main():
    """Main entry point using Python Fire for CLI interface."""
    fire.Fire(FileUtteranceTimestamper)


if __name__ == "__main__":
    main()
