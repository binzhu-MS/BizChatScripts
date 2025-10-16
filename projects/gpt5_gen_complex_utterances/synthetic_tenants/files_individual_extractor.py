#!/usr/bin/env python3
"""
File Permissions Analyzer
==========================================

This module analyzes a files.config.json file to extract file access information
for users and channels. It provides comprehensive analysis of file permissions,
sharing patterns, and user access statistics.

Features:
- Analyzes file sharing permissions (view/edit) for each user and email
- Orders users by total number of accessible files
- Generates detailed JSON report with user access patterns
- Provides console summary with logging
- Handles both shared files and file ownership
- Advanced prompt generation: Generates filled prompt templates with extracted file data for LLM evaluation
  - Prompts include guidance for syntactic variety (avoid template patterns)
  - Requires natural temporal expressions in utterances
  - Requires query timestamps in ISO 8601 format
  - Output format: TSV with Utterance, Email_Account, Query_Timestamp, Complexity_Reason

Usage:
    python files_individual_extractor.py analyze_file_access --config_file="data/files.config.json" --output_file="results/permissions_analysis.json"
    python files_individual_extractor.py get_user_accessible_files --config_file="data/files.config.json" --user_email="johndoe" --output_folder="results"
    python files_individual_extractor.py extract_user_file_content_with_prompt --config_file="data/files.config.json" --user_email="johndoe" --output_folder="results"
"""

import os
import sys
import json
import fire
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from collections import defaultdict, Counter
from contextlib import redirect_stdout
import io

# Add the root project directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.file_reader import UniversalFileReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class FileAccessAnalyzer:
    """Analyzer for file permissions and user access patterns"""

    def _load_config_file(self, config_file: str) -> List[Dict[str, Any]]:
        """Load and parse the config JSON file

        Args:
            config_file: Path to the config file

        Returns:
            List[Dict[str, Any]]: Parsed file data
        """
        logger.info(f"📖 Loading config file: {config_file}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            logger.info(f"✅ Successfully loaded {len(config_data)} file entries")
            return config_data

        except json.JSONDecodeError as e:
            error_msg = f"❌ Invalid JSON in config file: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"❌ Error reading config file: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _get_file_timestamps(
        self, file_location: str, config_file: str
    ) -> Dict[str, str]:
        """Get file timestamps (creation and modification time)

        Args:
            file_location: Path to the file
            config_file: Config file path for relative path resolution

        Returns:
            Dict[str, str]: Dictionary with creation_time, modification_time, and file_size
        """
        timestamps = {
            "creation_time": None,
            "modification_time": None,
            "file_size": None,
            "file_exists": False,
        }

        try:
            # Construct full file path
            full_file_path = file_location
            if not os.path.isabs(file_location):
                # If relative path, assume it's relative to config file directory
                config_dir = os.path.dirname(config_file)
                full_file_path = os.path.join(config_dir, file_location)

            # Check if file exists
            if os.path.exists(full_file_path):
                timestamps["file_exists"] = True
                stat_info = os.stat(full_file_path)

                # Get timestamps in ISO format
                timestamps["creation_time"] = datetime.fromtimestamp(
                    stat_info.st_ctime
                ).isoformat()
                timestamps["modification_time"] = datetime.fromtimestamp(
                    stat_info.st_mtime
                ).isoformat()
                timestamps["file_size"] = stat_info.st_size
            else:
                logger.warning(
                    f"⚠️ File not found for timestamp analysis: {full_file_path}"
                )

        except Exception as e:
            logger.warning(f"⚠️ Could not get timestamps for {file_location}: {e}")

        return timestamps

    def _analyze_user_permissions(
        self, config_data: List[Dict[str, Any]], config_file: str
    ) -> Dict[str, Any]:
        """Analyze user permissions and access patterns

        Args:
            config_data: List of file configuration data
            config_file: Config file path for timestamp resolution

        Returns:
            Dict[str, Any]: User analysis results
        """
        logger.info("🔍 Analyzing user permissions...")

        # Track user access patterns
        user_files = defaultdict(list)  # user -> list of accessible files
        user_permissions = defaultdict(set)  # user -> set of permission levels
        file_owners = {}  # file_id -> owner
        user_owned_files = defaultdict(list)  # owner -> list of owned files
        sharing_channels = defaultdict(
            list
        )  # user -> list of sharing channels (owners/emails)

        # Permission type statistics
        permission_stats = Counter()

        for file_entry in config_data:
            file_id = file_entry.get("FileId", "")
            file_name = file_entry.get("FileName", "")
            file_location = file_entry.get("FileLocation", "")
            owner = file_entry.get("Owner", "")
            shared_with = file_entry.get("SharedWith", [])
            file_destination = file_entry.get("FileDestination", "")

            # Get file timestamps
            file_timestamps = self._get_file_timestamps(file_location, config_file)

            # Track file owner
            if owner:
                file_owners[file_id] = owner
                file_info = {
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_location": file_location,
                    "permission_level": "owner",
                    "shared_channel": "owner",
                    "file_destination": file_destination,
                    **file_timestamps,  # Add all timestamp information
                }

                user_owned_files[owner].append(file_info)
                # Owner has implicit access to their own files
                user_files[owner].append(file_info)
                user_permissions[owner].add("owner")
                permission_stats["owner"] += 1

            # Process shared permissions
            if shared_with:  # Check if shared_with is not None
                for share_info in shared_with:
                    email = share_info.get("Email", "")
                    permission_level = share_info.get("PermissionLevel", "")

                    if email and permission_level:
                        shared_file_info = {
                            "file_id": file_id,
                            "file_name": file_name,
                            "file_location": file_location,
                            "permission_level": permission_level,
                            "shared_channel": (
                                f"shared_by_{owner}" if owner else "shared"
                            ),
                            "file_destination": file_destination,
                            **file_timestamps,  # Add all timestamp information
                        }

                        user_files[email].append(shared_file_info)
                        user_permissions[email].add(permission_level)
                        permission_stats[permission_level] += 1

                        # Track sharing channel
                        sharing_channels[email].append(
                            f"shared_by_{owner}" if owner else "shared"
                        )

        # Sort users by total number of accessible files (descending)
        users_by_file_count = sorted(
            user_files.items(), key=lambda x: len(x[1]), reverse=True
        )

        # Generate user statistics
        user_stats = []
        for user, files in users_by_file_count:
            channels = list(set(sharing_channels[user]))  # Remove duplicates

            # Count permission types for this user
            permission_counts = Counter(file["permission_level"] for file in files)

            # Group files by permission type
            files_by_permission = {"owner": [], "view": [], "edit": []}

            for file_info in files:
                permission = file_info["permission_level"]
                if permission in files_by_permission:
                    files_by_permission[permission].append(file_info)

            user_stats.append(
                {
                    "user_email": user,
                    "total_accessible_files": len(files),
                    "permission_counts": dict(permission_counts),
                    "sharing_channels": channels,
                    "accessible_files": files_by_permission,
                }
            )

        analysis_results = {
            "total_users": len(user_files),
            "total_files": len(config_data),
            "permission_statistics": dict(permission_stats),
            "users_ordered_by_access": user_stats,
            "file_owners": file_owners,
        }

        logger.info(f"📊 Analysis complete:")

        return analysis_results

    def _generate_permissions_report(
        self,
        config_data: List[Dict[str, Any]],
        user_analysis: Dict[str, Any],
        config_file: str,
    ) -> Dict[str, Any]:
        """Generate the final comprehensive permissions report

        Args:
            config_data: Original config data
            user_analysis: User analysis results
            config_file: Original config file path

        Returns:
            Dict[str, Any]: Final report
        """

        # Generate file-centric view for additional insights
        file_sharing_patterns = []
        for file_entry in config_data:
            file_id = file_entry.get("FileId", "")
            file_name = file_entry.get("FileName", "")
            file_location = file_entry.get("FileLocation", "")
            owner = file_entry.get("Owner", "")
            shared_with = file_entry.get("SharedWith", [])

            # Get file timestamps for the sharing patterns
            file_timestamps = self._get_file_timestamps(file_location, config_file)

            sharing_info = {
                "file_id": file_id,
                "file_name": file_name,
                "file_location": file_location,
                "owner": owner,
                "total_shared_users": len(shared_with) if shared_with else 0,
                "shared_users": [],
                **file_timestamps,  # Add all timestamp information
            }

            if shared_with:  # Check if shared_with is not None
                for share_info in shared_with:
                    sharing_info["shared_users"].append(
                        {
                            "email": share_info.get("Email", ""),
                            "permission_level": share_info.get("PermissionLevel", ""),
                        }
                    )

            file_sharing_patterns.append(sharing_info)

        # Calculate additional statistics
        sharing_stats = {
            "files_with_no_sharing": len(
                [f for f in file_sharing_patterns if f["total_shared_users"] == 0]
            ),
            "files_with_sharing": len(
                [f for f in file_sharing_patterns if f["total_shared_users"] > 0]
            ),
            "average_sharing_per_file": (
                sum(f["total_shared_users"] for f in file_sharing_patterns)
                / len(file_sharing_patterns)
                if file_sharing_patterns
                else 0
            ),
            "max_sharing_count": max(
                (f["total_shared_users"] for f in file_sharing_patterns), default=0
            ),
        }

        # Calculate file timestamp statistics
        existing_files = [
            f for f in file_sharing_patterns if f.get("file_exists", False)
        ]

        # Calculate timestamp extremes
        creation_times = [
            f.get("creation_time") for f in existing_files if f.get("creation_time")
        ]
        modification_times = [
            f.get("modification_time")
            for f in existing_files
            if f.get("modification_time")
        ]

        file_timestamp_stats = {
            "total_files_found": len(existing_files),
            "total_files_missing": len(file_sharing_patterns) - len(existing_files),
            "average_file_size_bytes": (
                sum(f.get("file_size", 0) for f in existing_files) / len(existing_files)
                if existing_files
                else 0
            ),
            "total_size_bytes": sum(f.get("file_size", 0) for f in existing_files),
            "earliest_creation_time": min(creation_times) if creation_times else None,
            "latest_creation_time": max(creation_times) if creation_times else None,
            "earliest_modification_time": (
                min(modification_times) if modification_times else None
            ),
            "latest_modification_time": (
                max(modification_times) if modification_times else None
            ),
        }

        # Add only file count/size stats to sharing stats (not timestamps)
        file_stats_for_sharing = {
            "total_files_found": file_timestamp_stats["total_files_found"],
            "total_files_missing": file_timestamp_stats["total_files_missing"],
            "average_file_size_bytes": file_timestamp_stats["average_file_size_bytes"],
            "total_size_bytes": file_timestamp_stats["total_size_bytes"],
        }

        sharing_stats.update(file_stats_for_sharing)

        # Build final report
        final_report = {
            "metadata": {
                "generated_by": "files_individual_extractor.py",
                "version": "1.0.0",
                "analysis_timestamp": datetime.now().isoformat(),
                "source_config_file": config_file,
            },
            "summary_statistics": {
                "total_users": user_analysis["total_users"],
                "total_files": user_analysis["total_files"],
                "permission_statistics": user_analysis["permission_statistics"],
                # Add timestamp extremes right after basic statistics
                "earliest_creation_time": file_timestamp_stats.get(
                    "earliest_creation_time"
                ),
                "latest_creation_time": file_timestamp_stats.get(
                    "latest_creation_time"
                ),
                "earliest_modification_time": file_timestamp_stats.get(
                    "earliest_modification_time"
                ),
                "latest_modification_time": file_timestamp_stats.get(
                    "latest_modification_time"
                ),
                "users_ordered_by_access": user_analysis["users_ordered_by_access"],
                "file_sharing_statistics": sharing_stats,
            },
            "user_access_analysis": user_analysis["users_ordered_by_access"],
            "file_sharing_patterns": file_sharing_patterns,
        }

        return final_report

    def _save_report(self, report: Dict[str, Any], output_file: str) -> None:
        """Save the report to a JSON file

        Args:
            report: Report data
            output_file: Output file path
        """
        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Report saved to: {output_file}")

    def _print_permissions_summary(
        self, report: Dict[str, Any], output_file: str
    ) -> None:
        """Print a summary of the permissions analysis

        Args:
            report: Final report
            output_file: Path to the output file
        """
        summary = report["summary_statistics"]
        users = report["user_access_analysis"]

        logger.info(f"🎉 File Permissions Analysis Complete!")
        logger.info(f"=" * 60)
        logger.info(
            f"📁 Source Config File: {report['metadata']['source_config_file']}"
        )
        logger.info(f"💾 Output File: {output_file}")
        logger.info(f"📊 Summary Statistics:")
        logger.info(f"   👥 Total Users: {summary['total_users']}")
        logger.info(f"   📁 Total Files: {summary['total_files']}")
        logger.info(
            f"   🔄 Files with Sharing: {summary['file_sharing_statistics']['files_with_sharing']}"
        )
        logger.info(
            f"   🔒 Files without Sharing: {summary['file_sharing_statistics']['files_with_no_sharing']}"
        )
        logger.info(
            f"   📈 Average Files Shared per File: {summary['file_sharing_statistics']['average_sharing_per_file']:.2f}"
        )

        logger.info(f"📁 File System Statistics:")
        logger.info(
            f"   ✅ Files Found: {summary['file_sharing_statistics']['total_files_found']}"
        )
        logger.info(
            f"   ❌ Files Missing: {summary['file_sharing_statistics']['total_files_missing']}"
        )
        logger.info(
            f"   📏 Average File Size: {summary['file_sharing_statistics']['average_file_size_bytes']:.0f} bytes"
        )
        logger.info(
            f"   💾 Total Size: {summary['file_sharing_statistics']['total_size_bytes']:,} bytes"
        )

        # Add timestamp range information
        earliest_creation = summary.get("earliest_creation_time")
        latest_creation = summary.get("latest_creation_time")
        earliest_modification = summary.get("earliest_modification_time")
        latest_modification = summary.get("latest_modification_time")

        if earliest_creation and latest_creation:
            logger.info(f"📅 Creation Time Range:")
            logger.info(f"   ⏪ Earliest: {earliest_creation}")
            logger.info(f"   ⏩ Latest: {latest_creation}")

        if earliest_modification and latest_modification:
            logger.info(f"📝 Modification Time Range:")
            logger.info(f"   ⏪ Earliest: {earliest_modification}")
            logger.info(f"   ⏩ Latest: {latest_modification}")

        logger.info(f"🔑 Permission Type Distribution:")
        for perm_type, count in summary["permission_statistics"].items():
            logger.info(f"   {perm_type}: {count}")

    def analyze_file_access(
        self,
        config_file: str,
        output_file: str,
    ) -> Dict[str, Any]:
        """Analyze file permissions from config file and generate comprehensive report

        Args:
            config_file: Path to the files.config.json file
            output_file: Output JSON file path for the analysis results

        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info("🚀 Starting file access analysis...")
        logger.info(f"📁 Config file: {config_file}")
        logger.info(f"💾 Output file: {output_file}")

        # Validate input file
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Load and parse config file
        config_data = self._load_config_file(config_file)

        # Analyze user permissions
        user_analysis = self._analyze_user_permissions(config_data, config_file)

        # Generate comprehensive report
        analysis_report = self._generate_permissions_report(
            config_data, user_analysis, config_file
        )

        # Save to file
        self._save_report(analysis_report, output_file)

        # Print summary
        self._print_permissions_summary(analysis_report, output_file)

        return analysis_report

    def get_user_accessible_files(
        self, config_file: str, user_email: str, output_folder: str = "."
    ) -> Dict[str, Any]:
        """Get a list of files accessible for a specific user

        Args:
            config_file: Path to the files.config.json file
            user_email: Specific user email to analyze (required)
            output_folder: Folder to save the output JSON file (default: current directory)

        Returns:
            Dict[str, Any]: User summary
        """
        logger.info(
            f"🔍 Getting user '{user_email}' accessible files from: {config_file}"
        )

        # Load and analyze data
        config_data = self._load_config_file(config_file)
        user_analysis = self._analyze_user_permissions(config_data, config_file)

        # Find specific user
        user_data = None
        for user in user_analysis["users_ordered_by_access"]:
            if user["user_email"] == user_email:
                user_data = user
                break

        if user_data:
            # Create detailed report for JSON output
            user_report = {
                "metadata": {
                    "generated_by": "files_individual_extractor.py",
                    "version": "1.0.0",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "source_config_file": config_file,
                    "target_user": user_email,
                },
                "user_analysis": user_data,
            }

            # Save to JSON file with specified format: email_account_accessible_files.json
            clean_email = user_email.replace("@", "_at_").replace(".", "_")
            output_filename = f"{clean_email}_accessible_files.json"
            output_path = os.path.join(output_folder, output_filename)
            self._save_report(user_report, output_path)

            # Show console summary
            logger.info(f"📊 Summary for {user_email}:")
            logger.info(
                f"   📁 Total accessible files: {user_data['total_accessible_files']}"
            )
            logger.info(f"   🔑 Permission counts: {user_data['permission_counts']}")
            logger.info(
                f"   📡 Sharing channels: {', '.join(user_data['sharing_channels'])}"
            )
            logger.info(f"💾 Detailed file list saved to: {output_path}")

            return user_data
        else:
            logger.warning(f"❌ User not found: {user_email}")
            return {}

    def extract_user_file_content_with_prompt(
        self,
        config_file: str,
        user_email: str,
        output_folder: str = ".",
        prompt_file: str = "prompt_file_gen_complex_utterance.md",
    ) -> Dict[str, Any]:
        """Extract full content of files accessible to a user and generate prompt

        Args:
            config_file: Path to the files.config.json file
            user_email: Specific user email to analyze (required)
            output_folder: Folder to save the output files (default: current directory)
            prompt_file: Prompt template file name

        Returns:
            Dict[str, Any]: Processing results
        """
        logger.info(f"🚀 Starting user file content extraction for: {user_email}")
        logger.info(f"📁 Config file: {config_file}")
        logger.info(f"📂 Output folder: {output_folder}")
        logger.info(f"📝 Prompt template: {prompt_file}")

        # Initialize file reader
        file_reader = UniversalFileReader(silent=True)

        # Get user's accessible files
        user_data = self.get_user_accessible_files(
            config_file, user_email, output_folder
        )

        if not user_data:
            logger.error(f"❌ No data found for user: {user_email}")
            return {}

        # Process files to extract content
        processed_files = []
        failed_files = []
        total_files = 0
        successful_files = 0

        # Process all accessible files across permission types
        accessible_files = user_data.get("accessible_files", {})

        for permission_type in ["owner", "view", "edit"]:
            files_list = accessible_files.get(permission_type, [])

            for file_info in files_list:
                total_files += 1
                try:
                    # Get file path from file_location
                    file_path = file_info.get("file_location", "")
                    file_name = file_info.get("file_name", "")

                    # logger.info(f"📖 Processing: {file_name}")

                    # Skip if no file path
                    if not file_path:
                        failed_files.append(
                            {
                                "file_name": file_name,
                                "error": "No file path specified",
                                "permission_level": file_info.get(
                                    "permission_level", ""
                                ),
                            }
                        )
                        continue

                    # Construct full file path
                    full_file_path = file_path
                    if not os.path.isabs(file_path):
                        # If relative path, assume it's relative to config file directory
                        config_dir = os.path.dirname(config_file)
                        full_file_path = os.path.join(config_dir, file_path)

                    # Check if file exists
                    if not os.path.exists(full_file_path):
                        failed_files.append(
                            {
                                "file_name": file_name,
                                "error": f"File not found: {full_file_path}",
                                "permission_level": file_info.get(
                                    "permission_level", ""
                                ),
                            }
                        )
                        continue

                    # Read file content
                    read_result = file_reader.read_file(full_file_path)

                    if read_result["status"] == "success":
                        # Get file modification time
                        try:
                            stat_info = os.stat(full_file_path)
                            modified_time = datetime.fromtimestamp(
                                stat_info.st_mtime
                            ).isoformat()
                        except:
                            modified_time = datetime.now().isoformat()

                        # Create processed file entry (remove file_id, add content and modified_time)
                        processed_file = {
                            "file_name": file_info.get("file_name", ""),
                            "file_location": file_info.get("file_location", ""),
                            "permission_level": file_info.get("permission_level", ""),
                            "shared_channel": file_info.get("shared_channel", ""),
                            "file_destination": file_info.get("file_destination", ""),
                            "content": read_result["content"],
                            "modified_time": modified_time,
                            "content_length": len(read_result["content"]),
                        }

                        # Add format-specific metadata if available
                        for key in [
                            "total_pages",
                            "total_sheets",
                            "slide_count",
                            "format_info",
                        ]:
                            if key in read_result:
                                processed_file[key] = read_result[key]

                        processed_files.append(processed_file)
                        successful_files += 1
                        # logger.info(
                        #     f"✅ Success: {len(read_result['content'])} characters extracted"
                        # )

                    else:
                        failed_files.append(
                            {
                                "file_name": file_name,
                                "error": read_result.get("error", "Unknown error"),
                                "permission_level": file_info.get(
                                    "permission_level", ""
                                ),
                            }
                        )
                        logger.warning(
                            f"❌ Failed to read: {file_name} - {read_result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    failed_files.append(
                        {
                            "file_name": file_info.get("file_name", "unknown"),
                            "error": str(e),
                            "permission_level": file_info.get("permission_level", ""),
                        }
                    )
                    logger.error(
                        f"❌ Error processing {file_info.get('file_name', 'unknown')}: {e}"
                    )

        # Create final report without metadata
        files_report = {
            "overall_statistics": {
                "target_user": user_email,
                "total_files_discovered": total_files,
                "successful_files": successful_files,
                "failed_files": len(failed_files),
            },
            "files": processed_files,
            "processing_issues": {"failed_files": failed_files},
        }

        # Save content JSON file
        clean_email = user_email.replace("@", "_at_").replace(".", "_")
        content_filename = f"{clean_email}_files_content.json"
        content_path = os.path.join(output_folder, content_filename)
        self._save_report(files_report, content_path)

        # Generate and save prompt file
        prompt_file_path = self._save_content_prompt_file(
            files_report, content_path, prompt_file, output_folder, user_email
        )

        # Print summary
        self._print_content_summary(files_report, content_path, prompt_file_path)

        return files_report

    def _save_content_prompt_file(
        self,
        report: Dict[str, Any],
        content_file: str,
        prompt_file: str,
        output_folder: str,
        user_email: str,
    ) -> str:
        """Save the complete prompt file with JSON data inserted

        Args:
            report: Full report data
            content_file: Content file path
            prompt_file: Prompt template file name
            output_folder: Output folder
            user_email: User email to replace in template

        Returns:
            str: Path to the saved prompt file
        """
        # Read the prompt template
        prompt_template_path = Path(__file__).parent / prompt_file

        logger.info(f"📖 Reading prompt template: {prompt_template_path}")
        try:
            with open(prompt_template_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            logger.info(
                f"✅ Successfully read prompt template ({len(prompt_template)} characters)"
            )
        except FileNotFoundError:
            error_msg = (
                f"❌ ERROR: Prompt template file not found at {prompt_template_path}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"❌ ERROR: Failed to read prompt template at {prompt_template_path}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # Convert JSON to formatted string
        json_str = json.dumps(report, indent=2, ensure_ascii=False)

        # Replace placeholders in template
        complete_prompt = prompt_template.replace("{file_json_data}", json_str)
        complete_prompt = complete_prompt.replace("{email_account}", user_email)

        # Generate prompt file path
        content_path_obj = Path(content_file)
        filename_base = content_path_obj.stem
        prompt_file_path = os.path.join(output_folder, f"{filename_base}_prompt.md")

        # Create directory if needed
        Path(prompt_file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save prompt file
        with open(prompt_file_path, "w", encoding="utf-8") as f:
            f.write(complete_prompt)

        logger.info(f"📝 Prompt file saved to: {prompt_file_path}")
        return prompt_file_path

    def _print_content_summary(
        self, report: Dict[str, Any], content_file: str, prompt_file_path: str
    ) -> None:
        """Print a summary of the content extraction

        Args:
            report: Final report
            content_file: Path to the content file
            prompt_file_path: Path to the generated prompt file
        """
        stats = report["overall_statistics"]

        logger.info(f"🎉 User File Content Extraction Complete!")
        logger.info(f"=" * 60)
        logger.info(f"👤 Target User: {stats['target_user']}")
        logger.info(f"💾 Content File: {content_file}")
        logger.info(f"📝 Prompt File: {prompt_file_path}")
        logger.info(f"📊 Total Files Discovered: {stats['total_files_discovered']}")
        logger.info(f"✅ Successfully Processed: {stats['successful_files']}")
        logger.info(f"❌ Failed to Process: {stats['failed_files']}")

        # Calculate total content length
        total_content_length = sum(
            file_data["content_length"] for file_data in report["files"]
        )
        logger.info(f"📝 Total Content Length: {total_content_length:,} characters")

        # Show file types processed
        file_extensions = {}
        for file_data in report["files"]:
            file_name = file_data.get("file_name", "")
            ext = Path(file_name).suffix.lower() or "no_extension"
            file_extensions[ext] = file_extensions.get(ext, 0) + 1

        if file_extensions:
            logger.info(f"📋 File Types Processed:")
            for ext, count in sorted(file_extensions.items()):
                ext_display = ext if ext != "no_extension" else "(no extension)"
                logger.info(f"   {ext_display}: {count} files")


def main():
    """Main entry point for Fire CLI"""
    try:
        # Capture stdout to suppress Fire's automatic return value printing
        stdout_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer):
            fire.Fire(FileAccessAnalyzer, serialize=False)
    except Exception as e:
        # Ensure errors are visible even with stdout capture
        logger.error(f"❌ Critical error in main: {e}")
        raise


if __name__ == "__main__":
    main()
