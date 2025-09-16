#!/usr/bin/env python3
"""
File Extractor
==========================================

This module reads all files in a specified folder and generates a
JSON output file containing folder statistics and individual file contents.

Features:
- Scans entire folder (including subdirectories)
- Handles multiple file formats (PDF, Word, Excel, PowerPoint, Text, etc.)
- Generates comprehensive JSON output with folder stats and file contents
- Robust error handling and logging
- Configurable content preservation (full content vs. previews)

Usage:
    python file_reader_example.py analyze_folder --input_folder="data" --output_file="folder_analysis.json"
    python file_reader_example.py analyze_folder --input_folder="documents" --include_subdirs=False
"""

import os
import sys
import json
import fire
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
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
        logging.FileHandler("files_extractor.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class FolderContentExtractor:
    """Production module for comprehensive folder content analysis"""

    def __init__(self, silent: bool = True):
        self.silent = silent
        self.reader = UniversalFileReader(
            silent=silent
        )  # Enable silent mode to reduce noise

    def _log_info(self, message: str) -> None:
        """Log info message only if not in silent mode"""
        if not self.silent:
            logger.info(message)

    def analyze_folder(
        self,
        folder_path: str,
        output_file: str = "files_data.json",
        include_subdirs: bool = True,
        max_content_length: Optional[int] = None,
        include_empty_files: bool = False,
        prompt_file: str = "prompt_file_gen_complex_utterance.md",
    ) -> Dict[str, Any]:
        """Analyze all files in a folder and generate comprehensive JSON report

        Args:
            folder_path: Path to the folder to analyze
            output_file: Output JSON file path
            include_subdirs: Whether to include subdirectories (recursive scan)
            max_content_length: Maximum content length to include (None = full content)
            include_empty_files: Whether to include empty files in the output
            prompt_file: Prompt template file name (default: prompt_file_gen_complex_utterance.md)

        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info("🚀 Starting folder analysis...")
        logger.info(f"📁 Input folder: {folder_path}")
        logger.info(f"💾 Output file: {output_file}")
        logger.info(f"🔄 Include subdirectories: {include_subdirs}")
        logger.info(f"📄 Include empty files: {include_empty_files}")
        logger.info(f"📝 Prompt template file: {prompt_file}")

        # Validate input folder
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Input folder not found: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        # Discover all files
        files_info = self._discover_files(folder_path, include_subdirs)

        logger.info(f"📋 Discovered {len(files_info)} files, processing them ...")

        # Process files and build analysis results
        analysis_results = self._process_files(
            files_info, max_content_length, include_empty_files
        )

        # Generate final report
        final_report = self._generate_report(
            folder_path, analysis_results, include_subdirs, include_empty_files
        )

        # Save to file
        self._save_report(final_report, output_file)

        # Generate and save prompt file
        prompt_file_path = self._save_prompt_file(
            final_report, output_file, prompt_file
        )

        # Print summary - ensure this always happens
        try:
            self._print_summary(final_report, output_file, prompt_file_path)
        except Exception as e:
            logger.error(f"❌ Error printing summary: {e}")
            # Print basic summary even if detailed summary fails
            logger.info("🎉 Analysis Complete!")
            logger.info(f"📁 Folder: {folder_path}")
            logger.info(f"💾 Output File: {output_file}")
            logger.info(f"📝 Prompt File: {prompt_file_path}")

        return final_report

    def _discover_files(
        self, folder_path: str, include_subdirs: bool
    ) -> List[Dict[str, Any]]:
        """Discover all files in the folder with metadata

        Args:
            folder_path: Path to scan
            include_subdirs: Whether to scan recursively

        Returns:
            List[Dict[str, Any]]: File information list
        """
        files_info = []
        folder_path_obj = Path(folder_path)

        # Choose scanning method
        if include_subdirs:
            file_paths = folder_path_obj.rglob("*")
        else:
            file_paths = folder_path_obj.glob("*")

        for file_path in file_paths:
            if file_path.is_file():
                try:
                    stat_info = file_path.stat()
                    file_info = {
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "file_extension": file_path.suffix.lower(),
                        "file_size_bytes": stat_info.st_size,
                        "modified_time": datetime.fromtimestamp(
                            stat_info.st_mtime
                        ).isoformat(),
                        "is_empty": stat_info.st_size == 0,
                        "is_supported": file_path.suffix.lower()
                        in self.reader.all_extensions,
                    }
                    files_info.append(file_info)
                except Exception as e:
                    logger.warning(f"⚠️ Error getting info for {file_path}: {e}")

        return files_info

    def _clean_file_data(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove unwanted fields from file data for final output

        Args:
            file_data: Raw file data dictionary

        Returns:
            Dict[str, Any]: Cleaned file data dictionary
        """
        # Create a copy and remove unwanted fields
        cleaned_data = {
            k: v
            for k, v in file_data.items()
            if k not in ["relative_path", "file_size_mb", "is_empty", "is_supported"]
        }
        return cleaned_data

    def _process_files(
        self,
        files_info: List[Dict[str, Any]],
        max_content_length: Optional[int],
        include_empty_files: bool,
    ) -> Dict[str, Any]:
        """Process all files and extract content

        Args:
            files_info: List of file information
            max_content_length: Maximum content length (None = extract full content without truncation)
            include_empty_files: Whether to process empty files

        Returns:
            Dict[str, Any]: Processing results
        """
        results = {
            "processed_files": [],
            "skipped_files": [],
            "failed_files": [],
            "empty_files": [],
        }

        processed_count = 0
        skipped_count = 0
        failed_count = 0
        empty_count = 0

        for file_info in files_info:
            try:
                # Skip empty files if not requested
                if file_info["is_empty"]:
                    empty_count += 1
                    results["empty_files"].append(
                        {
                            "file_name": file_info["file_name"],
                            "file_path": file_info["file_path"],
                            "reason": "Empty file",
                        }
                    )
                    logger.info(f"Empty file: {file_info['file_name']}")
                    if not include_empty_files:
                        continue

                # Skip unsupported files
                if not file_info["is_supported"]:
                    skipped_count += 1
                    results["skipped_files"].append(
                        {
                            "file_name": file_info["file_name"],
                            "file_path": file_info["file_path"],
                            "reason": f"Unsupported extension: {file_info['file_extension']}",
                        }
                    )
                    logger.info(f"Skipped unsupported file: {file_info['file_name']}")
                    continue

                # Process the file
                self._log_info(f"📖 Processing: {file_info['file_name']}")

                if file_info["is_empty"]:
                    # Handle empty files
                    processed_file = {
                        **file_info,
                        "content": "",
                        "content_length": 0,
                        "extraction_method": "empty_file",
                        "status": "success",
                    }
                else:
                    # Read file content
                    read_result = self.reader.read_file(file_info["file_path"])

                    if read_result["status"] == "success":
                        content = read_result["content"]

                        # Apply content length limit if specified
                        if max_content_length and len(content) > max_content_length:
                            content_preview = content[:max_content_length]
                            processed_file = {
                                **file_info,
                                "content": content_preview,
                                "content_full_length": len(content),
                                "content_length": len(content_preview),
                                "content_truncated": True,
                                "extraction_method": read_result.get(
                                    "extraction_method", "unknown"
                                ),
                                "status": "success",
                            }
                        else:
                            processed_file = {
                                **file_info,
                                "content": content,
                                "content_length": len(content),
                                "content_truncated": False,
                                "extraction_method": read_result.get(
                                    "extraction_method", "unknown"
                                ),
                                "status": "success",
                            }

                        # Add format-specific metadata
                        for key in [
                            "total_pages",
                            "total_sheets",
                            "slide_count",
                            "format_info",
                        ]:
                            if key in read_result:
                                processed_file[key] = read_result[key]

                    else:
                        # Failed to read
                        failed_count += 1
                        results["failed_files"].append(
                            self._clean_file_data(
                                {
                                    **file_info,
                                    "error": read_result["error"],
                                    "status": "failed",
                                }
                            )
                        )
                        continue

                results["processed_files"].append(self._clean_file_data(processed_file))
                processed_count += 1
                self._log_info(
                    f"✅ Success: {processed_file['content_length']} characters"
                )

            except Exception as e:
                failed_count += 1
                results["failed_files"].append(
                    self._clean_file_data(
                        {**file_info, "error": str(e), "status": "failed"}
                    )
                )
                logger.error(f"❌ Error processing {file_info['file_name']}: {e}")

        self._log_info(f"📊 Processing Summary:")
        self._log_info(f"   ✅ Processed: {processed_count}")
        self._log_info(f"   ⏭️ Skipped: {skipped_count}")
        self._log_info(f"   ❌ Failed: {failed_count}")
        self._log_info(f"   📄 Empty: {empty_count}")

        return results

    def _generate_report(
        self,
        folder_path: str,
        analysis_results: Dict[str, Any],
        include_subdirs: bool,
        include_empty_files: bool,
    ) -> Dict[str, Any]:
        """Generate the final comprehensive report

        Args:
            folder_path: Original input folder
            analysis_results: Processing results
            include_subdirs: Whether subdirectories were included
            include_empty_files: Whether empty files were included

        Returns:
            Dict[str, Any]: Final report
        """
        total_files = (
            len(analysis_results["processed_files"])
            + len(analysis_results["skipped_files"])
            + len(analysis_results["failed_files"])
            + len(analysis_results["empty_files"])
        )

        # Calculate total content length
        total_content_length = sum(
            file_data["content_length"]
            for file_data in analysis_results["processed_files"]
        )

        # Get file extension statistics
        extension_stats = {}
        for file_data in analysis_results["processed_files"]:
            ext = file_data["file_extension"] or "no_extension"
            extension_stats[ext] = extension_stats.get(ext, 0) + 1

        # Calculate file size statistics (in bytes)
        file_sizes = [
            file_data["file_size_bytes"]
            for file_data in analysis_results["processed_files"]
        ]

        # File size distribution (create buckets)
        size_distribution = {}
        if file_sizes:
            max_size = max(file_sizes)
            min_size = min(file_sizes)

            # Define size buckets in bytes
            size_buckets = [
                (0, 1024, "0-1KB"),
                (1024, 10240, "1KB-10KB"),
                (10240, 102400, "10KB-100KB"),
                (102400, 1048576, "100KB-1MB"),
                (1048576, 10485760, "1MB-10MB"),
                (10485760, float("inf"), "10MB+"),
            ]

            for min_bucket, max_bucket, label in size_buckets:
                count = sum(1 for size in file_sizes if min_bucket <= size < max_bucket)
                if count > 0:
                    size_distribution[label] = count
        else:
            max_size = 0
            min_size = 0

        # Overall statistics
        overall_stats = {
            "input_folder": folder_path,
            "scan_recursive": include_subdirs,
            "include_empty_files": include_empty_files,
            "total_files_discovered": total_files,
            "successful_files": len(analysis_results["processed_files"]),
            "skipped_files": len(analysis_results["skipped_files"]),
            "failed_files": len(analysis_results["failed_files"]),
            "empty_files": len(analysis_results["empty_files"]),
            "total_content_length_characters": total_content_length,
            "file_extension_stats": extension_stats,
            "file_size_stats_bytes": {
                "maximum_file_size": max_size,
                "minimum_file_size": min_size,
                "file_size_distribution": size_distribution,
            },
        }

        # Build final report
        final_report = {
            "metadata": {
                "generated_by": "files_extractor.py",
                "version": "1.0.0",
            },
            "overall_statistics": overall_stats,
            "files": analysis_results["processed_files"],
            "processing_issues": {
                "skipped_files": analysis_results["skipped_files"],
                "failed_files": analysis_results["failed_files"],
                "empty_files": analysis_results["empty_files"],
            },
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

        self._log_info(f"💾 Report saved to: {output_file}")

    def _create_prompt_ready_json(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simplified JSON for prompt insertion with reduced content

        Args:
            report: Full report data

        Returns:
            Dict[str, Any]: Simplified JSON for prompt use
        """
        # Create simplified JSON by removing specified fields
        prompt_json = {}

        # Remove metadata entirely
        # Keep overall_statistics but only with total files count and files field
        if "overall_statistics" in report:
            stats = report["overall_statistics"]
            prompt_json["overall_statistics"] = {
                "total_files_discovered": stats.get("total_files_discovered", 0)
            }

        # Process files array with field removal
        if "files" in report:
            simplified_files = []
            for file_data in report["files"]:
                # Create simplified file entry by excluding specific fields
                simplified_file = {
                    k: v
                    for k, v in file_data.items()
                    if k
                    not in [
                        "file_path",
                        "file_extension",
                        "content_length",
                        "content_truncated",
                        "extraction_method",
                        "status",
                    ]
                }
                simplified_files.append(simplified_file)

            prompt_json["files"] = simplified_files

        # Include processing_issues if they exist (might be relevant for context)
        if "processing_issues" in report:
            prompt_json["processing_issues"] = report["processing_issues"]

        return prompt_json

    def _generate_complete_prompt(
        self, prompt_json: Dict[str, Any], prompt_file: str
    ) -> str:
        """Generate complete prompt by inserting JSON into the prompt template

        Args:
            prompt_json: Simplified JSON data for prompt
            prompt_file: Prompt template file name

        Returns:
            str: Complete prompt ready for LLM
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
        json_str = json.dumps(prompt_json, indent=2, ensure_ascii=False)

        # Insert JSON into template
        complete_prompt = prompt_template.replace("{file_json_data}", json_str)

        return complete_prompt

    def _save_prompt_file(
        self, report: Dict[str, Any], output_file: str, prompt_file: str
    ) -> str:
        """Save the complete prompt file with JSON data inserted

        Args:
            report: Full report data
            output_file: Original output file path (used to derive prompt file path)
            prompt_file: Prompt template file name

        Returns:
            str: Path to the saved prompt file
        """
        # Create prompt-ready JSON
        prompt_json = self._create_prompt_ready_json(report)

        # Generate complete prompt
        complete_prompt = self._generate_complete_prompt(prompt_json, prompt_file)

        # Determine prompt file path using same format as output_file
        # Extract directory and filename components while preserving path separators
        if "\\" in output_file or "/" in output_file:
            # Has directory component
            if "\\" in output_file:
                separator = "\\"
                dir_part, file_part = output_file.rsplit("\\", 1)
            else:
                separator = "/"
                dir_part, file_part = output_file.rsplit("/", 1)

            # Extract filename without extension
            if "." in file_part:
                filename_base = file_part.rsplit(".", 1)[0]
            else:
                filename_base = file_part

            prompt_file_path = f"{dir_part}{separator}{filename_base}_prompt.md"
        else:
            # No directory component
            if "." in output_file:
                filename_base = output_file.rsplit(".", 1)[0]
            else:
                filename_base = output_file
            prompt_file_path = f"{filename_base}_prompt.md"

        # Create directory if needed
        prompt_path_obj = Path(prompt_file_path)
        prompt_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save prompt file
        with open(prompt_file_path, "w", encoding="utf-8") as f:
            f.write(complete_prompt)

        return prompt_file_path

    def _print_summary(
        self, report: Dict[str, Any], output_file: str, prompt_file_path: str
    ) -> None:
        """Print a summary of the analysis

        Args:
            report: Final report
            output_file: Path to the output file
            prompt_file_path: Path to the generated prompt file
        """
        stats = report["overall_statistics"]

        logger.info(f"🎉 Files processing Complete!")
        logger.info(f"=" * 50)
        logger.info(f"📁 Folder: {stats['input_folder']}")
        logger.info(f"💾 Output File: {output_file}")
        logger.info(f"📝 Prompt File: {prompt_file_path}")
        logger.info(f"📊 Total Files Discovered: {stats['total_files_discovered']}")
        logger.info(f"✅ Successfully Processed: {stats['successful_files']}")
        logger.info(f"⏭️ Skipped (Unsupported): {stats['skipped_files']}")
        logger.info(f"❌ Failed to Process: {stats['failed_files']}")
        logger.info(f"📄 Empty Files: {stats['empty_files']}")
        logger.info(
            f"📝 Total Content Length: {stats['total_content_length_characters']:,} characters"
        )

        if stats["file_extension_stats"]:
            logger.info(f"📋 File Types Processed:")
            for ext, count in sorted(stats["file_extension_stats"].items()):
                ext_display = ext if ext != "no_extension" else "(no extension)"
                logger.info(f"   {ext_display}: {count} files")

    def get_folder_info(self, folder_path: str) -> Dict[str, Any]:
        """Get basic information about a folder without processing files

        Args:
            folder_path: Path to analyze

        Returns:
            Dict[str, Any]: Folder information
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        files_info = self._discover_files(folder_path, include_subdirs=True)

        # Count by categories
        supported_count = sum(1 for f in files_info if f["is_supported"])
        empty_count = sum(1 for f in files_info if f["is_empty"])
        total_size = sum(f["file_size_bytes"] for f in files_info)

        return {
            "folder_path": folder_path,
            "total_files": len(files_info),
            "supported_files": supported_count,
            "empty_files": empty_count,
            "total_size_mb": round(total_size / (1024 * 1024), 3),
            "file_extensions": list(set(f["file_extension"] for f in files_info)),
        }


def main():
    """Main entry point for Fire CLI"""
    try:
        # Capture stdout to suppress Fire's automatic return value printing
        stdout_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer):
            fire.Fire(FolderContentExtractor, serialize=False)
    except Exception as e:
        # Ensure errors are visible even with stdout capture
        logger.error(f"❌ Critical error in main: {e}")
        raise


if __name__ == "__main__":
    main()
