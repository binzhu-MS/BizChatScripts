#!/usr/bin/env python3
"""
Teams Message Analytics Tool

A comprehensive analysis tool for Microsoft Teams chat and channel data, providing statistics on user messaging activity,
extracting accessible messages for specific users, and identifying file references in conversations.
This module focuses on daily collaboration patterns and excludes meeting-type chats from LLM prompt generation.

Features:
- User messaging statistics (sent/received counts, conversation participation) - includes ALL chat types
- Message access analysis for specific users (excludes meeting chats from LLM prompts)
- File reference extraction and analysis (URLs, attachments, file paths)
- Export capabilities in Markdown format
- Chat type breakdown (Group, OneOnOne, Meeting) - statistics include all types
- Top user rankings by various metrics
- Channel message accessibility analysis

Main Functions:
1. generate_comprehensive_analytics: Generate comprehensive analysis including user statistics, channel analysis, and file references
2. extract_user_teams_messages_with_files_and_prompt: Extract user messages (chat+channel), match files, and generate LLM prompts (excludes meeting chats)

Usage Examples:
    # Generate comprehensive analytics report (user stats + file references + channel analysis)
    python teams_message_analytics.py generate_comprehensive_analytics --chats_file="data/chats.config.json" --output_file="results/comprehensive_analytics.md" --include_channel_messages=true --data_folder=data

    # Extract user messages with files and generate LLM prompt (including channel messages, excluding meeting chats)
    python teams_message_analytics.py extract_user_teams_messages_with_files_and_prompt --chats_file="data/chats.config.json" --user_email="alex.khan" --include_channel_messages=true --data_folder=data

    # Generate comprehensive analytics with console display only
    python teams_message_analytics.py generate_comprehensive_analytics --chats_file="data/chats.config.json"

Input Requirements:
- chats.config.json: Main chat data file with conversations and messages

Note: For meeting-focused complex query generation, use meetings_events_analytics.py instead.
This separation enables domain-specific query generation optimized for different collaboration contexts.
"""

import json
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import sys
import logging
import re
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Fire for CLI interface
import fire

# Import file reader
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
try:
    from utils.file_reader import UniversalFileReader
except ImportError as e:
    logger.warning(f"⚠️ Import warning: {e}. File reading features may not work.")


@dataclass
class FileReference:
    """Represents a file reference found in a Teams message"""

    file_type: str  # 'url', 'file_path', 'file_id', 'attachment'
    reference: str  # The actual URL, path, or ID
    file_name: str  # Extracted filename
    file_extension: str  # File extension
    message_id: str
    chat_id: str
    chat_name: str
    from_user: str
    sent_datetime: str
    context: str  # The surrounding message content
    exists_in_files_folder: bool = False  # Whether file exists in data/files
    matching_files: List[str] = field(
        default_factory=list
    )  # Files with matching extensions


@dataclass
class MessageStats:
    """Statistics for a user's messaging activity"""

    sent_count: int = 0
    received_count: int = 0
    group_chats_participated: Set[str] = field(default_factory=set)
    one_on_one_chats: Set[str] = field(default_factory=set)
    meeting_chats: Set[str] = field(default_factory=set)
    total_conversations: int = 0

    def __post_init__(self):
        self.total_conversations = (
            len(self.group_chats_participated)
            + len(self.one_on_one_chats)
            + len(self.meeting_chats)
        )


@dataclass
class ChatMessage:
    """Represents a chat message"""

    chat_id: str
    chat_name: str
    chat_type: str
    message_id: str
    from_user: str
    content: str
    content_type: str
    sent_datetime: str
    members: List[str]


class MessageAnalytics:
    """Main class for analyzing team messaging data"""

    def __init__(self, chats_file: str):
        """
        Initialize the analytics engine

        Args:
            chats_file: Path to chats.config.json
        """
        self.chats_file = chats_file
        self.chats_data = []
        self.user_stats = defaultdict(MessageStats)
        self.available_files = {}  # Dict mapping extension to list of files
        self.files_folder = Path(chats_file).parent / "files"
        self.load_data()
        self.scan_files_folder()

    def scan_files_folder(self):
        """Scan the data/files folder and catalog all available files by extension"""
        if not self.files_folder.exists():
            print(f"Warning: Files folder not found at {self.files_folder}")
            return

        print(f"Scanning files in {self.files_folder}...")
        file_count = 0

        for file_path in self.files_folder.iterdir():
            if file_path.is_file():
                file_count += 1
                extension = file_path.suffix.lower().lstrip(".")
                if extension not in self.available_files:
                    self.available_files[extension] = []
                self.available_files[extension].append(file_path.name)

        total_extensions = len(self.available_files)
        print(
            f"Found {file_count} files with {total_extensions} different extensions in files folder"
        )

    def match_file_references_to_files(
        self, file_references: List[FileReference]
    ) -> List[FileReference]:
        """
        Match file references to actual files in the data/files folder

        Args:
            file_references: List of file references to match

        Returns:
            Updated list with matching information
        """
        print("Matching file references to available files...")

        for ref in file_references:
            # Check for exact filename match
            if ref.file_extension and ref.file_extension in self.available_files:
                exact_matches = [
                    f
                    for f in self.available_files[ref.file_extension]
                    if f.lower() == ref.file_name.lower()
                ]
                if exact_matches:
                    ref.exists_in_files_folder = True
                    ref.matching_files = exact_matches
                else:
                    # Look for files with same extension (potential matches)
                    ref.matching_files = self.available_files[ref.file_extension]

        matched_count = sum(1 for ref in file_references if ref.exists_in_files_folder)
        print(
            f"Found exact matches for {matched_count} out of {len(file_references)} file references"
        )

        return file_references

    def load_data(self):
        """Load chat data from JSON file"""
        try:
            with open(self.chats_file, "r", encoding="utf-8") as f:
                self.chats_data = json.load(f)
            print(f"Loaded {len(self.chats_data)} chats from {self.chats_file}")
        except Exception as e:
            print(f"Error loading chats file: {e}")
            raise

    def collect_user_statistics(self) -> Dict[str, MessageStats]:
        """
        Collect comprehensive statistics for each user

        Returns:
            Dictionary mapping username to MessageStats
        """
        print("Collecting user statistics...")

        for chat in self.chats_data:
            chat_id = chat.get("ChatId", "")
            chat_name = chat.get("ChatName", "")
            chat_type = chat.get("ChatType", "")
            members = chat.get("Members", [])
            messages = chat.get("ChatMessages", [])

            # Track chat participation for all members
            for member in members:
                if chat_type == "Group":
                    self.user_stats[member].group_chats_participated.add(chat_id)
                elif chat_type == "OneOnOne":
                    self.user_stats[member].one_on_one_chats.add(chat_id)
                elif chat_type == "Meeting":
                    self.user_stats[member].meeting_chats.add(chat_id)

            # Count sent messages (handle case where messages might be None)
            if messages:
                for message in messages:
                    from_user = message.get("From", "")
                    if from_user:
                        self.user_stats[from_user].sent_count += 1

                        # Count received messages (all other members in the chat)
                        for member in members:
                            if member != from_user:
                                self.user_stats[member].received_count += 1

        # Update total conversation counts
        for username, stats in self.user_stats.items():
            stats.total_conversations = (
                len(stats.group_chats_participated)
                + len(stats.one_on_one_chats)
                + len(stats.meeting_chats)
            )

        print(f"Collected statistics for {len(self.user_stats)} users")
        return dict(self.user_stats)

    def get_user_accessible_messages(self, username: str) -> List[ChatMessage]:
        """
        Extract all messages that a specific person can access

        Args:
            username: The username to extract messages for

        Returns:
            List of ChatMessage objects accessible to the user
        """
        print(f"Extracting accessible messages for user: {username}")
        accessible_messages = []

        for chat in self.chats_data:
            chat_id = chat.get("ChatId", "")
            chat_name = chat.get("ChatName", "")
            chat_type = chat.get("ChatType", "")
            members = chat.get("Members", [])
            messages = chat.get("ChatMessages", [])

            # Check if user is a member of this chat
            if username in members and messages:
                for message in messages:
                    chat_message = ChatMessage(
                        chat_id=chat_id,
                        chat_name=chat_name,
                        chat_type=chat_type,
                        message_id=message.get("ChatMessageId", ""),
                        from_user=message.get("From", ""),
                        content=message.get("Content", ""),
                        content_type=message.get("ContentType", ""),
                        sent_datetime=message.get("SentDateTime", ""),
                        members=members.copy(),
                    )
                    accessible_messages.append(chat_message)

        # Sort by datetime
        accessible_messages.sort(key=lambda x: x.sent_datetime)
        print(f"Found {len(accessible_messages)} accessible messages for {username}")
        return accessible_messages

    def print_user_statistics(self, top_n: Optional[int] = 10):
        """Print top users by various metrics

        Args:
            top_n: Number of top users to show in rankings. If None, shows all users.
        """
        # Handle case where top_n might be passed as a string from command line
        if isinstance(top_n, str):
            try:
                top_n = int(top_n)
            except ValueError:
                print(f"Warning: Invalid top_n value '{top_n}', using default of 10")
                top_n = 10

        if not self.user_stats:
            self.collect_user_statistics()

        print(f"\n{'='*60}")
        print("USER MESSAGING STATISTICS")
        print(f"{'='*60}")

        # Determine how many users to show
        display_count = len(self.user_stats) if top_n is None else top_n
        display_label = "All" if top_n is None else f"Top {top_n}"

        # Top users by total message activity (sent + received)
        print(f"\n{display_label} Most Active Users by Total Messages:")
        print("-" * 75)
        print(f"{'Rank':<4} {'Username':<20} {'Sent':<8} {'Received':<10} {'Total':<8}")
        print("-" * 75)

        # Calculate total messages and sort by total
        user_totals = []
        for username, stats in self.user_stats.items():
            total_messages = stats.sent_count + stats.received_count
            user_totals.append((username, stats, total_messages))

        sorted_by_total = sorted(user_totals, key=lambda x: x[2], reverse=True)
        display_list = sorted_by_total if top_n is None else sorted_by_total[:top_n]

        for i, (username, stats, total) in enumerate(display_list, 1):
            print(
                f"{i:<4} {username:<20} {stats.sent_count:<8} {stats.received_count:<10} {total:<8}"
            )

        # Most active in conversations
        print(f"\n{display_label} Most Active in Conversations:")
        print("-" * 40)
        sorted_by_conversations = sorted(
            self.user_stats.items(),
            key=lambda x: x[1].total_conversations,
            reverse=True,
        )
        display_list = (
            sorted_by_conversations
            if top_n is None
            else sorted_by_conversations[:top_n]
        )
        for i, (username, stats) in enumerate(display_list, 1):
            print(f"{i:2}. {username:<20} {stats.total_conversations:>6} conversations")

        # Overall summary
        print(f"\nOverall Summary:")
        print("-" * 40)
        total_users = len(self.user_stats)
        total_sent = sum(stats.sent_count for stats in self.user_stats.values())
        total_received = sum(stats.received_count for stats in self.user_stats.values())
        avg_sent = total_sent / total_users if total_users > 0 else 0
        avg_received = total_received / total_users if total_users > 0 else 0

        print(f"Total users: {total_users}")
        print(f"Total messages sent: {total_sent:,}")
        print(f"Total message receptions: {total_received:,}")
        print(f"Average messages sent per user: {avg_sent:.1f}")
        print(f"Average messages received per user: {avg_received:.1f}")

    def export_user_messages(self, username: str, output_file: str):
        """
        Export all accessible messages for a user to a JSON file

        Args:
            username: Username to export messages for
            output_file: Output file path (JSON format)
        """
        messages = self.get_user_accessible_messages(username)
        self._export_json(messages, output_file)
        print(f"Exported {len(messages)} messages to {output_file}")

    def _export_json(self, messages: List[ChatMessage], output_file: str):
        """Export messages as JSON"""
        data = []
        for msg in messages:
            data.append(
                {
                    "chat_id": msg.chat_id,
                    "chat_name": msg.chat_name,
                    "chat_type": msg.chat_type,
                    "message_id": msg.message_id,
                    "from_user": msg.from_user,
                    "content": msg.content,
                    "content_type": msg.content_type,
                    "sent_datetime": msg.sent_datetime,
                    "members": msg.members,
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _generate_markdown_report(
        self,
        stats: Dict[str, MessageStats],
        output_file: str,
        top_n: Optional[int] = None,
    ):
        """Generate markdown report with user statistics"""

        # Determine how many users to show
        display_label = "All" if top_n is None else f"Top {top_n}"

        # Calculate total messages and sort by total
        user_totals = []
        for username, user_stats in stats.items():
            total_messages = user_stats.sent_count + user_stats.received_count
            user_totals.append((username, user_stats, total_messages))

        sorted_by_total = sorted(user_totals, key=lambda x: x[2], reverse=True)
        display_list = sorted_by_total if top_n is None else sorted_by_total[:top_n]

        # Sort by conversations (for chat type breakdown)
        sorted_by_conversations = sorted(
            stats.items(),
            key=lambda x: x[1].total_conversations,
            reverse=True,
        )

        # Generate markdown content
        markdown_content = f"""# Teams Message Analytics Report

Generated on: {self._get_current_timestamp()}

## {display_label} Most Active Users by Total Messages

| Rank | Username | Sent | Received | Total | Conversations |
|------|----------|------|----------|-------|---------------|
"""

        for i, (username, user_stats, total) in enumerate(display_list, 1):
            markdown_content += f"| {i} | {username} | {user_stats.sent_count} | {user_stats.received_count} | {total} | {user_stats.total_conversations} |\n"

        # Overall summary
        total_users = len(stats)
        total_sent = sum(stat.sent_count for stat in stats.values())
        total_received = sum(stat.received_count for stat in stats.values())
        avg_sent = total_sent / total_users if total_users > 0 else 0
        avg_received = total_received / total_users if total_users > 0 else 0

        markdown_content += f"""
## Overall Summary

- **Total users**: {total_users}
- **Total messages sent**: {total_sent:,}
- **Total message receptions**: {total_received:,}
- **Average messages sent per user**: {avg_sent:.1f}
- **Average messages received per user**: {avg_received:.1f}

## Chat Type Breakdown

### Users by Chat Type Participation

| Username | Group Chats | One-on-One | Meetings | Total Conversations |
|----------|-------------|------------|----------|-------------------|
"""

        # Add chat type breakdown for all users (sorted by total conversations)
        for username, user_stats in sorted_by_conversations:
            markdown_content += f"| {username} | {len(user_stats.group_chats_participated)} | {len(user_stats.one_on_one_chats)} | {len(user_stats.meeting_chats)} | {user_stats.total_conversations} |\n"

        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    def _get_current_timestamp(self):
        """Get current timestamp for report generation"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def extract_file_references(self) -> List[FileReference]:
        """
        Extract all file references from Teams messages

        Returns:
            List of FileReference objects containing file information and context
        """
        print("Extracting file references from Teams messages...")
        file_references = []

        # Patterns for different types of file references
        patterns = {
            "url": r'https://[^\s\'"]+\.(?:xlsx?|csv|pdf|docx?|pptx?|txt|zip|json|xml|log|md|py|js|ts|go|cpp|c|h|yaml|yml)',
            "github_url": r'https://github\.com/[^\s\'"]+',
            "file_path": r"(?:/[a-zA-Z0-9_.-]+)+\.(?:xlsx?|csv|pdf|docx?|pptx?|txt|zip|json|xml|log|md|py|js|ts|go|cpp|c|h|yaml|yml)",
            "file_id": r"FileId[\s:]+([a-f0-9\-]{36})",
            "attachment_mention": r"(?:attach|upload|download|file)(?:ing|ed)?\s+(?:the\s+)?([a-zA-Z0-9_.-]+\.(?:xlsx?|csv|pdf|docx?|pptx?|txt|zip|json|xml|log|md|py|js|ts|go|cpp|c|h|yaml|yml))",
            "file_reference": r"\b([a-zA-Z0-9_.-]+\.(?:xlsx?|csv|pdf|docx?|pptx?|txt|zip|json|xml|log|md|py|js|ts|go|cpp|c|h|yaml|yml))\b",
        }

        for chat in self.chats_data:
            chat_id = chat.get("ChatId", "")
            chat_name = chat.get("ChatName", "")
            messages = chat.get("ChatMessages", [])

            if not messages:
                continue

            for message in messages:
                message_id = message.get("ChatMessageId", "")
                from_user = message.get("From", "")
                content = message.get("Content", "")
                sent_datetime = message.get("SentDateTime", "")

                if not content:
                    continue

                # Search for each pattern type
                for file_type, pattern in patterns.items():
                    matches = re.finditer(pattern, content, re.IGNORECASE)

                    for match in matches:
                        if file_type == "file_id":
                            reference = match.group(1)
                            file_name = f"FileId_{reference}"
                            file_extension = ""
                        elif file_type == "attachment_mention":
                            reference = match.group(1)
                            file_name = reference
                            file_extension = self._extract_extension(reference)
                        else:
                            reference = match.group(0)
                            file_name = self._extract_filename(reference)
                            file_extension = self._extract_extension(reference)

                        # Create context (50 characters before and after the match)
                        start_pos = max(0, match.start() - 50)
                        end_pos = min(len(content), match.end() + 50)
                        context = content[start_pos:end_pos].strip()

                        file_ref = FileReference(
                            file_type=file_type,
                            reference=reference,
                            file_name=file_name,
                            file_extension=file_extension,
                            message_id=message_id,
                            chat_id=chat_id,
                            chat_name=chat_name,
                            from_user=from_user,
                            sent_datetime=sent_datetime,
                            context=context,
                        )

                        file_references.append(file_ref)

        # Remove duplicates based on reference and message_id
        seen = set()
        unique_references = []
        for ref in file_references:
            key = (ref.reference, ref.message_id)
            if key not in seen:
                seen.add(key)
                unique_references.append(ref)

        # Sort by datetime
        unique_references.sort(key=lambda x: x.sent_datetime)

        print(f"Found {len(unique_references)} unique file references")

        # Match file references to actual files
        unique_references = self.match_file_references_to_files(unique_references)

        return unique_references

    def _extract_filename(self, reference: str) -> str:
        """Extract filename from a path or URL"""
        if "/" in reference:
            return reference.split("/")[-1]
        return reference

    def _extract_extension(self, filename: str) -> str:
        """Extract file extension from filename"""
        if "." in filename:
            return filename.split(".")[-1].lower()
        return ""

    def print_file_statistics(self, file_references: List[FileReference]):
        """Print statistics about file references"""
        if not file_references:
            print("No file references found.")
            return

        print(f"\n{'='*60}")
        print("FILE REFERENCE STATISTICS")
        print(f"{'='*60}")

        # File matching statistics
        matched_files = [ref for ref in file_references if ref.exists_in_files_folder]
        print(f"\nFile Matching Statistics:")
        print("-" * 40)
        print(f"Total file references: {len(file_references)}")
        print(f"Exact matches found: {len(matched_files)}")
        print(
            f"Match percentage: {len(matched_files)/len(file_references)*100:.1f}%"
            if file_references
            else "0.0%"
        )

        # Available file extensions in data/files folder
        print(f"\nAvailable File Extensions in data/files:")
        print("-" * 40)
        for ext, files in sorted(self.available_files.items()):
            print(f".{ext:<19} {len(files):>6} files")

        # Count by file type
        type_counts = Counter(ref.file_type for ref in file_references)
        print(f"\nFile References by Type:")
        print("-" * 40)
        for file_type, count in type_counts.most_common():
            matched_count = len(
                [
                    ref
                    for ref in file_references
                    if ref.file_type == file_type and ref.exists_in_files_folder
                ]
            )
            print(f"{file_type:<20} {count:>6} ({matched_count} matched)")

        # Count by file extension
        ext_counts = Counter(
            ref.file_extension for ref in file_references if ref.file_extension
        )
        print(f"\nFile References by Extension:")
        print("-" * 40)
        for ext, count in ext_counts.most_common(10):
            matched_count = len(
                [
                    ref
                    for ref in file_references
                    if ref.file_extension == ext and ref.exists_in_files_folder
                ]
            )
            available_count = len(self.available_files.get(ext, []))
            print(
                f".{ext:<19} {count:>6} ({matched_count} matched, {available_count} available)"
            )

        # Count by user
        user_counts = Counter(ref.from_user for ref in file_references)
        print(f"\nTop Users Sharing Files:")
        print("-" * 40)
        for user, count in user_counts.most_common(10):
            matched_count = len(
                [
                    ref
                    for ref in file_references
                    if ref.from_user == user and ref.exists_in_files_folder
                ]
            )
            print(f"{user:<20} {count:>6} ({matched_count} matched)")

        # Count by chat
        chat_counts = Counter(ref.chat_name for ref in file_references)
        print(f"\nTop Chats with File References:")
        print("-" * 50)
        for chat_name, count in chat_counts.most_common(10):
            matched_count = len(
                [
                    ref
                    for ref in file_references
                    if ref.chat_name == chat_name and ref.exists_in_files_folder
                ]
            )
            truncated_name = (
                (chat_name[:45] + "...") if len(chat_name) > 45 else chat_name
            )
            print(f"{truncated_name:<48} {count:>6} ({matched_count} matched)")

        # Show exact matches
        if matched_files:
            print(f"\nExact File Matches Found:")
            print("-" * 60)
            for ref in matched_files[:10]:  # Show first 10
                print(
                    f"  📁 {ref.file_name} ({ref.file_extension}) - Referenced by {ref.from_user}"
                )
            if len(matched_files) > 10:
                print(f"  ... and {len(matched_files) - 10} more matches")

        print(f"\nTotal file references: {len(file_references)}")
        print(f"Total exact matches: {len(matched_files)}")

    def _generate_file_references_markdown(
        self, file_references: List[FileReference], output_file: str
    ):
        """Generate markdown report for file references"""
        if not file_references:
            markdown_content = "# Teams File References Report\n\nNo file references found in the analyzed chats.\n"
        else:
            # Count statistics
            type_counts = Counter(ref.file_type for ref in file_references)
            ext_counts = Counter(
                ref.file_extension for ref in file_references if ref.file_extension
            )
            user_counts = Counter(ref.from_user for ref in file_references)
            chat_counts = Counter(ref.chat_name for ref in file_references)
            matched_files = [
                ref for ref in file_references if ref.exists_in_files_folder
            ]

            markdown_content = f"""# Teams File References Report

Generated on: {self._get_current_timestamp()}

## Summary Statistics

- **Total file references**: {len(file_references)}
- **Exact matches found**: {len(matched_files)}
- **Match percentage**: {len(matched_files)/len(file_references)*100:.1f}%
- **Unique file types**: {len(type_counts)}
- **Users sharing files**: {len(user_counts)}
- **Chats with file references**: {len(chat_counts)}
- **Available file extensions in data/files**: {len(self.available_files)}

## File Matching Summary

| Extension | References | Exact Matches | Available Files |
|-----------|------------|---------------|----------------|
"""

            # Create extension matching table
            all_extensions = set(
                ref.file_extension for ref in file_references if ref.file_extension
            )
            all_extensions.update(self.available_files.keys())

            for ext in sorted(all_extensions):
                ref_count = len(
                    [ref for ref in file_references if ref.file_extension == ext]
                )
                match_count = len(
                    [
                        ref
                        for ref in file_references
                        if ref.file_extension == ext and ref.exists_in_files_folder
                    ]
                )
                avail_count = len(self.available_files.get(ext, []))
                markdown_content += (
                    f"| .{ext} | {ref_count} | {match_count} | {avail_count} |\n"
                )

            markdown_content += """
## File References by Type

| File Type | Count | Exact Matches |
|-----------|-------|---------------|
"""

            for file_type, count in type_counts.most_common():
                matched_count = len(
                    [
                        ref
                        for ref in file_references
                        if ref.file_type == file_type and ref.exists_in_files_folder
                    ]
                )
                markdown_content += f"| {file_type} | {count} | {matched_count} |\n"

            markdown_content += """
## File References by Extension

| Extension | Count |
|-----------|-------|
"""

            for ext, count in ext_counts.most_common(15):
                ext_display = f".{ext}" if ext else "(no extension)"
                markdown_content += f"| {ext_display} | {count} |\n"

            markdown_content += """
## Top Users Sharing Files

| Username | File References |
|----------|----------------|
"""

            for user, count in user_counts.most_common(15):
                markdown_content += f"| {user} | {count} |\n"

            markdown_content += """
## Top Chats with File References

| Chat Name | File References |
|-----------|----------------|
"""

            for chat_name, count in chat_counts.most_common(15):
                # Handle empty or missing chat names
                display_name = chat_name.strip() if chat_name else "(empty)"
                if not display_name:
                    display_name = "(empty)"

                # Escape markdown special characters in chat name
                escaped_name = (
                    display_name.replace("|", "\\|")
                    .replace("[", "\\[")
                    .replace("]", "\\]")
                )
                markdown_content += f"| {escaped_name} | {count} |\n"

            # Add exact matches section if any found
            if matched_files:
                markdown_content += f"""
## Exact File Matches ({len(matched_files)} found)

| File Name | Extension | Referenced By | Chat | Date |
|-----------|-----------|---------------|------|------|
"""
                for ref in matched_files:  # Show all matches
                    escaped_chat = (
                        ref.chat_name.replace("|", "\\|")
                        .replace("[", "\\[")
                        .replace("]", "\\]")
                    )
                    date_part = (
                        ref.sent_datetime.split("T")[0]
                        if "T" in ref.sent_datetime
                        else ref.sent_datetime
                    )
                    markdown_content += f"| {ref.file_name} | .{ref.file_extension} | {ref.from_user} | {escaped_chat} | {date_part} |\n"

            markdown_content += """
## Detailed File References

| Date | User | Chat | File Type | Reference | Match Status | Context |
|------|------|------|-----------|-----------|--------------|---------|
"""

            # Show recent 50 file references
            recent_refs = (
                file_references[-50:] if len(file_references) > 50 else file_references
            )

            for ref in recent_refs:
                # Clean up context for markdown (remove newlines, limit length)
                context_clean = ref.context.replace("\n", " ").replace("\r", " ")[:100]
                if len(ref.context) > 100:
                    context_clean += "..."

                # Escape markdown special characters
                context_clean = context_clean.replace("|", "\\|")
                reference_clean = ref.reference.replace("|", "\\|")
                chat_clean = ref.chat_name.replace("|", "\\|")

                # Extract date from datetime
                date_part = (
                    ref.sent_datetime.split("T")[0]
                    if "T" in ref.sent_datetime
                    else ref.sent_datetime
                )

                # Determine match status
                match_status = (
                    "✅ Exact"
                    if ref.exists_in_files_folder
                    else (
                        "🟡 Extension"
                        if ref.file_extension in self.available_files
                        else "❌ No match"
                    )
                )

                markdown_content += f"| {date_part} | {ref.from_user} | {chat_clean} | {ref.file_type} | {reference_clean} | {match_status} | {context_clean} |\n"

        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    def _generate_comprehensive_markdown_report(
        self,
        user_stats: Dict[str, MessageStats],
        file_references: List[FileReference],
        output_file: str,
        top_n: Optional[int] = None,
        channel_stats: Optional[Dict] = None,
    ):
        """Generate comprehensive markdown report with both user statistics and file references"""

        # Determine how many users to show
        display_label = "All" if top_n is None else f"Top {top_n}"

        # Calculate total messages including both chat and channel messages
        user_totals = []
        for username, user_stats_item in user_stats.items():
            chat_total = user_stats_item.sent_count + user_stats_item.received_count

            # Add channel messages if available
            channel_total = 0
            if channel_stats and username in channel_stats:
                channel_total = (
                    channel_stats[username].total_accessible_messages
                    + channel_stats[username].accessible_replies
                )

            combined_total = chat_total + channel_total
            user_totals.append(
                (username, user_stats_item, chat_total, channel_total, combined_total)
            )

        # Sort by combined total (chat + channel messages)
        sorted_by_total = sorted(user_totals, key=lambda x: x[4], reverse=True)
        display_list = sorted_by_total if top_n is None else sorted_by_total[:top_n]

        # Sort by conversations (for chat type breakdown)
        sorted_by_conversations = sorted(
            user_stats.items(),
            key=lambda x: x[1].total_conversations,
            reverse=True,
        )

        # File statistics
        matched_files = [ref for ref in file_references if ref.exists_in_files_folder]
        type_counts = Counter(ref.file_type for ref in file_references)
        ext_counts = Counter(
            ref.file_extension for ref in file_references if ref.file_extension
        )
        user_counts = Counter(ref.from_user for ref in file_references)
        chat_counts = Counter(ref.chat_name for ref in file_references)

        # Generate comprehensive markdown content
        channel_summary = ""
        meetings_summary = ""
        chat_stats_heading_suffix = ""
        if channel_stats:
            channel_users = len(channel_stats)
            total_accessible_messages = sum(
                stats.total_accessible_messages for stats in channel_stats.values()
            )
            total_accessible_replies = sum(
                stats.accessible_replies for stats in channel_stats.values()
            )
            channel_summary = f"""
### Channel Message Overview
- **Users with channel access**: {channel_users}
- **Total accessible channel messages**: {total_accessible_messages:,}
- **Total accessible channel replies**: {total_accessible_replies:,}
- **Total channel communications**: {total_accessible_messages + total_accessible_replies:,}
"""
            chat_stats_heading_suffix = " - Combined Chat & Channel Messages"
        else:
            chat_stats_heading_suffix = " - Chat Messages Only"

        markdown_content = f"""# Comprehensive Teams Analytics Report

Generated on: {self._get_current_timestamp()}

## Executive Summary

### Chat Activity Overview
- **Total users analyzed**: {len(user_stats)}
- **Total messages sent**: {sum(stat.sent_count for stat in user_stats.values()):,}
- **Total message receptions**: {sum(stat.received_count for stat in user_stats.values()):,}
- **Average messages per user**: {(sum(stat.sent_count + stat.received_count for stat in user_stats.values()) / len(user_stats)):.1f}
{channel_summary}{meetings_summary}
### File Sharing Overview
- **Total file references**: {len(file_references)}
- **Files available in data/files**: {len(matched_files)}
- **Match percentage**: {(len(matched_files)/len(file_references)*100):.1f}% 
- **Users sharing files**: {len(user_counts)}
- **Chats with file references**: {len(chat_counts)}

---

# 📊 USER MESSAGING STATISTICS

## Summary Statistics ({display_label} Users){chat_stats_heading_suffix}

| Rank | Username | **Grand Total** | Chat Sent | Chat Received | Chat Total | Channel Messages | Channel Replies | Channel Total | Group Chats | One-on-One | Meeting Chats | Total Conversations |
|------|----------|----------------|-----------|---------------|------------|------------------|-----------------|---------------|-------------|------------|---------------|-------------------|
"""

        # Create a mapping of usernames to their chat type statistics
        user_chat_stats = {
            username: user_stats_item
            for username, user_stats_item in user_stats.items()
        }

        # Add user statistics table with merged chat and channel data
        for i, (
            username,
            user_stats_item,
            chat_total,
            channel_total,
            combined_total,
        ) in enumerate(display_list, 1):
            # Get channel-specific data if available
            channel_messages = 0
            channel_replies = 0
            if channel_stats and username in channel_stats:
                channel_messages = channel_stats[username].total_accessible_messages
                channel_replies = channel_stats[username].accessible_replies

            markdown_content += f"| {i} | {username} | **{combined_total:,}** | {user_stats_item.sent_count:,} | {user_stats_item.received_count:,} | {chat_total:,} | {channel_messages:,} | {channel_replies:,} | {channel_total:,} | {len(user_stats_item.group_chats_participated)} | {len(user_stats_item.one_on_one_chats)} | {len(user_stats_item.meeting_chats)} | {user_stats_item.total_conversations} |\n"

        # Add channel statistics summary if available
        if channel_stats:
            channel_users = len(channel_stats)
            total_accessible_messages = sum(
                stats.total_accessible_messages for stats in channel_stats.values()
            )
            total_accessible_replies = sum(
                stats.accessible_replies for stats in channel_stats.values()
            )

            markdown_content += f"""

---

# 🔗 CHANNEL MESSAGE STATISTICS SUMMARY

- **Users with channel access**: {channel_users}
- **Total accessible channel messages**: {total_accessible_messages:,}
- **Total accessible channel replies**: {total_accessible_replies:,}
- **Total channel communications**: {total_accessible_messages + total_accessible_replies:,}

*Note: Individual user channel statistics are included in the User Messaging Statistics table above.*
"""

        # Add file references section
        markdown_content += f"""

---

# 📎 FILE REFERENCES ANALYSIS

## File Reference Summary

- **Total file references**: {len(file_references)}
- **Exact matches found**: {len(matched_files)}
- **Match percentage**: {len(matched_files)/len(file_references)*100:.1f}%
- **Unique file types**: {len(type_counts)}
- **Available file extensions in data/files**: {len(self.available_files)}

## File Matching Summary

| Extension | References | Exact Matches | Available Files |
|-----------|------------|---------------|----------------|
"""

        # Create extension matching table
        all_extensions = set(
            ref.file_extension for ref in file_references if ref.file_extension
        )
        all_extensions.update(self.available_files.keys())

        for ext in sorted(all_extensions):
            ref_count = len(
                [ref for ref in file_references if ref.file_extension == ext]
            )
            match_count = len(
                [
                    ref
                    for ref in file_references
                    if ref.file_extension == ext and ref.exists_in_files_folder
                ]
            )
            avail_count = len(self.available_files.get(ext, []))
            markdown_content += (
                f"| .{ext} | {ref_count} | {match_count} | {avail_count} |\n"
            )

        markdown_content += """
## File References by Type

| File Type | Count | Exact Matches |
|-----------|-------|---------------|
"""

        for file_type, count in type_counts.most_common():
            matched_count = len(
                [
                    ref
                    for ref in file_references
                    if ref.file_type == file_type and ref.exists_in_files_folder
                ]
            )
            markdown_content += f"| {file_type} | {count} | {matched_count} |\n"

        markdown_content += """
## File References by Extension

| Extension | Count |
|-----------|-------|
"""

        for ext, count in ext_counts.most_common(15):
            ext_display = f".{ext}" if ext else "(no extension)"
            markdown_content += f"| {ext_display} | {count} |\n"

        markdown_content += """
## Top Users Sharing Files

| Username | File References |
|----------|----------------|
"""

        for user, count in user_counts.most_common(15):
            markdown_content += f"| {user} | {count} |\n"

        markdown_content += """
## Top Chats with File References

| Chat Name | File References |
|-----------|----------------|
"""

        for chat_name, count in chat_counts.most_common(15):
            # Handle empty or missing chat names
            display_name = chat_name.strip() if chat_name else "(empty)"
            if not display_name:
                display_name = "(empty)"

            # Escape markdown special characters in chat name
            escaped_name = (
                display_name.replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")
            )
            markdown_content += f"| {escaped_name} | {count} |\n"

        # Add exact matches section if any found
        if matched_files:
            markdown_content += f"""
## Exact File Matches ({len(matched_files)} found)

| File Name | Extension | Referenced By | Chat | Date |
|-----------|-----------|---------------|------|------|
"""
            for ref in matched_files:  # Show all matches
                escaped_chat = (
                    ref.chat_name.replace("|", "\\|")
                    .replace("[", "\\[")
                    .replace("]", "\\]")
                )
                date_part = (
                    ref.sent_datetime.split("T")[0]
                    if "T" in ref.sent_datetime
                    else ref.sent_datetime
                )
                markdown_content += f"| {ref.file_name} | .{ref.file_extension} | {ref.from_user} | {escaped_chat} | {date_part} |\n"

        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)


def generate_comprehensive_analytics(
    chats_file: str,
    top_n: Optional[int] = None,
    output_file: Optional[str] = None,
    include_channel_messages: bool = False,
    data_folder: Optional[str] = None,
) -> None:
    """
    Generate comprehensive Teams analytics including user statistics and file references

    Now supports both regular Teams chat messages and channel messages for complete analysis.

    This function combines user messaging statistics and file reference analysis
    into a single comprehensive report. It provides:
    - User messaging statistics (sent/received counts, conversation participation)
    - Channel message accessibility analysis (when enabled)
    - File reference extraction and availability analysis
    - Export capabilities in unified Markdown format

    Args:
        chats_file: Path to chats.config.json
        top_n: Number of top users to show in rankings. If None, shows all users.
        output_file: Optional file to save comprehensive report as markdown
        include_channel_messages: Whether to include channel message analysis (default: False)
        data_folder: Path to folder with channel config files (required if include_channel_messages=True)

    Returns:
        None - Results are printed to console and optionally saved to file
    """
    analytics = MessageAnalytics(chats_file)

    # Collect user statistics from chat messages
    print("📊 Collecting chat user statistics...")
    user_stats = analytics.collect_user_statistics()

    # Include channel message analysis if requested
    channel_stats = None
    if include_channel_messages:
        if not data_folder:
            print(
                "⚠️ Warning: data_folder required for channel message analysis. Skipping channel messages."
            )
        else:
            try:
                print("🔗 Analyzing channel message accessibility...")
                channel_analyzer = ChannelMessageAccessibilityAnalyzer(data_folder)
                channel_analyzer.run_full_analysis()
                channel_stats = channel_analyzer.user_accessibility
                print(
                    f"✅ Channel analysis complete: {len(channel_stats)} users analyzed"
                )
            except Exception as e:
                print(f"❌ Error analyzing channel messages: {e}")
                channel_stats = None

    # Extract file references
    print("📎 Extracting file references...")
    file_references = analytics.extract_file_references()

    if output_file:
        # Generate comprehensive markdown report
        analytics._generate_comprehensive_markdown_report(
            user_stats, file_references, output_file, top_n, channel_stats
        )

        # Show summary on console when saving to file
        total_users = len(user_stats)
        total_sent = sum(stat.sent_count for stat in user_stats.values())
        total_received = sum(stat.received_count for stat in user_stats.values())
        total_files = len(file_references)
        exact_matches = sum(1 for fr in file_references if fr.exists_in_files_folder)

        print(f"\n{'='*70}")
        print("COMPREHENSIVE TEAMS ANALYTICS SUMMARY")
        print(f"{'='*70}")
        print(f"📊 CHAT MESSAGE STATISTICS:")
        print(f"   Total users: {total_users}")
        print(f"   Total messages sent: {total_sent:,}")
        print(f"   Total message receptions: {total_received:,}")

        # Add channel statistics if available
        if channel_stats:
            channel_users = len(channel_stats)
            total_accessible_messages = sum(
                stats.total_accessible_messages for stats in channel_stats.values()
            )
            total_accessible_replies = sum(
                stats.accessible_replies for stats in channel_stats.values()
            )
            print(f"\n🔗 CHANNEL MESSAGE STATISTICS:")
            print(f"   Users with channel access: {channel_users}")
            print(
                f"   Total accessible channel messages: {total_accessible_messages:,}"
            )
            print(f"   Total accessible channel replies: {total_accessible_replies:,}")
            print(
                f"   Total channel communications: {total_accessible_messages + total_accessible_replies:,}"
            )

        print(f"\n📎 FILE REFERENCES:")
        print(f"   Total file references: {total_files}")
        print(f"   Files available in data/files: {exact_matches}")
        print(
            f"   Match percentage: {(exact_matches/total_files*100):.1f}%"
            if total_files > 0
            else "   Match percentage: 0.0%"
        )

        logger.info(f"Comprehensive analytics saved to: {output_file}")
    else:
        # Show both statistics on console when not saving to file
        print("\n" + "=" * 70)
        print("USER STATISTICS")
        print("=" * 70)
        analytics.print_user_statistics(top_n)

        print("\n" + "=" * 70)
        print("FILE REFERENCES")
        print("=" * 70)
        analytics.print_file_statistics(file_references)

    return None


def extract_user_teams_messages_with_files_and_prompt(
    chats_file: str,
    user_email: str,
    output_folder: str = "results",
    prompt_file: str = "prompt_teams_gen_complex_utterance.md",
    files_folder: str = "data/files",
    include_channel_messages: bool = False,
    data_folder: Optional[str] = None,
) -> str:
    """Extract Teams messages for a user, match file references to actual files, and generate prompt
    
    Now supports both regular Teams chat messages and channel messages with comprehensive 
    file content extraction from both sources. Meeting-type chat messages are EXCLUDED 
    from LLM prompt generation to maintain separation between daily collaboration (chats/channels)
    and formal meeting contexts (handled by meetings_events_analytics.py).

    Args:
        chats_file: Path to the chats.config.json file (for Teams messages)
        user_email: Specific user email to analyze (required)
        output_folder: Folder to save the output files (default: "results")
        prompt_file: Prompt template file name (default: "prompt_teams_gen_complex_utterance.md")
        files_folder: Folder containing actual files (default: "data/files")
        include_channel_messages: Whether to include channel messages accessible to user (default: False)
        data_folder: Path to folder with channel config files (required if include_channel_messages=True)

    Returns:
        str: Human-readable summary of the processing results
        
    Note:
        - Meeting-type chats are excluded from prompt generation but still counted in statistics
        - For meeting-focused complex query generation, use meetings_events_analytics.py instead
        - This function focuses on daily collaboration patterns (chats, channels, informal discussions)

    Example:
        # With channel messages (excludes meeting chats from prompt)
        python teams_message_analytics.py extract_user_teams_messages_with_files_and_prompt \
            --chats_file=data/chats.config.json \
            --user_email=alex.khan \
            --output_folder=results \
            --prompt_file=prompt_teams_gen_complex_utterance.md \
            --files_folder=data/files \
            --include_channel_messages=true \
            --data_folder=data
    """
    logger.info(f"🚀 Starting Teams messages + files extraction for: {user_email}")
    logger.info(f"💬 Chats config: {chats_file}")
    logger.info(f"📂 Output folder: {output_folder}")
    logger.info(f"📝 Prompt template: {prompt_file}")
    logger.info(f"📁 Files folder: {files_folder}")
    if include_channel_messages:
        logger.info(f"🔗 Channel messages: ENABLED (data folder: {data_folder})")
    else:
        logger.info(f"🔗 Channel messages: DISABLED")

    # Ensure output directory exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Initialize file access analyzer
    try:
        file_reader = UniversalFileReader(silent=True)
        logger.info("✅ File reader initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize file reader: {e}")
        raise

    # Get list of available files from files folder
    logger.info(f"🔍 Scanning files folder: {files_folder}")
    try:
        available_files = []
        if os.path.exists(files_folder):
            for file_name in os.listdir(files_folder):
                file_path = os.path.join(files_folder, file_name)
                if os.path.isfile(file_path):
                    available_files.append(
                        {"file_name": file_name, "file_path": file_path}
                    )

        logger.info(f"📁 Found {len(available_files)} files in {files_folder}")
    except Exception as e:
        logger.error(f"❌ Error scanning files folder: {e}")
        available_files = []

    # Get user's Teams messages (exclude meeting-type chats from prompt generation)
    logger.info(f"💬 Extracting Teams chat messages for user: {user_email}")
    try:
        teams_analytics = MessageAnalytics(chats_file)
        all_user_messages = teams_analytics.get_user_accessible_messages(user_email)

        # Filter out meeting-type chats for LLM prompt generation
        # (statistics will still include all chat types)
        user_messages = [msg for msg in all_user_messages if msg.chat_type != "Meeting"]
        meeting_messages_filtered = len(all_user_messages) - len(user_messages)

        logger.info(
            f"📬 Found {len(all_user_messages)} total Teams chat messages for {user_email}"
        )
        logger.info(
            f"🔄 Filtered out {meeting_messages_filtered} meeting-type chat messages for LLM prompt"
        )
        logger.info(
            f"📝 Using {len(user_messages)} non-meeting chat messages for complex query generation"
        )
    except Exception as e:
        logger.error(f"❌ Error extracting Teams chat messages: {e}")
        user_messages = []

    # Get user's channel messages if requested
    channel_messages = []
    channel_file_references = []
    if include_channel_messages:
        if not data_folder:
            logger.warning(
                "⚠️ data_folder required for channel message analysis. Skipping channel messages."
            )
        else:
            logger.info(f"🔗 Extracting channel messages for user: {user_email}")
            try:
                channel_analyzer = ChannelMessageAccessibilityAnalyzer(data_folder)
                channel_analyzer.run_full_analysis()

                if user_email in channel_analyzer.user_accessibility:
                    # Get channel messages accessible to this user
                    for channel in channel_analyzer.channels_data:
                        channel_id = channel.get("ChannelId")
                        channel_name = channel.get("Name", "")
                        members = channel.get("Members", [])

                        if user_email in members:  # User has access to this channel
                            # Get messages from this channel
                            for message in channel_analyzer.messages_data:
                                if message.get("ChannelId") == channel_id:
                                    channel_msg = ChatMessage(
                                        chat_id=channel_id or "",
                                        chat_name=f"Channel: {channel_name}",
                                        chat_type="channel",
                                        message_id=message.get("MessageId", ""),
                                        from_user=message.get("FromUser", ""),
                                        content=message.get("Content", ""),
                                        content_type=message.get("ContentType", "text"),
                                        sent_datetime=message.get("SentDateTime", ""),
                                        members=members,
                                    )
                                    channel_messages.append(channel_msg)

                            # Also get replies from this channel
                            for reply in channel_analyzer.replies_data:
                                if reply.get("ChannelId") == channel_id:
                                    reply_msg = ChatMessage(
                                        chat_id=channel_id or "",
                                        chat_name=f"Channel: {channel_name}",
                                        chat_type="channel_reply",
                                        message_id=reply.get("ReplyId", ""),
                                        from_user=reply.get("FromUser", ""),
                                        content=reply.get("Content", ""),
                                        content_type=reply.get("ContentType", "text"),
                                        sent_datetime=reply.get("SentDateTime", ""),
                                        members=members,
                                    )
                                    channel_messages.append(reply_msg)

                logger.info(
                    f"🔗 Found {len(channel_messages)} accessible channel messages"
                )

                # Note: File reference extraction from channel messages will be enhanced in future versions
                logger.info(
                    f"📎 Channel file references: Feature available but not implemented in this version"
                )

            except Exception as e:
                logger.error(f"❌ Error extracting channel messages: {e}")
                channel_messages = []
                channel_file_references = []

    # Extract file references from Teams messages
    logger.info(f"📎 Extracting file references from Teams messages...")
    try:
        all_file_references = teams_analytics.extract_file_references()

        # Filter file references to only those from the user
        user_file_references = [
            ref for ref in all_file_references if ref.from_user == user_email
        ]

        logger.info(
            f"📎 Found {len(user_file_references)} file references in {user_email}'s messages"
        )
    except Exception as e:
        logger.error(f"❌ Error extracting file references: {e}")
        user_file_references = []

    # Process Teams messages and match with files
    processed_messages = []
    matched_files_content = []
    unmatched_file_refs = []

    # Combine all messages (chat + channel)
    all_user_messages = user_messages + channel_messages
    logger.info(
        f"📝 Processing {len(all_user_messages)} total messages ({len(user_messages)} chat + {len(channel_messages)} channel)..."
    )

    for msg in all_user_messages:
        message_dict = {
            "chat_id": msg.chat_id,
            "chat_name": msg.chat_name,
            "chat_type": msg.chat_type,
            "message_id": msg.message_id,
            "from_user": msg.from_user,
            "content": msg.content,
            "content_type": msg.content_type,
            "sent_datetime": msg.sent_datetime,
            "members": msg.members,
            "message_source": (
                "channel" if msg.chat_type.startswith("channel") else "chat"
            ),
        }
        processed_messages.append(message_dict)

    # Try to match file references to actual accessible files
    logger.info(
        f"🔗 Matching {len(user_file_references)} file references to accessible files..."
    )
    matched_count = 0

    for file_ref in user_file_references:
        matched = False

        # Try to match by filename
        ref_filename = file_ref.file_name.lower()

        for available_file in available_files:
            available_filename = available_file["file_name"].lower()

            # Check for exact filename match or partial match
            if (
                ref_filename == available_filename
                or (ref_filename in available_filename and len(ref_filename) > 3)
                or (available_filename in ref_filename and len(available_filename) > 3)
            ):

                # Found a match, try to read the file content
                try:
                    file_path = available_file["file_path"]

                    if os.path.exists(file_path):
                        # Read file content
                        read_result = file_reader.read_file(file_path)

                        if read_result["status"] == "success":
                            matched_file_info = {
                                "referenced_in_message": {
                                    "message_id": file_ref.message_id,
                                    "chat_name": file_ref.chat_name,
                                    "sent_datetime": file_ref.sent_datetime,
                                    "reference_text": file_ref.reference,
                                    "context": file_ref.context,
                                },
                                "file_details": {
                                    "file_name": available_file["file_name"],
                                    "file_location": file_path,
                                    "permission_level": "accessible",  # Assuming accessible since in Teams chat
                                    "content": read_result["content"],
                                    "content_length": len(read_result["content"]),
                                },
                            }

                            # Add format-specific metadata if available
                            for key in [
                                "total_pages",
                                "total_sheets",
                                "slide_count",
                                "format_info",
                            ]:
                                if key in read_result:
                                    matched_file_info["file_details"][key] = (
                                        read_result[key]
                                    )

                            matched_files_content.append(matched_file_info)
                            matched_count += 1
                            matched = True
                            break
                        else:
                            logger.debug(
                                f"⚠️ Could not read file content: {read_result.get('error', 'Unknown error')}"
                            )
                    else:
                        logger.debug(f"⚠️ File does not exist: {file_path}")
                except Exception as e:
                    logger.warning(
                        f"⚠️ Error reading matched file {available_file['file_name']}: {e}"
                    )

        if not matched:
            unmatched_ref = {
                "message_id": file_ref.message_id,
                "chat_name": file_ref.chat_name,
                "sent_datetime": file_ref.sent_datetime,
                "reference_text": file_ref.reference,
                "file_name": file_ref.file_name,
                "context": file_ref.context,
            }
            unmatched_file_refs.append(unmatched_ref)

    logger.info(f"✅ Successfully matched {matched_count} files")
    logger.info(f"❌ {len(unmatched_file_refs)} file references could not be matched")

    # Create comprehensive report
    teams_files_report = {
        "overall_statistics": {
            "target_user": user_email,
            "total_messages": len(processed_messages),
            "chat_messages": len(user_messages),
            "channel_messages": len(channel_messages),
            "include_channel_messages": include_channel_messages,
            "total_file_references": len(user_file_references),
            "matched_files": len(matched_files_content),
        },
        "teams_messages": processed_messages,
        "accessible_files_and_their_full_content": matched_files_content,
    }

    # Save comprehensive JSON file
    clean_email = user_email.replace("@", "_at_").replace(".", "_")
    teams_content_filename = f"{clean_email}_teams_messages_with_files.json"
    teams_content_path = os.path.join(output_folder, teams_content_filename)

    logger.info(f"💾 Saving comprehensive report to: {teams_content_path}")
    with open(teams_content_path, "w", encoding="utf-8") as f:
        json.dump(teams_files_report, f, indent=2, ensure_ascii=False)

    # Generate and save prompt file
    prompt_file_path = _save_teams_content_prompt_file(
        teams_files_report, teams_content_path, prompt_file, output_folder, user_email
    )

    # Print summary
    _print_teams_content_summary(
        teams_files_report, teams_content_path, prompt_file_path
    )

    # Return a human-readable completion message
    channel_info = (
        f"\n🔗 Channel messages: {len(channel_messages)} included"
        if include_channel_messages
        else "\n🔗 Channel messages: Not included"
    )

    return f"""✅ Extraction completed successfully!
📧 User: {user_email}
💬 Chat messages: {len(user_messages)}
{channel_info}
📝 Total messages processed: {len(processed_messages)}
📎 Files matched: {len(matched_files_content)}
📄 Output files created:
   - Raw data: {teams_content_path}
   - LLM prompt: {prompt_file_path}"""


def _save_teams_content_prompt_file(
    report: Dict[str, Any],
    content_file: str,
    prompt_file: str,
    output_folder: str,
    user_email: str,
) -> str:
    """Save the complete Teams messages + files prompt file with JSON data inserted

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

    logger.info(f"📖 Reading Teams prompt template: {prompt_template_path}")
    try:
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        logger.info(
            f"✅ Successfully read Teams prompt template ({len(prompt_template)} characters)"
        )
    except FileNotFoundError:
        error_msg = (
            f"❌ ERROR: Teams prompt template file not found at {prompt_template_path}"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"❌ ERROR: Failed to read Teams prompt template at {prompt_template_path}: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)

    # Convert JSON to formatted string
    json_str = json.dumps(report, indent=2, ensure_ascii=False)

    # Replace placeholders in template
    complete_prompt = prompt_template.replace(
        "{teams_messages_files_json_data}", json_str
    )
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

    logger.info(f"📝 Teams prompt file saved to: {prompt_file_path}")
    return prompt_file_path


def _print_teams_content_summary(
    report: Dict[str, Any], content_file: str, prompt_file_path: str
) -> None:
    """Print a summary of the Teams messages + files extraction

    Args:
        report: Final report
        content_file: Path to the content file
        prompt_file_path: Path to the generated prompt file
    """
    stats = report["overall_statistics"]

    logger.info(f"🎉 Teams Messages + Files Extraction Complete!")
    logger.info(f"=" * 70)
    logger.info(f"👤 Target User: {stats['target_user']}")
    logger.info(f"💾 Content File: {content_file}")
    logger.info(f"📝 Prompt File: {prompt_file_path}")
    logger.info(f"💬 Total Messages: {stats['total_messages']}")
    logger.info(f"📎 Total File References: {stats['total_file_references']}")
    logger.info(f"✅ Matched Files with Content: {stats['matched_files']}")

    # Calculate total content length for matched files
    total_content_length = sum(
        file_data["file_details"]["content_length"]
        for file_data in report["accessible_files_and_their_full_content"]
    )
    logger.info(f"📝 Total File Content Length: {total_content_length:,} characters")

    # Show chat types
    chat_types = {}
    for message in report["teams_messages"]:
        chat_type = message.get("chat_type", "unknown")
        chat_types[chat_type] = chat_types.get(chat_type, 0) + 1

    if chat_types:
        logger.info(f"💬 Message Distribution by Chat Type:")
        for chat_type, count in sorted(chat_types.items()):
            logger.info(f"   {chat_type}: {count} messages")

    # Show matched file types
    if report["accessible_files_and_their_full_content"]:
        file_extensions = {}
        for file_data in report["accessible_files_and_their_full_content"]:
            file_name = file_data["file_details"].get("file_name", "")
            ext = Path(file_name).suffix.lower() or "no_extension"
            file_extensions[ext] = file_extensions.get(ext, 0) + 1

        logger.info(f"📋 Matched File Types:")
        for ext, count in sorted(file_extensions.items()):
            ext_display = ext if ext != "no_extension" else "(no extension)"
            logger.info(f"   {ext_display}: {count} files")


# CHANNEL STATISTICS MODULE
# ============================================


@dataclass
class ChannelStats:
    """Statistics for a single channel"""

    channel_id: str
    name: str
    description: str
    team_id: str
    member_count: int
    members: List[str]
    creator: str
    is_private: bool
    created_datetime: str
    message_count: int = 0
    reply_count: int = 0
    total_activity: int = 0
    most_active_user: str = ""
    activity_by_user: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.activity_by_user is None:
            self.activity_by_user = {}


@dataclass
class ChannelUserStats:
    """Statistics for a single user in channel analysis"""

    username: str
    channels_member_of: List[str]
    channels_created: List[str]
    total_messages_sent: int = 0
    total_replies_sent: int = 0
    total_activity: int = 0
    most_active_channel: str = ""
    activity_by_channel: Optional[Dict[str, int]] = None
    activity_by_date: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.activity_by_channel is None:
            self.activity_by_channel = {}
        if self.activity_by_date is None:
            self.activity_by_date = {}


@dataclass
class OverallChannelStats:
    """Overall statistics across all channel data"""

    total_channels: int
    total_messages: int
    total_replies: int
    total_users: int
    total_teams: int
    private_channels: int
    public_channels: int
    average_members_per_channel: float
    most_active_channel: str
    most_active_user: str
    activity_by_date: Dict[str, int]
    activity_by_hour: Dict[int, int]
    message_length_stats: Dict[str, float]


@dataclass
class UserMessageAccessibility:
    """Message accessibility statistics for a user account"""

    username: str
    accessible_channels: List[str] = field(default_factory=list)
    total_accessible_messages: int = 0
    messages_sent_by_user: int = 0
    messages_received_by_user: int = 0  # Messages from others in accessible channels
    accessible_replies: int = 0
    replies_sent_by_user: int = 0
    replies_received_by_user: int = 0
    channel_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.channel_breakdown:
            self.channel_breakdown = {}


class ChannelMessageAccessibilityAnalyzer:
    """Analyzer for determining message accessibility per user account based on channel membership"""

    def __init__(self, data_folder: str = "data"):
        """Initialize with path to data folder containing the config files"""
        self.data_folder = Path(data_folder)
        self.channels_file = self.data_folder / "channels.config.json"
        self.messages_file = self.data_folder / "channelmessages.config.json"
        self.replies_file = self.data_folder / "channelmessagereplies.config.json"

        # Data storage
        self.channels_data: List[Dict] = []
        self.messages_data: List[Dict] = []
        self.replies_data: List[Dict] = []

        # Channel membership mapping: channel_id -> list of members
        self.channel_membership: Dict[str, List[str]] = {}

        # User accessibility statistics
        self.user_accessibility: Dict[str, UserMessageAccessibility] = {}

    def load_data(self) -> None:
        """Load data from all three configuration files"""
        print("🔄 Loading configuration files...")

        # Load channels
        if self.channels_file.exists():
            with open(self.channels_file, "r", encoding="utf-8") as f:
                self.channels_data = json.load(f)
            print(f"✅ Loaded {len(self.channels_data)} channels")
        else:
            print(f"⚠️ Channels file not found: {self.channels_file}")

        # Load messages
        if self.messages_file.exists():
            with open(self.messages_file, "r", encoding="utf-8") as f:
                self.messages_data = json.load(f)
            print(f"✅ Loaded {len(self.messages_data)} messages")
        else:
            print(f"⚠️ Messages file not found: {self.messages_file}")

        # Load replies
        if self.replies_file.exists():
            with open(self.replies_file, "r", encoding="utf-8") as f:
                self.replies_data = json.load(f)
            print(f"✅ Loaded {len(self.replies_data)} replies")
        else:
            print(f"⚠️ Replies file not found: {self.replies_file}")

    def analyze_channel_membership(self) -> None:
        """Build channel membership mapping"""
        print("📊 Analyzing channel membership...")

        for channel in self.channels_data:
            channel_id = channel.get("ChannelId")
            members = channel.get("Members", [])

            if channel_id:
                self.channel_membership[channel_id] = members

        print(f"✅ Mapped membership for {len(self.channel_membership)} channels")

    def analyze_message_accessibility(self) -> None:
        """Analyze message accessibility per user based on channel membership"""
        print("💬 Analyzing message accessibility per user...")

        # Initialize user accessibility stats
        all_users = set()
        for members in self.channel_membership.values():
            all_users.update(members)

        for username in all_users:
            self.user_accessibility[username] = UserMessageAccessibility(
                username=username
            )

        # Find accessible channels per user
        for channel_id, members in self.channel_membership.items():
            for member in members:
                if member in self.user_accessibility:
                    self.user_accessibility[member].accessible_channels.append(
                        channel_id
                    )
                    self.user_accessibility[member].channel_breakdown[channel_id] = {
                        "messages_in_channel": 0,
                        "messages_sent": 0,
                        "messages_received": 0,
                        "replies_in_channel": 0,
                        "replies_sent": 0,
                        "replies_received": 0,
                    }

        # Count accessible messages
        for message in self.messages_data:
            channel_id = message.get("ChannelId")
            from_user = message.get("From", "")

            if channel_id in self.channel_membership:
                channel_members = self.channel_membership[channel_id]

                # Each channel member can access this message
                for member in channel_members:
                    if member in self.user_accessibility:
                        self.user_accessibility[member].total_accessible_messages += 1
                        self.user_accessibility[member].channel_breakdown[channel_id][
                            "messages_in_channel"
                        ] += 1

                        if member == from_user:
                            # Message sent by this user
                            self.user_accessibility[member].messages_sent_by_user += 1
                            self.user_accessibility[member].channel_breakdown[
                                channel_id
                            ]["messages_sent"] += 1
                        else:
                            # Message received by this user (from someone else)
                            self.user_accessibility[
                                member
                            ].messages_received_by_user += 1
                            self.user_accessibility[member].channel_breakdown[
                                channel_id
                            ]["messages_received"] += 1

        # Count accessible replies
        for reply in self.replies_data:
            # Find the original message to get channel_id
            channel_message_id = reply.get("ChannelMessageId")
            from_user = reply.get("From", "")

            # Find channel for this reply by matching message ID
            channel_id = None
            for message in self.messages_data:
                if message.get("ChannelMessageId") == channel_message_id:
                    channel_id = message.get("ChannelId")
                    break

            if channel_id and channel_id in self.channel_membership:
                channel_members = self.channel_membership[channel_id]

                # Each channel member can access this reply
                for member in channel_members:
                    if member in self.user_accessibility:
                        self.user_accessibility[member].accessible_replies += 1
                        self.user_accessibility[member].channel_breakdown[channel_id][
                            "replies_in_channel"
                        ] += 1

                        if member == from_user:
                            # Reply sent by this user
                            self.user_accessibility[member].replies_sent_by_user += 1
                            self.user_accessibility[member].channel_breakdown[
                                channel_id
                            ]["replies_sent"] += 1
                        else:
                            # Reply received by this user (from someone else)
                            self.user_accessibility[
                                member
                            ].replies_received_by_user += 1
                            self.user_accessibility[member].channel_breakdown[
                                channel_id
                            ]["replies_received"] += 1

        print(f"✅ Analyzed accessibility for {len(self.user_accessibility)} users")

    def run_full_analysis(self) -> None:
        """Run complete message accessibility analysis"""
        print("🚀 Starting message accessibility analysis...\n")

        self.load_data()
        self.analyze_channel_membership()
        self.analyze_message_accessibility()

        print("\n✅ Analysis complete!")

    def print_user_message_accessibility(self, limit: int = 10) -> None:
        """Print top users by message accessibility"""
        if not self.user_accessibility:
            print("❌ No accessibility data available. Run analysis first.")
            return

        print(f"\n{'='*80}")
        print("� TOP USERS BY MESSAGE ACCESSIBILITY")
        print(f"{'='*80}")

        # Sort users by total accessible messages
        sorted_users = sorted(
            self.user_accessibility.items(),
            key=lambda x: x[1].total_accessible_messages,
            reverse=True,
        )

        print(f"\nTop {limit} Users by Total Accessible Messages:")
        print("-" * 95)
        print(
            f"{'Rank':<4} {'Username':<20} {'Accessible':<11} {'Sent':<6} {'Received':<8} {'Replies':<7} {'Channels':<8}"
        )
        print("-" * 95)

        for i, (username, stats) in enumerate(sorted_users[:limit], 1):
            total_messages = stats.total_accessible_messages + stats.accessible_replies
            print(
                f"{i:<4} {username:<20} {total_messages:<11} {stats.messages_sent_by_user:<6} "
                f"{stats.messages_received_by_user:<8} {stats.accessible_replies:<7} {len(stats.accessible_channels):<8}"
            )

    def print_channel_accessibility_summary(self, limit: int = 10) -> None:
        """Print summary of channel accessibility"""
        print(f"\n{'='*80}")
        print("📺 CHANNEL ACCESSIBILITY SUMMARY")
        print(f"{'='*80}")

        # Count messages per channel
        channel_message_counts = {}
        for message in self.messages_data:
            channel_id = message.get("ChannelId")
            if channel_id:
                channel_message_counts[channel_id] = (
                    channel_message_counts.get(channel_id, 0) + 1
                )

        # Get channel info and sort by message count
        channel_info = []
        for channel in self.channels_data:
            channel_id = channel.get("ChannelId")
            if channel_id:
                message_count = channel_message_counts.get(channel_id, 0)
                member_count = len(channel.get("Members", []))
                channel_info.append(
                    {
                        "id": channel_id,
                        "name": channel.get("Name", ""),
                        "members": member_count,
                        "messages": message_count,
                        "accessibility_score": message_count
                        * member_count,  # Messages × Members
                    }
                )

        channel_info.sort(key=lambda x: x["accessibility_score"], reverse=True)

        print(f"\nTop {limit} Channels by Accessibility Impact (Messages × Members):")
        print("-" * 85)
        print(
            f"{'Rank':<4} {'Channel Name':<35} {'Members':<8} {'Messages':<9} {'Impact':<8}"
        )
        print("-" * 85)

        for i, channel in enumerate(channel_info[:limit], 1):
            print(
                f"{i:<4} {channel['name'][:34]:<35} {channel['members']:<8} "
                f"{channel['messages']:<9} {channel['accessibility_score']:<8}"
            )

    def export_accessibility_statistics_markdown(self, output_file: str) -> None:
        """Export message accessibility statistics to Markdown"""
        if not self.user_accessibility:
            print("❌ No accessibility data available. Run analysis first.")
            return

        # Sort users by total accessible messages
        sorted_users = sorted(
            self.user_accessibility.items(),
            key=lambda x: x[1].total_accessible_messages + x[1].accessible_replies,
            reverse=True,
        )

        # Sort channels by accessibility impact
        channel_message_counts = {}
        for message in self.messages_data:
            channel_id = message.get("ChannelId")
            if channel_id:
                channel_message_counts[channel_id] = (
                    channel_message_counts.get(channel_id, 0) + 1
                )

        channel_info = []
        for channel in self.channels_data:
            channel_id = channel.get("ChannelId")
            if channel_id:
                message_count = channel_message_counts.get(channel_id, 0)
                member_count = len(channel.get("Members", []))
                channel_info.append(
                    {
                        "id": channel_id,
                        "name": channel.get("Name", ""),
                        "members": member_count,
                        "messages": message_count,
                        "accessibility_score": message_count * member_count,
                    }
                )

        channel_info.sort(key=lambda x: x["accessibility_score"], reverse=True)

        # Generate markdown content
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        markdown_content = f"""# Teams Message Accessibility Report

Generated on: {timestamp}

## Executive Summary

- **Total Users Analyzed**: {len(self.user_accessibility)}
- **Total Channels**: {len(self.channel_membership)}
- **Total Messages**: {len(self.messages_data)}
- **Total Replies**: {len(self.replies_data)}
- **Total Communications**: {len(self.messages_data) + len(self.replies_data)}

## Top Users by Message Accessibility

Users ranked by total accessible content (messages + replies) for complex query generation:

| Rank | Username | Total Accessible | Messages Sent | Messages Received | Replies | Accessible Channels |
|------|----------|-----------------|---------------|-------------------|---------|-------------------|
"""

        for i, (username, stats) in enumerate(sorted_users, 1):
            total_accessible = (
                stats.total_accessible_messages + stats.accessible_replies
            )
            markdown_content += f"| {i} | {username} | {total_accessible} | {stats.messages_sent_by_user} | {stats.messages_received_by_user} | {stats.accessible_replies} | {len(stats.accessible_channels)} |\n"

        markdown_content += f"""

## Channel Accessibility Impact

Channels ranked by accessibility impact (Messages × Members) - higher scores indicate more queryable content:

| Rank | Channel Name | Members | Messages | Impact Score |
|------|-------------|---------|----------|--------------|
"""

        for i, channel in enumerate(channel_info, 1):
            markdown_content += f"| {i} | {channel['name']} | {channel['members']} | {channel['messages']} | {channel['accessibility_score']} |\n"

        markdown_content += f"""

## Detailed User Accessibility Breakdown

### Top 10 Users - Channel-by-Channel Analysis

"""

        # Add detailed breakdown for top 10 users
        for i, (username, stats) in enumerate(sorted_users[:10], 1):
            total_accessible = (
                stats.total_accessible_messages + stats.accessible_replies
            )
            markdown_content += f"""
#### {i}. {username}
- **Total Accessible Communications**: {total_accessible}
- **Messages**: {stats.messages_sent_by_user} sent, {stats.messages_received_by_user} received  
- **Replies**: {stats.replies_sent_by_user} sent, {stats.replies_received_by_user} received
- **Accessible Channels**: {len(stats.accessible_channels)}

**Channel Breakdown:**

| Channel ID | Messages in Channel | Messages Sent | Messages Received | Replies | Replies Sent | Replies Received |
|-----------|-------------------|---------------|-------------------|---------|--------------|------------------|
"""

            for channel_id, breakdown in stats.channel_breakdown.items():
                # Find channel name
                channel_name = channel_id
                for channel in self.channels_data:
                    if channel.get("ChannelId") == channel_id:
                        channel_name = channel.get("Name", channel_id)
                        break

                markdown_content += f"| {channel_name} | {breakdown['messages_in_channel']} | {breakdown['messages_sent']} | {breakdown['messages_received']} | {breakdown['replies_in_channel']} | {breakdown['replies_sent']} | {breakdown['replies_received']} |\n"

        markdown_content += f"""

## Query Generation Recommendations

### Optimal Target Users
Based on message accessibility analysis:

1. **High-Content Users** (50+ accessible communications):
"""
        high_content_users = [
            user
            for user, stats in sorted_users
            if (stats.total_accessible_messages + stats.accessible_replies) >= 50
        ]

        for username, stats in sorted_users:
            if (stats.total_accessible_messages + stats.accessible_replies) >= 50:
                markdown_content += f"   - **{username}**: {stats.total_accessible_messages + stats.accessible_replies} total accessible communications\n"

        markdown_content += f"""
2. **Medium-Content Users** (20-49 accessible communications):
"""
        for username, stats in sorted_users:
            total = stats.total_accessible_messages + stats.accessible_replies
            if 20 <= total < 50:
                markdown_content += (
                    f"   - **{username}**: {total} total accessible communications\n"
                )

        markdown_content += f"""

### High-Impact Channels for Query Context
Channels with the highest accessibility impact for generating complex queries:

"""
        for i, channel in enumerate(channel_info[:5], 1):
            markdown_content += f"{i}. **{channel['name']}**: {channel['accessibility_score']} impact score ({channel['members']} members × {channel['messages']} messages)\n"

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"✅ Message accessibility report saved to: {output_file}")


# ============================================
# MAIN FUNCTION REGISTRY
# ============================================

if __name__ == "__main__":
    # Create a dictionary of available functions
    functions = {
        "generate_comprehensive_analytics": generate_comprehensive_analytics,
        "extract_user_teams_messages_with_files_and_prompt": extract_user_teams_messages_with_files_and_prompt,
    }

    fire.Fire(functions)
