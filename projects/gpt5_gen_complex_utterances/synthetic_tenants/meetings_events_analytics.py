#!/usr/bin/env python3
"""
Meetings & Events Analytics Tool

A specialized analysis tool for Microsoft Teams meetings, calendar events, and transcripts,
designed to generate complex queries for LLM training focused on formal meeting discussions
and structured collaboration patterns.

Features:
- Meeting and event statistics analysis
- Transcript content extraction and speaker analysis
- Meeting-type chat message correlation with formal events
- Export capabilities in Markdown format optimized for LLM training
- Complex query generation for formal meeting contexts

Main Functions:
1. generate_meetings_events_analytics: Generate comprehensive meetings/events analysis
2. extract_meetings_for_complex_queries: Extract meeting content for LLM prompt generation

Usage Examples:
    # Generate comprehensive meetings/events analytics report
    python meetings_events_analytics.py generate_meetings_events_analytics --data_folder=data --output_file="results/meetings_analytics.md"

    # Extract meeting content for complex query generation
    python meetings_events_analytics.py extract_meetings_for_complex_queries --data_folder=data --participant_filter="alex.khan"

Input Requirements:
- onlinemeetings.config.json: Meeting data with participants and transcript references
- events.config.json: Calendar events with subjects, attendees, and attachments
- transcripts/*.vtt: VTT transcript files with speaker attribution
- chats.config.json: Chat data to correlate meeting-type conversations
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
class MeetingStats:
    """Statistics for online meetings"""

    meeting_id: str
    meeting_type: str
    event_id: str = ""
    start_datetime: str = ""
    end_datetime: str = ""
    owner: str = ""
    participants: List[str] = field(default_factory=list)
    transcript_available: bool = False
    transcript_speakers: Set[str] = field(default_factory=set)
    transcript_duration_seconds: int = 0
    transcript_utterances: int = 0


@dataclass
class EventStats:
    """Statistics for calendar events"""

    event_id: str
    sender: str
    start_datetime: str
    end_datetime: str
    subject: str = ""
    body: str = ""
    required_attendees: List[str] = field(default_factory=list)
    optional_attendees: List[str] = field(default_factory=list)
    is_online_meeting: bool = False
    has_attachments: bool = False
    attachment_count: int = 0
    attachments: List[str] = field(default_factory=list)


@dataclass
class MeetingChatCorrelation:
    """Correlation between meetings and chat messages"""

    meeting_id: str
    event_id: str
    chat_id: str
    chat_messages: List[Dict] = field(default_factory=list)
    message_count: int = 0
    participants_overlap: Set[str] = field(default_factory=set)


@dataclass
class MeetingsEventsAnalytics:
    """Container for meetings and events analytics"""

    total_meetings: int = 0
    total_events: int = 0
    total_transcripts: int = 0
    meetings_with_transcripts: int = 0
    total_meeting_chats: int = 0
    unique_meeting_participants: Set[str] = field(default_factory=set)
    unique_event_attendees: Set[str] = field(default_factory=set)
    meeting_stats: List[MeetingStats] = field(default_factory=list)
    event_stats: List[EventStats] = field(default_factory=list)
    meeting_chat_correlations: List[MeetingChatCorrelation] = field(
        default_factory=list
    )
    participant_meeting_counts: Dict[str, int] = field(default_factory=dict)
    participant_event_counts: Dict[str, int] = field(default_factory=dict)
    transcript_speaker_counts: Dict[str, int] = field(default_factory=dict)
    # File statistics
    total_file_references: int = 0
    unique_file_references: Set[str] = field(default_factory=set)
    files_found_in_folder: int = 0
    files_in_folder: Set[str] = field(default_factory=set)


class MeetingsEventsAnalyzer:
    """Analyzer for meetings, events, transcripts, and meeting-type chats"""

    def __init__(self, data_folder: str):
        """
        Initialize the meetings and events analyzer

        Args:
            data_folder: Path to folder containing config files and transcripts
        """
        self.data_folder = Path(data_folder)
        self.meetings_file = self.data_folder / "onlinemeetings.config.json"
        self.events_file = self.data_folder / "events.config.json"
        self.chats_file = self.data_folder / "chats.config.json"
        self.transcripts_folder = self.data_folder / "transcripts"
        self.files_folder = self.data_folder / "files"

        self.meetings_data = []
        self.events_data = []
        self.chats_data = []
        self.transcript_files = []
        self.analytics = MeetingsEventsAnalytics()

    def load_data(self) -> None:
        """Load meetings, events, chats, and transcripts data"""
        print("📅 Loading meetings, events, and chat data...")

        # Load meetings data
        if self.meetings_file.exists():
            try:
                with open(self.meetings_file, "r", encoding="utf-8") as f:
                    self.meetings_data = json.load(f)
                print(
                    f"✅ Loaded {len(self.meetings_data)} meetings from {self.meetings_file.name}"
                )
            except Exception as e:
                print(f"❌ Error loading meetings file: {e}")

        # Load events data
        if self.events_file.exists():
            try:
                with open(self.events_file, "r", encoding="utf-8") as f:
                    self.events_data = json.load(f)
                print(
                    f"✅ Loaded {len(self.events_data)} events from {self.events_file.name}"
                )
            except Exception as e:
                print(f"❌ Error loading events file: {e}")

        # Load chats data (to correlate meeting-type chats)
        if self.chats_file.exists():
            try:
                with open(self.chats_file, "r", encoding="utf-8") as f:
                    self.chats_data = json.load(f)
                print(
                    f"✅ Loaded {len(self.chats_data)} chats from {self.chats_file.name}"
                )
            except Exception as e:
                print(f"❌ Error loading chats file: {e}")

        # Scan transcripts folder
        if self.transcripts_folder.exists():
            self.transcript_files = list(
                self.transcripts_folder.glob("transcript-*.vtt")
            )
            print(f"✅ Found {len(self.transcript_files)} transcript files")
        else:
            print(f"⚠️ Transcripts folder not found: {self.transcripts_folder}")

    def parse_transcript(self, transcript_path: Path) -> Dict:
        """
        Parse a VTT transcript file and extract statistics

        Args:
            transcript_path: Path to VTT transcript file

        Returns:
            Dictionary with transcript statistics and content
        """
        speakers = set()
        utterances = 0
        duration_seconds = 0
        transcript_content = []

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract meeting ID from filename
            meeting_id = transcript_path.stem.replace("transcript-", "")

            # Parse VTT content
            lines = content.split("\n")
            current_speaker = ""
            current_text = ""
            current_timestamp = ""

            for i, line in enumerate(lines):
                # Look for time stamps
                if "-->" in line:
                    current_timestamp = line.strip()
                # Look for speaker tags: <v Speaker Name>
                elif line.startswith("<v ") and ">" in line:
                    speaker_end = line.find(">")
                    if speaker_end > 3:
                        current_speaker = line[3:speaker_end].strip()
                        current_text = (
                            line[speaker_end + 1 :].replace("</v>", "").strip()
                        )
                        speakers.add(current_speaker)
                        utterances += 1

                        # Store transcript content for query generation
                        transcript_content.append(
                            {
                                "timestamp": current_timestamp,
                                "speaker": current_speaker,
                                "text": current_text,
                            }
                        )

                        # Calculate duration from timestamp
                        try:
                            if "-->" in current_timestamp:
                                time_parts = current_timestamp.split(" --> ")
                                if len(time_parts) == 2:
                                    end_time = time_parts[1].strip()
                                    # Convert timestamp to seconds (format: HH:MM:SS.mmm)
                                    time_components = end_time.split(":")
                                    if len(time_components) == 3:
                                        hours = int(time_components[0])
                                        minutes = int(time_components[1])
                                        seconds_float = float(time_components[2])
                                        total_seconds = (
                                            hours * 3600 + minutes * 60 + seconds_float
                                        )
                                        duration_seconds = max(
                                            duration_seconds, total_seconds
                                        )
                        except:
                            continue

            return {
                "meeting_id": meeting_id,
                "speakers": speakers,
                "utterances": utterances,
                "duration_seconds": int(duration_seconds),
                "content": transcript_content,
            }

        except Exception as e:
            print(f"❌ Error parsing transcript {transcript_path}: {e}")
            return {
                "meeting_id": transcript_path.stem.replace("transcript-", ""),
                "speakers": set(),
                "utterances": 0,
                "duration_seconds": 0,
                "content": [],
            }

    def analyze_meetings(self) -> None:
        """Analyze meetings data and extract statistics"""
        print("📊 Analyzing meetings data...")

        # Parse all transcript files first to create a lookup
        transcript_data = {}
        for transcript_file in self.transcript_files:
            transcript_info = self.parse_transcript(transcript_file)
            transcript_data[transcript_info["meeting_id"]] = transcript_info

        # Process meetings
        for meeting in self.meetings_data:
            meeting_id = meeting.get("OnlineMeetingId", "")
            participants = meeting.get("Participants", [])

            # Check if transcript exists
            transcript_available = meeting_id in transcript_data
            transcript_info = transcript_data.get(meeting_id, {})

            meeting_stat = MeetingStats(
                meeting_id=meeting_id,
                meeting_type=meeting.get("OnlineMeetingType", ""),
                event_id=meeting.get("EventId", ""),
                start_datetime=meeting.get("StartDateTime", ""),
                end_datetime=meeting.get("EndDateTime", ""),
                owner=meeting.get("Owner", ""),
                participants=participants,
                transcript_available=transcript_available,
                transcript_speakers=transcript_info.get("speakers", set()),
                transcript_duration_seconds=transcript_info.get("duration_seconds", 0),
                transcript_utterances=transcript_info.get("utterances", 0),
            )

            self.analytics.meeting_stats.append(meeting_stat)

            # Update analytics
            for participant in participants:
                self.analytics.unique_meeting_participants.add(participant)
                self.analytics.participant_meeting_counts[participant] = (
                    self.analytics.participant_meeting_counts.get(participant, 0) + 1
                )

            # Update transcript speaker counts
            for speaker in transcript_info.get("speakers", set()):
                self.analytics.transcript_speaker_counts[speaker] = (
                    self.analytics.transcript_speaker_counts.get(speaker, 0) + 1
                )

        self.analytics.total_meetings = len(self.meetings_data)
        self.analytics.total_transcripts = len(self.transcript_files)
        self.analytics.meetings_with_transcripts = len(
            [m for m in self.analytics.meeting_stats if m.transcript_available]
        )

    def analyze_events(self) -> None:
        """Analyze events data and extract statistics"""
        print("📊 Analyzing events data...")

        for event in self.events_data:
            required_attendees = []
            optional_attendees = []

            # Extract required attendees
            req_list = event.get("RequiredAttendees", []) or []
            for attendee in req_list:
                if isinstance(attendee, dict) and "Email" in attendee:
                    required_attendees.append(attendee["Email"])
                elif isinstance(attendee, str):
                    required_attendees.append(attendee)

            # Extract optional attendees
            opt_list = event.get("OptionalAttendees", []) or []
            for attendee in opt_list:
                if isinstance(attendee, dict) and "Email" in attendee:
                    optional_attendees.append(attendee["Email"])
                elif isinstance(attendee, str):
                    optional_attendees.append(attendee)

            attachments = event.get("Attachments", []) or []

            event_stat = EventStats(
                event_id=event.get("EventId", ""),
                sender=event.get("Sender", ""),
                start_datetime=event.get("StartDateTime", ""),
                end_datetime=event.get("EndDateTime", ""),
                subject=event.get("Subject", ""),
                body=event.get("Body", ""),
                required_attendees=required_attendees,
                optional_attendees=optional_attendees,
                is_online_meeting=event.get("IsOnlineMeeting", False),
                has_attachments=len(attachments) > 0,
                attachment_count=len(attachments),
                attachments=attachments,
            )

            self.analytics.event_stats.append(event_stat)

            # Update analytics for all attendees
            all_attendees = required_attendees + optional_attendees
            for attendee in all_attendees:
                self.analytics.unique_event_attendees.add(attendee)
                self.analytics.participant_event_counts[attendee] = (
                    self.analytics.participant_event_counts.get(attendee, 0) + 1
                )

        self.analytics.total_events = len(self.events_data)

    def analyze_meeting_chats(self) -> None:
        """Analyze meeting-type chat messages and correlate with formal meetings"""
        print("📊 Analyzing meeting-type chat messages...")

        meeting_chats = [
            chat for chat in self.chats_data if chat.get("ChatType") == "Meeting"
        ]
        self.analytics.total_meeting_chats = len(meeting_chats)

        # Create mapping of event IDs to meetings for correlation
        event_to_meeting = {}
        for meeting in self.analytics.meeting_stats:
            if meeting.event_id:
                event_to_meeting[meeting.event_id] = meeting

        for chat in meeting_chats:
            event_id = chat.get("EventId", "")
            chat_id = chat.get("ChatId", "")
            members = set(chat.get("Members", []))
            messages = chat.get("ChatMessages", [])

            # Find corresponding meeting
            corresponding_meeting = event_to_meeting.get(event_id)
            participants_overlap = set()
            if corresponding_meeting:
                participants_overlap = members.intersection(
                    set(corresponding_meeting.participants)
                )

            correlation = MeetingChatCorrelation(
                meeting_id=(
                    corresponding_meeting.meeting_id if corresponding_meeting else ""
                ),
                event_id=event_id,
                chat_id=chat_id,
                chat_messages=messages,
                message_count=len(messages),
                participants_overlap=participants_overlap,
            )

            self.analytics.meeting_chat_correlations.append(correlation)

    def analyze_file_references(self) -> None:
        """Extract and analyze file references from meeting chats"""
        print("📁 Analyzing file references in meeting chats...")

        import re

        file_references = set()

        # Extract file names from meeting chat messages
        for correlation in self.analytics.meeting_chat_correlations:
            for message in correlation.chat_messages:
                content = message.get("Content", "")
                # Look for common file patterns mentioned in text
                # Match patterns like: filename.ext, /path/to/file.ext, etc.
                # Pattern to match filenames with common extensions
                file_pattern = r"\b[\w\-\.]+\.(txt|pdf|docx|xlsx|pptx|csv|json|md|py|js|ts|java|cpp|h|xml|html|css|sql|log|zip|tar|gz|c|vtt|config|tsv|png|jpg|jpeg|gif|mp4|avi)\b"
                matches = re.findall(file_pattern, content, re.IGNORECASE)
                for match in matches:
                    # match is the full filename (not a tuple since we're not using groups)
                    file_references.add(match)

        self.analytics.total_file_references = len(file_references)
        self.analytics.unique_file_references = file_references

        # Count actual files in files folder
        if self.files_folder.exists():
            actual_files = set()
            for file_path in self.files_folder.rglob("*"):
                if file_path.is_file():
                    actual_files.add(file_path.name)

            self.analytics.files_in_folder = actual_files

            # Count how many referenced files exist in folder
            files_found = file_references.intersection(actual_files)
            self.analytics.files_found_in_folder = len(files_found)

            print(f"   📎 File references in chats: {len(file_references)}")
            print(f"   📂 Actual files in folder: {len(actual_files)}")
            print(f"   ✅ Files found: {len(files_found)}")
        else:
            print(f"   ⚠️ Files folder not found: {self.files_folder}")

    def generate_meetings_markdown_report(
        self, output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive markdown report for meetings and events

        Args:
            output_file: Optional file path to save the report

        Returns:
            Markdown content as string
        """
        report_lines = []

        # Header
        report_lines.extend(
            [
                "# Comprehensive Meetings & Events Analytics Report",
                "",
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Executive Summary",
                "",
                "### Meetings & Events Overview",
                f"- **Total meetings**: {self.analytics.total_meetings}",
                f"- **Meetings with transcripts**: {self.analytics.meetings_with_transcripts}",
                f"- **Total events**: {self.analytics.total_events}",
                f"- **Unique meeting participants**: {len(self.analytics.unique_meeting_participants)}",
                f"- **Unique event attendees**: {len(self.analytics.unique_event_attendees)}",
                "",
                "---",
                "",
            ]
        )

        # Add meeting chat statistics before the table
        if self.analytics.meeting_chat_correlations:
            report_lines.extend(
                [
                    "## Meeting Chat Summary",
                    "",
                    f"- **Total meeting-type chats**: {self.analytics.total_meeting_chats}",
                    f"- **Chats with event correlation**: {len([c for c in self.analytics.meeting_chat_correlations if c.event_id])}",
                    f"- **Total meeting chat messages**: {sum(c.message_count for c in self.analytics.meeting_chat_correlations)}",
                    "",
                ]
            )

        # Add file statistics
        if (
            self.analytics.total_file_references > 0
            or len(self.analytics.files_in_folder) > 0
        ):
            report_lines.extend(
                [
                    "## File References",
                    "",
                    f"- **File references in meeting chats**: {self.analytics.total_file_references}",
                    f"- **Actual files in folder**: {len(self.analytics.files_in_folder)}",
                    f"- **Files found in folder**: {self.analytics.files_found_in_folder}",
                    "",
                ]
            )

        # Top participants table
        report_lines.extend(
            [
                "## Combined Participation Ranking",
                "",
                "| Rank | Participant | Meetings Attended | Events Attended | Total Participations | Speaking Sessions |",
                "|------|-------------|------------------|----------------|---------------------|------------------|",
            ]
        )

        # Calculate combined participation scores
        combined_scores = {}
        for participant in self.analytics.unique_meeting_participants.union(
            self.analytics.unique_event_attendees
        ):
            meeting_count = self.analytics.participant_meeting_counts.get(
                participant, 0
            )
            event_count = self.analytics.participant_event_counts.get(participant, 0)
            combined_scores[participant] = {
                "meetings": meeting_count,
                "events": event_count,
                "total": meeting_count + event_count,
            }

        # Sort by total participation
        sorted_participants = sorted(
            combined_scores.items(), key=lambda x: x[1]["total"], reverse=True
        )

        for rank, (participant, scores) in enumerate(sorted_participants[:20], 1):
            # Get transcript speaker count for this participant
            # Match participant email to speaker name (case-insensitive, partial match)
            speaking_sessions = 0
            participant_name = participant.split("@")[0].replace(".", " ").title()
            for speaker, count in self.analytics.transcript_speaker_counts.items():
                # Check if speaker name matches participant (case-insensitive)
                if (
                    speaker.lower().replace(" ", ".")
                    == participant.split("@")[0].lower()
                ):
                    speaking_sessions = count
                    break

            report_lines.append(
                f"| {rank} | {participant} | {scores['meetings']} | {scores['events']} | {scores['total']} | {speaking_sessions} |"
            )

        # Combine all lines
        markdown_content = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                print(f"✅ Meetings analytics report saved to: {output_file}")
            except Exception as e:
                print(f"❌ Error saving report: {e}")

        return markdown_content

    def extract_meetings_for_complex_queries(
        self, participant_filter: Optional[str] = None, min_participants: int = 2
    ) -> Dict:
        """
        Extract meeting content optimized for complex query generation

        Args:
            participant_filter: Optional email to filter by specific participant
            min_participants: Minimum number of participants required

        Returns:
            Dictionary containing structured meeting data for query generation
        """
        complex_query_data = {
            "meetings_with_transcripts": [],
            "events_with_details": [],
            "meeting_chat_correlations": [],
            "statistics": {
                "total_meetings": self.analytics.total_meetings,
                "total_events": self.analytics.total_events,
                "total_transcripts": self.analytics.total_transcripts,
                "unique_participants": len(self.analytics.unique_meeting_participants),
                "unique_attendees": len(self.analytics.unique_event_attendees),
            },
        }

        # Extract meeting data with transcripts
        for meeting in self.analytics.meeting_stats:
            if len(meeting.participants) >= min_participants:
                if (
                    participant_filter is None
                    or participant_filter in meeting.participants
                ):
                    # Get transcript content
                    transcript_content = []
                    if meeting.transcript_available:
                        transcript_file = (
                            self.transcripts_folder
                            / f"transcript-{meeting.meeting_id}.vtt"
                        )
                        if transcript_file.exists():
                            transcript_info = self.parse_transcript(transcript_file)
                            transcript_content = transcript_info.get("content", [])

                    meeting_data = {
                        "meeting_id": meeting.meeting_id,
                        "meeting_type": meeting.meeting_type,
                        "event_id": meeting.event_id,
                        "participants": meeting.participants,
                        "participant_count": len(meeting.participants),
                        "transcript_available": meeting.transcript_available,
                        "transcript_speakers": list(meeting.transcript_speakers),
                        "transcript_utterances": meeting.transcript_utterances,
                        "transcript_content": (
                            transcript_content[:50] if transcript_content else []
                        ),  # Limit for processing
                    }
                    complex_query_data["meetings_with_transcripts"].append(meeting_data)

        # Extract events with rich content
        for event in self.analytics.event_stats:
            all_attendees = event.required_attendees + event.optional_attendees
            if len(all_attendees) >= min_participants:
                if participant_filter is None or participant_filter in all_attendees:
                    event_data = {
                        "event_id": event.event_id,
                        "sender": event.sender,
                        "subject": event.subject,
                        "body": (
                            event.body[:1000] if event.body else ""
                        ),  # Limit for processing
                        "required_attendees": event.required_attendees,
                        "optional_attendees": event.optional_attendees,
                        "total_attendees": len(all_attendees),
                        "has_attachments": event.has_attachments,
                        "attachment_count": event.attachment_count,
                        "attachments": (
                            event.attachments[:10] if event.attachments else []
                        ),  # Limit for processing
                    }
                    complex_query_data["events_with_details"].append(event_data)

        # Extract meeting chat correlations
        for correlation in self.analytics.meeting_chat_correlations:
            if (
                len(correlation.participants_overlap) >= min_participants
                or min_participants <= 1
            ):
                if (
                    participant_filter is None
                    or participant_filter in correlation.participants_overlap
                ):
                    correlation_data = {
                        "meeting_id": correlation.meeting_id,
                        "event_id": correlation.event_id,
                        "chat_id": correlation.chat_id,
                        "message_count": correlation.message_count,
                        "participants_overlap": list(correlation.participants_overlap),
                        "chat_messages": (
                            correlation.chat_messages[:20]
                            if correlation.chat_messages
                            else []
                        ),  # Limit for processing
                    }
                    complex_query_data["meeting_chat_correlations"].append(
                        correlation_data
                    )

        return complex_query_data

    def run_full_analysis(self) -> MeetingsEventsAnalytics:
        """
        Run complete analysis of meetings, events, transcripts, meeting chats, and file references

        Returns:
            Complete analytics results
        """
        self.load_data()
        self.analyze_meetings()
        self.analyze_events()
        self.analyze_meeting_chats()
        self.analyze_file_references()

        print(f"✅ Meetings & Events analysis complete:")
        print(f"   📅 Total meetings: {self.analytics.total_meetings}")
        print(f"   📊 Total events: {self.analytics.total_events}")
        print(f"   🎤 Total transcripts: {self.analytics.total_transcripts}")
        print(
            f"   💬 Meetings with transcripts: {self.analytics.meetings_with_transcripts}"
        )
        print(f"   📝 Total meeting-type chats: {self.analytics.total_meeting_chats}")
        print(f"   📎 File references in chats: {self.analytics.total_file_references}")
        print(
            f"   📂 Files found in folder: {self.analytics.files_found_in_folder}/{len(self.analytics.files_in_folder)}"
        )
        print(
            f"   👥 Unique meeting participants: {len(self.analytics.unique_meeting_participants)}"
        )
        print(
            f"   📧 Unique event attendees: {len(self.analytics.unique_event_attendees)}"
        )

        return self.analytics


# ============================================
# CLI INTERFACE


def generate_meetings_events_analytics(
    data_folder: str = "data", output_file: Optional[str] = None
) -> None:
    """
    Generate comprehensive meetings and events analytics

    Args:
        data_folder: Path to folder containing config files and transcripts
        output_file: Optional path to save markdown report
    """
    print("🚀 Starting Meetings & Events Analytics...")

    analyzer = MeetingsEventsAnalyzer(data_folder)
    analytics_results = analyzer.run_full_analysis()

    # Generate and optionally save report
    if output_file:
        analyzer.generate_meetings_markdown_report(output_file)
    else:
        print("\n" + "=" * 60)
        print("📊 MEETINGS & EVENTS ANALYTICS SUMMARY")
        print("=" * 60)
        content = analyzer.generate_meetings_markdown_report()
        print(content[:2000] + "..." if len(content) > 2000 else content)

    print(f"\n✅ Meetings & Events analytics completed successfully!")


def extract_meetings_events_with_prompt(
    user_email: str,
    data_folder: str = "data",
    min_participants: int = 2,
    output_folder: str = "results",
    prompt_file: str = "prompt_meetings_gen_complex_utterance.md",
) -> str:
    """
    Extract meetings, events, and transcripts for a specific user with LLM prompt generation
    
    Extracts meetings, events, transcripts, and meeting chats for a specific user to generate 
    complex utterances that require reasoning capabilities. Focuses on formal meeting contexts,
    decision-making processes, and structured collaboration patterns from the user's perspective.
    
    Generates two output files:
    1. JSON data file with extracted meeting/event content for the specific user
    2. LLM prompt file with template populated with user-specific extracted data
    
    This mirrors the Teams analytics approach where queries are individualized for each user,
    as search results are personalized based on user permissions and meeting participation.

    Args:
        user_email: Specific user email to analyze (required) - generates queries from this user's perspective
        data_folder: Path to folder containing config files and transcripts
        min_participants: Minimum number of participants required (default: 2)
        output_folder: Folder to save the output files (default: "results")
        prompt_file: Prompt template file name (default: "prompt_meetings_gen_complex_utterance.md")
        
    Returns:
        str: Human-readable summary of the processing results

    Example:
        python meetings_events_analytics.py extract_meetings_events_with_prompt \
            --user_email=marion.chen \
            --data_folder=data \
            --min_participants=2 \
            --output_folder=results \
            --prompt_file=prompt_meetings_gen_complex_utterance.md
    """
    logger.info(f"🎯 Extracting meetings data for user: {user_email}")
    logger.info(f"   📂 Data folder: {data_folder}")
    logger.info(f"   � Minimum participants: {min_participants}")
    logger.info(f"   �📂 Output folder: {output_folder}")
    logger.info(f"   📝 Prompt template: {prompt_file}")

    # Ensure output directory exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    analyzer = MeetingsEventsAnalyzer(data_folder)
    analyzer.run_full_analysis()

    query_data = analyzer.extract_meetings_for_complex_queries(
        participant_filter=user_email, min_participants=min_participants
    )

    logger.info(f"\n📊 Extraction Results:")
    logger.info(
        f"   📅 Meetings with transcripts: {len(query_data['meetings_with_transcripts'])}"
    )
    logger.info(f"   📋 Events with details: {len(query_data['events_with_details'])}")
    logger.info(
        f"   💬 Meeting chat correlations: {len(query_data['meeting_chat_correlations'])}"
    )

    # Save JSON data file - use user_email for filename (similar to Teams implementation)
    user_name = user_email.split("@")[0].replace(".", "_")
    json_filename = f"{user_name}_meetings_events_data.json"
    json_filepath = os.path.join(output_folder, json_filename)

    logger.info(f"💾 Saving meetings data to: {json_filepath}")
    try:
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(query_data, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"✅ Meetings data saved successfully")
    except Exception as e:
        logger.error(f"❌ Error saving JSON data: {e}")
        raise

    # Generate and save prompt file
    prompt_filepath = _save_meetings_prompt_file(
        query_data, json_filepath, prompt_file, output_folder, user_email
    )

    # Print summary
    _print_meetings_extraction_summary(query_data, json_filepath, prompt_filepath)

    # Display sample of extracted data
    if query_data["meetings_with_transcripts"]:
        sample_meeting = query_data["meetings_with_transcripts"][0]
        logger.info(
            f"\n� Sample meeting: {sample_meeting['meeting_id'][:20]}... with {sample_meeting['participant_count']} participants"
        )
    if query_data["events_with_details"]:
        sample_event = query_data["events_with_details"][0]
        logger.info(f"🔍 Sample event: {sample_event['subject'][:50]}...")

    return f"""✅ Meeting extraction completed successfully!
🔍 User: {user_email}
👥 Min participants: {min_participants}
📅 Meetings extracted: {len(query_data['meetings_with_transcripts'])}
📋 Events extracted: {len(query_data['events_with_details'])}
💬 Chat correlations: {len(query_data['meeting_chat_correlations'])}
📄 Output files created:
   - Raw data: {json_filepath}
   - LLM prompt: {prompt_filepath}"""


def _save_meetings_prompt_file(
    query_data: Dict[str, Any],
    json_file: str,
    prompt_file: str,
    output_folder: str,
    email_account: str,
) -> str:
    """Save the complete meetings + events prompt file with JSON data inserted

    Args:
        query_data: Extracted meetings and events data
        json_file: JSON data file path
        prompt_file: Prompt template file name
        output_folder: Output folder
        email_account: Email account for replacement in template

    Returns:
        str: Path to the saved prompt file
    """
    # Read the prompt template
    prompt_template_path = Path(__file__).parent / prompt_file

    logger.info(f"� Reading meetings prompt template: {prompt_template_path}")
    try:
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        logger.info(
            f"✅ Successfully read meetings prompt template ({len(prompt_template)} characters)"
        )
    except FileNotFoundError:
        error_msg = f"❌ ERROR: Meetings prompt template file not found at {prompt_template_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"❌ ERROR: Failed to read meetings prompt template at {prompt_template_path}: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)

    # Convert JSON to formatted string
    json_str = json.dumps(query_data, indent=2, default=str, ensure_ascii=False)

    # Replace placeholders in template
    complete_prompt = prompt_template.replace("{meetings_events_json_data}", json_str)
    complete_prompt = complete_prompt.replace("{email_account}", email_account)

    # Generate prompt file path - use user email for filename (similar to Teams implementation)
    user_name = email_account.split("@")[0].replace(".", "_")
    prompt_file_path = os.path.join(
        output_folder, f"{user_name}_meetings_events_prompt.md"
    )

    # Save prompt file
    with open(prompt_file_path, "w", encoding="utf-8") as f:
        f.write(complete_prompt)

    logger.info(f"📝 Meetings prompt file saved to: {prompt_file_path}")
    return prompt_file_path


def _print_meetings_extraction_summary(
    query_data: Dict[str, Any], json_file: str, prompt_file_path: str
) -> None:
    """Print a summary of the meetings extraction

    Args:
        query_data: Extracted meetings and events data
        json_file: Path to the JSON data file
        prompt_file_path: Path to the generated prompt file
    """
    stats = query_data["statistics"]

    logger.info(f"\n🎉 Meetings & Events Extraction Complete!")
    logger.info(f"=" * 70)
    logger.info(f"💾 JSON Data File: {json_file}")
    logger.info(f"📝 Prompt File: {prompt_file_path}")
    logger.info(f"📅 Total Meetings: {stats['total_meetings']}")
    logger.info(f"📊 Total Events: {stats['total_events']}")
    logger.info(f"🎤 Total Transcripts: {stats['total_transcripts']}")
    logger.info(f"👥 Unique Participants: {stats['unique_participants']}")
    logger.info(f"📧 Unique Attendees: {stats['unique_attendees']}")
    logger.info(
        f"📅 Meetings Extracted: {len(query_data['meetings_with_transcripts'])}"
    )
    logger.info(f"📋 Events Extracted: {len(query_data['events_with_details'])}")
    logger.info(f"💬 Chat Correlations: {len(query_data['meeting_chat_correlations'])}")

    # Show meeting types distribution
    if query_data["meetings_with_transcripts"]:
        meeting_types = {}
        for meeting in query_data["meetings_with_transcripts"]:
            meeting_type = meeting.get("meeting_type", "unknown")
            meeting_types[meeting_type] = meeting_types.get(meeting_type, 0) + 1

        if meeting_types:
            logger.info(f"\n📊 Meeting Types Distribution:")
            for meeting_type, count in sorted(meeting_types.items()):
                logger.info(f"   {meeting_type}: {count} meetings")

    # Show transcript coverage
    meetings_with_transcripts = sum(
        1 for m in query_data["meetings_with_transcripts"] if m["transcript_available"]
    )
    logger.info(
        f"\n🎤 Transcript Coverage: {meetings_with_transcripts}/{len(query_data['meetings_with_transcripts'])} meetings"
    )


# ============================================
# MAIN CLI ENTRY POINT

if __name__ == "__main__":
    fire.Fire(
        {
            "generate_meetings_events_analytics": generate_meetings_events_analytics,
            "extract_meetings_events_with_prompt": extract_meetings_events_with_prompt,
        }
    )
