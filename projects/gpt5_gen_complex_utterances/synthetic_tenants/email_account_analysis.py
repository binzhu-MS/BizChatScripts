#!/usr/bin/env python3
"""
Comprehensive Email Account Classification System

This module implements a structured approach to classify email accounts:
1. Extract all email accounts from emails database
2. Check against users file to identify individual accounts
3. Classify remaining accounts as bot, group, or unknown based on patterns
4. For group accounts, identify members using LLM analysis
5. Generate detailed reports with account classifications and group membership

Usage:
    Run this program as a Python module from the BizChatScripts root directory:

    python -m projects.gpt5_gen_complex_utterances.synthetic_tenants.email_account_analysis --help

    This will show all available commands and their options.
"""

import json
import re
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
import fire
import sys
import os

# Import LLM functionality - should always be available
from llms.llm_api import LLMAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class UserInfo:
    """User information extracted from users.config.json"""

    mail_nickname: str
    display_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    job_title: Optional[str] = None
    department: Optional[str] = None
    company_name: Optional[str] = None
    manager: Optional[str] = None
    office_location: Optional[str] = None


@dataclass
class AccountClassification:
    """Classification result for an email account"""

    email: str
    account_type: str  # 'user', 'bot', 'group', 'unknown'
    frequency: int  # Total email occurrences
    emails_sent: int  # Number of emails sent
    emails_received: int  # Number of emails received
    user_info: Optional[UserInfo] = None
    group_meaning: Optional[str] = None  # For group accounts
    potential_members: Optional[List[str]] = None  # For group accounts


class EmailAccountClassifier:
    """Comprehensive email account classification system"""

    def __init__(
        self, emails_file: Optional[str] = None, users_file: Optional[str] = None
    ):
        """Initialize classifier with data files"""
        self.emails_file = emails_file
        self.users_file = users_file
        self.emails_data = None
        self.users_data = None
        self.users_by_nickname: Dict[str, UserInfo] = {}
        self.user_aliases: Dict[str, List[str]] = (
            {}
        )  # Maps primary nickname to all aliases

        # Classification patterns
        self.bot_patterns = [
            r".*\.bot$",
            r".*bot$",
            r".*\.ci$",
            r"^ci\.",
            r".*system$",
            r".*automation$",
            r".*deployer$",
            r".*validator$",
            r".*reporter$",
            r".*alertmanager$",
            r".*jenkins$",
            r".*\.alerts$",
        ]

        self.group_patterns = [
            r".*team$",
            r".*\.team$",
            r".*-team$",
            r"^ci\.team$",
            r".*ops$",
            r".*\.ops$",
            r".*-ops$",
            r".*admin$",
            r".*\.admin$",
            r".*-admin$",
            r".*leads$",
            r".*\.leads$",
            r".*-leads$",
            r".*support$",
            r".*\.support$",
            r".*-support$",
            r".*security$",
            r".*\.security$",
            r".*compliance$",
            r".*\.compliance$",
            r".*review$",
            r".*\.review$",
            r".*board$",
            r".*\.board$",
            r".*director$",
            r".*\.director$",
        ]

        # Group meanings
        self.group_meanings = {
            "devops": "Development Operations team responsible for CI/CD, infrastructure, and deployment",
            "security": "Security team responsible for cybersecurity, compliance, and threat management",
            "ops": "Operations team responsible for system operations and maintenance",
            "admin": "Administrative team responsible for system administration",
            "leads": "Technical leads or team leadership group",
            "support": "Technical support team for user assistance",
            "alerts": "Automated alert distribution system",
            "monitoring": "System monitoring and observability team",
            "compliance": "Compliance and regulatory team",
            "engineering": "Engineering team for software development",
            "qa": "Quality Assurance team for testing",
            "pm": "Product Management team",
            "arch": "Architecture review team for technical design",
            "sre": "Site Reliability Engineering team",
            "platform": "Platform engineering team",
            "network": "Network operations team",
            "ci": "Continuous Integration team/system",
            "ml": "Machine Learning operations team",
            "db": "Database administration team",
        }

    def load_data(self) -> None:
        """Load both emails and users data"""
        # Fail if files are not defined
        if not self.emails_file:
            raise ValueError(
                "emails_file must be provided - cannot proceed without email data file"
            )
        if not self.users_file:
            raise ValueError(
                "users_file must be provided - cannot proceed without users data file"
            )

        # Load emails data
        try:
            with open(self.emails_file, "r", encoding="utf-8") as f:
                self.emails_data = json.load(f)
            logger.info(
                f"Loaded {len(self.emails_data)} emails from {self.emails_file}"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"✗ Error: Could not find emails file: {self.emails_file}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"✗ Error: Invalid JSON in emails file {self.emails_file}: {e}"
            )

        # Load users data
        try:
            with open(self.users_file, "r", encoding="utf-8") as f:
                self.users_data = json.load(f)

            # Track all aliases for each person for display purposes
            self.user_aliases = {}  # Maps primary nickname to list of all aliases

            # First pass: Group users by full name to detect duplicates/aliases
            users_by_full_name = {}
            for user in self.users_data:
                if isinstance(user, dict) and "MailNickName" in user:
                    display_name = user.get("DisplayName", "").strip()
                    first_name = user.get("FirstName", "").strip()
                    last_name = user.get("LastName", "").strip()

                    # Create a normalized full name for grouping
                    if display_name:
                        full_name = display_name.lower()
                    elif first_name and last_name:
                        full_name = f"{first_name} {last_name}".lower()
                    else:
                        # Use MailNickName as fallback
                        full_name = user["MailNickName"].lower()

                    if full_name not in users_by_full_name:
                        users_by_full_name[full_name] = []
                    users_by_full_name[full_name].append(user)

            # Second pass: Consolidate users and create mappings
            for full_name, user_list in users_by_full_name.items():
                # Find the most complete user record (one with most non-empty fields)
                best_user = max(
                    user_list,
                    key=lambda u: sum(
                        1
                        for v in [
                            u.get("JobTitle"),
                            u.get("Department"),
                            u.get("CompanyName"),
                            u.get("Manager"),
                            u.get("OfficeLocation"),
                            u.get("Address"),
                        ]
                        if v and str(v).strip()
                    ),
                )

                # Check for conflicts in key fields among users with same name
                conflicts = []
                company_names = set(
                    u.get("CompanyName", "").strip()
                    for u in user_list
                    if u.get("CompanyName", "").strip()
                )
                departments = set(
                    u.get("Department", "").strip()
                    for u in user_list
                    if u.get("Department", "").strip()
                )

                if len(company_names) > 1:
                    conflicts.append(f"companies: {', '.join(company_names)}")
                if len(departments) > 1:
                    conflicts.append(f"departments: {', '.join(departments)}")

                # If there are conflicts, treat as separate users
                if conflicts:
                    logger.warning(
                        f"Conflict detected for '{full_name}' - {'; '.join(conflicts)}. Treating as separate users."
                    )
                    # Process each user separately without consolidation
                    for user in user_list:
                        self._create_user_mappings(user)
                else:
                    # Consolidate: use best_user info but map all nicknames to it
                    user_info = UserInfo(
                        mail_nickname=best_user["MailNickName"].lower(),
                        display_name=best_user.get("DisplayName"),
                        first_name=best_user.get("FirstName"),
                        last_name=best_user.get("LastName"),
                        job_title=best_user.get("JobTitle"),
                        department=best_user.get("Department"),
                        company_name=best_user.get("CompanyName"),
                        manager=best_user.get("Manager"),
                        office_location=best_user.get("OfficeLocation"),
                    )

                    # Collect all aliases for this person
                    primary_nickname = best_user["MailNickName"].lower()
                    all_aliases = []

                    # Map all nicknames from all user records to this consolidated info
                    for user in user_list:
                        nickname = user["MailNickName"].lower()
                        self.users_by_nickname[nickname] = user_info
                        all_aliases.append(nickname)

                    # Store all aliases for this person
                    self.user_aliases[primary_nickname] = sorted(set(all_aliases))

                    if len(user_list) > 1:
                        nicknames = [u["MailNickName"] for u in user_list]
                        logger.info(
                            f"Consolidated {len(user_list)} accounts for '{full_name}': {', '.join(nicknames)}"
                        )

            logger.info(f"Loaded {len(self.users_data)} users from {self.users_file}")
            logger.info(f"Processed {len(self.users_by_nickname)} user nicknames")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"✗ Error: Could not find users file: {self.users_file}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"✗ Error: Invalid JSON in users file {self.users_file}: {e}"
            )

    def _create_user_mappings(self, user):
        """Create mappings for a single user without consolidation"""
        user_info = UserInfo(
            mail_nickname=user["MailNickName"].lower(),
            display_name=user.get("DisplayName"),
            first_name=user.get("FirstName"),
            last_name=user.get("LastName"),
            job_title=user.get("JobTitle"),
            department=user.get("Department"),
            company_name=user.get("CompanyName"),
            manager=user.get("Manager"),
            office_location=user.get("OfficeLocation"),
        )

        # Primary mapping
        nickname = user["MailNickName"].lower()
        self.users_by_nickname[nickname] = user_info

        # Track aliases for this user (only actual MailNickName from input file)
        all_aliases = [nickname]

        # Store all aliases for this person
        self.user_aliases[nickname] = sorted(set(all_aliases))

    def load_prompt_template(self) -> str:
        """Load the LLM prompt template from markdown file"""
        try:
            prompt_file = os.path.join(
                os.path.dirname(__file__),
                "prompt_email_account_analysis_for_group_membership.md",
            )
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error("Prompt file group_membership_prompt.md not found")
            raise

    def extract_all_email_accounts(
        self,
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        """Extract all email accounts that have sent or received emails
        Returns: (total_counts, sent_counts, received_counts)"""
        if not self.emails_data:
            self.load_data()

        if not self.emails_data:
            return {}, {}, {}

        email_accounts = defaultdict(int)
        sent_counts = defaultdict(int)
        received_counts = defaultdict(int)

        for email in self.emails_data:
            # Extract sender
            if "Sender" in email and email["Sender"]:
                sender = email["Sender"].lower()
                if sender:
                    email_accounts[sender] += 1
                    sent_counts[sender] += 1

            # Extract recipients from all recipient fields
            for field in ["ToRecipients", "CcRecipients", "BccRecipients"]:
                if field in email and email[field] is not None:
                    for recipient in email[field]:
                        if isinstance(recipient, dict) and "Recipient" in recipient:
                            addr = recipient["Recipient"].lower()
                            email_accounts[addr] += 1
                            received_counts[addr] += 1
                        elif isinstance(recipient, str):
                            addr = recipient.lower()
                            email_accounts[addr] += 1
                            received_counts[addr] += 1

        return dict(email_accounts), dict(sent_counts), dict(received_counts)

    def classify_account(self, email: str) -> str:
        """Classify an email account as user, bot, group, or unknown"""
        # Check if it's a known user
        if email in self.users_by_nickname:
            # Even if in users file, check for bot patterns
            for pattern in self.bot_patterns:
                if re.match(pattern, email, re.IGNORECASE):
                    return "bot"

            # Check for group patterns
            for pattern in self.group_patterns:
                if re.match(pattern, email, re.IGNORECASE):
                    return "group"

            return "user"

        # Not in users file - classify by patterns
        for pattern in self.bot_patterns:
            if re.match(pattern, email, re.IGNORECASE):
                return "bot"

        for pattern in self.group_patterns:
            if re.match(pattern, email, re.IGNORECASE):
                return "group"

        return "unknown"

    def get_group_meaning(self, email: str) -> str:
        """Determine the meaning/purpose of a group account"""
        email_lower = email.lower()

        # Find matching group type
        for group_type, meaning in self.group_meanings.items():
            if group_type in email_lower:
                return meaning

        # Generic meanings based on suffix
        if email_lower.endswith("team") or ".team" in email_lower:
            return f"Team distribution list for {email.split('.')[0]} group"
        elif email_lower.endswith("ops") or ".ops" in email_lower:
            return f"Operations team for {email.split('.')[0]} domain"
        elif email_lower.endswith("admin") or ".admin" in email_lower:
            return f"Administrative team for {email.split('.')[0]} systems"
        elif email_lower.endswith("leads") or ".leads" in email_lower:
            return f"Leadership/senior members of {email.split('.')[0]} team"
        elif email_lower.endswith("support") or ".support" in email_lower:
            return f"Support team for {email.split('.')[0]} services"

        return f"Group account for {email} (manual evaluation needed)"

    def _extract_json_array(self, content: str) -> Optional[List[str]]:
        """Extract JSON array from LLM response using multiple strategies"""
        import re

        # Strategy 1: Direct JSON parsing (if LLM followed instructions)
        try:
            members = json.loads(content.strip())
            if isinstance(members, list):
                return members
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove markdown code blocks
        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            members = json.loads(cleaned)
            if isinstance(members, list):
                return members
        except json.JSONDecodeError:
            pass

        # Strategy 3: Extract JSON array using regex
        try:
            # Look for a JSON array pattern in the text
            json_pattern = r'\[\s*(?:"[^"]*"\s*(?:,\s*"[^"]*"\s*)*|)\s*\]'
            matches = re.findall(json_pattern, content)
            if matches:
                # Try to parse the first match
                members = json.loads(matches[0])
                if isinstance(members, list):
                    return members
        except (json.JSONDecodeError, IndexError):
            pass

        # Strategy 4: Look for quoted strings that might be email addresses
        try:
            # Extract all quoted strings that look like email addresses or nicknames
            quoted_strings = re.findall(r'"([^"]+)"', content)
            if quoted_strings:
                # Filter for reasonable email/nickname patterns
                email_pattern = r"^[a-zA-Z0-9._-]+(@[a-zA-Z0-9.-]+)?$"
                potential_members = [
                    s
                    for s in quoted_strings
                    if re.match(email_pattern, s) and len(s) > 1
                ]
                if potential_members:
                    return potential_members
        except Exception:
            pass

        return None

    def find_group_members_llm(self, group_email: str, group_meaning: str) -> List[str]:
        """Use LLM to intelligently determine group membership based on user profiles"""

        logger.info(f"🤖 Analyzing group membership for: {group_email}")
        logger.info(f"   Purpose: {group_meaning}")

        try:
            # Get all email accounts to include inactive users too
            email_accounts, _, _ = self.extract_all_email_accounts()

            # Prepare user data for LLM analysis
            user_profiles = []

            # Track which users we've already processed to avoid duplicates
            processed_primary_users = set()

            # Include users from users.config.json (active and inactive)
            # Create one consolidated profile per person that includes all their email aliases
            for nickname, user_info in self.users_by_nickname.items():
                primary_nickname = user_info.mail_nickname

                # Skip if we've already processed this person (avoid duplicates for same person)
                if primary_nickname in processed_primary_users:
                    continue

                processed_primary_users.add(primary_nickname)

                # Get all aliases for this person
                all_aliases = self.user_aliases.get(primary_nickname, [nickname])

                # Filter to only include aliases that actually exist in our system
                valid_aliases = [
                    alias for alias in all_aliases if alias in self.users_by_nickname
                ]

                # Create a single consolidated profile with all email aliases
                # Use the complete user_info (which has the most comprehensive data)
                profile = {
                    "email_aliases": valid_aliases,  # List all email addresses for this person
                    "primary_email": primary_nickname,  # The primary/canonical email
                    "display_name": user_info.display_name or "N/A",
                    "job_title": user_info.job_title or "N/A",
                    "department": user_info.department or "N/A",
                    "company": user_info.company_name or "N/A",
                    "manager": user_info.manager or "N/A",
                    "office_location": user_info.office_location or "N/A",
                }
                user_profiles.append(profile)

            # Also include email accounts that have activity but aren't in users.config.json
            for email_account in email_accounts:
                if email_account not in self.users_by_nickname:
                    # This is someone with email activity but not in users file
                    profile = {
                        "email": email_account,
                        "display_name": "N/A",
                        "job_title": "N/A",
                        "department": "N/A",
                        "company": "N/A",
                        "manager": "N/A",
                        "office_location": "N/A",
                    }
                    user_profiles.append(profile)

            # Create LLM applier
            llm = LLMAPI()

            # Prepare the prompt
            prompt_template = self.load_prompt_template()
            prompt = prompt_template.format(
                group_email=group_email,
                group_meaning=group_meaning,
                user_profiles=json.dumps(user_profiles, indent=2),
            )

            # Prepare model config and messages
            model_config = {
                "model": "dev-gpt-41-longco-2025-04-14",
                "temperature": 0.1,
                "max_tokens": 1000,
            }

            messages = [{"role": "user", "content": prompt}]

            # Make LLM request
            logger.info(f"   🔄 Requesting LLM analysis...")
            response = llm.chat_completion(model_config, {"messages": messages})

            # Extract and parse the response
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"].strip()
                logger.info(f"   ✅ LLM response received, parsing membership...")

                # Try multiple approaches to extract JSON array
                members = self._extract_json_array(content)
                if members is not None:
                    # Filter to only include valid nicknames that exist in our data
                    # (either in users.config.json or have email activity)
                    all_valid_accounts = set(self.users_by_nickname.keys()) | set(
                        email_accounts.keys()
                    )
                    valid_members = [
                        member for member in members if member in all_valid_accounts
                    ]
                    logger.info(
                        f"   ✅ Found {len(valid_members)} valid members for {group_email}"
                    )
                    if valid_members:
                        logger.info(
                            f"      Members: {', '.join(valid_members[:5])}{' ...' if len(valid_members) > 5 else ''}"
                        )
                    return valid_members
                else:
                    logger.warning(
                        f"   ❌ Could not parse LLM response as JSON for {group_email}: {content[:100]}..."
                    )
                    return []

            logger.warning(f"   ❌ No valid response received for {group_email}")
            return []

        except Exception as e:
            logger.warning(f"   ❌ LLM analysis failed for {group_email}: {e}")
            return []

    def user_matches_group(self, user_info: UserInfo, keywords: List[str]) -> bool:
        """Check if a user matches group based on keywords"""
        if not keywords:
            return False

        # Check job title
        if user_info.job_title:
            job_title_lower = user_info.job_title.lower()
            for keyword in keywords:
                if keyword in job_title_lower:
                    return True

        # Check department
        if user_info.department:
            dept_lower = user_info.department.lower()
            for keyword in keywords:
                if keyword in dept_lower:
                    return True

        return False

    def classify_all_accounts(self) -> Dict[str, List[AccountClassification]]:
        """Classify all email accounts and return organized results"""
        if not self.emails_data or not self.users_by_nickname:
            self.load_data()

        email_accounts, sent_counts, received_counts = self.extract_all_email_accounts()

        results = {"user": [], "bot": [], "group": [], "unknown": []}

        # Count total groups for progress tracking
        total_groups = sum(
            1
            for email in email_accounts.keys()
            if self.classify_account(email) == "group"
        )
        processed_groups = 0

        if total_groups > 0:
            logger.info(
                f"📊 Found {total_groups} group accounts - starting LLM analysis..."
            )

        # Track processed users to avoid duplicates
        processed_users = set()  # Track by primary mail_nickname

        for email, frequency in email_accounts.items():
            account_type = self.classify_account(email)

            # Get user info if available
            user_info = self.users_by_nickname.get(email)

            # For user accounts, check if we've already processed this person
            if account_type == "user" and user_info:
                primary_nickname = user_info.mail_nickname
                if primary_nickname in processed_users:
                    # This person is already processed, consolidate the email counts
                    for existing_acc in results["user"]:
                        if (
                            existing_acc.user_info
                            and existing_acc.user_info.mail_nickname == primary_nickname
                        ):
                            # Add this email's activity to existing entry
                            existing_acc.frequency += frequency
                            existing_acc.emails_sent += sent_counts.get(email, 0)
                            existing_acc.emails_received += received_counts.get(
                                email, 0
                            )
                            break
                    continue
                else:
                    processed_users.add(primary_nickname)

            # Get group meaning and members for group accounts
            group_meaning = None
            potential_members = []
            if account_type == "group":
                processed_groups += 1
                logger.info(f"📈 Processing group {processed_groups}/{total_groups}")
                group_meaning = self.get_group_meaning(email)
                potential_members = self.find_group_members_llm(email, group_meaning)

            classification = AccountClassification(
                email=email,
                account_type=account_type,
                frequency=frequency,
                user_info=user_info,
                group_meaning=group_meaning,
                potential_members=potential_members or [],
                emails_sent=sent_counts.get(email, 0),
                emails_received=received_counts.get(email, 0),
            )

            results[account_type].append(classification)

        # Log completion of group processing
        if total_groups > 0:
            logger.info(
                f"🎉 Completed LLM analysis for all {total_groups} group accounts!"
            )

        # Sort each category by frequency
        for category in results:
            results[category].sort(key=lambda x: x.frequency, reverse=True)

        return results

    def find_inactive_users(self) -> List[UserInfo]:
        """Find users from users.config.json who have no email activity"""
        if not self.emails_data or not self.users_by_nickname:
            self.load_data()

        # Get all email accounts with activity
        email_accounts, _, _ = self.extract_all_email_accounts()
        active_users = set(email_accounts.keys())

        # Find users not in email activity
        inactive_users = []
        for nickname, user_info in self.users_by_nickname.items():
            if nickname not in active_users:
                inactive_users.append(user_info)

        # Sort by display name for consistent output
        inactive_users.sort(key=lambda x: x.display_name or x.mail_nickname)
        return inactive_users

    def generate_comprehensive_report(self, output_file: str) -> None:
        """Generate comprehensive analysis report in markdown format"""
        if not self.emails_data or not self.users_by_nickname:
            self.load_data()

        results = self.classify_all_accounts()

        report_lines = []

        # Markdown header
        report_lines.append("# Comprehensive Email Account Classification Report")
        report_lines.append("*with LLM-Enhanced Group Membership Analysis*")
        report_lines.append("")
        report_lines.append(f"**Analysis Date:** {self.get_current_date()}")
        report_lines.append("")

        # Summary statistics
        total_accounts = sum(len(results[cat]) for cat in results)
        inactive_users = self.find_inactive_users()

        report_lines.append("## Summary Statistics")
        report_lines.append("")
        report_lines.append(
            f"- **Total users in directory:** {len(self.users_by_nickname)}"
        )
        report_lines.append(
            f"- **Total email accounts with activity:** {total_accounts}"
        )
        report_lines.append(f"- **User accounts:** {len(results['user'])}")
        report_lines.append(f"- **Bot accounts:** {len(results['bot'])}")
        report_lines.append(f"- **Group accounts:** {len(results['group'])}")
        report_lines.append(f"- **Unknown accounts:** {len(results['unknown'])}")
        report_lines.append(
            f"- **Inactive users (no email activity):** {len(inactive_users)}"
        )
        report_lines.append("")

        # Detailed account classifications
        for category_name, accounts in results.items():
            if not accounts:
                continue

            # Customize titles for different account types
            if category_name == "user":
                title = f"User Accounts with Email Activity ({len(accounts)})"
            elif category_name == "bot":
                title = f"Bot Accounts ({len(accounts)})"
            elif category_name == "group":
                title = f"Group Accounts with LLM-Enhanced Membership ({len(accounts)})"
            else:
                title = f"{category_name.capitalize()} Accounts ({len(accounts)})"

            report_lines.append(f"## {title}")
            report_lines.append("")

            for acc in accounts:
                # For user accounts, show all aliases if available
                header_text = acc.email
                if category_name == "user" and acc.user_info:
                    # Check if this user has multiple aliases
                    primary_nickname = acc.user_info.mail_nickname
                    if (
                        primary_nickname in self.user_aliases
                        and len(self.user_aliases[primary_nickname]) > 1
                    ):
                        # Show all aliases in the header
                        all_aliases = self.user_aliases[primary_nickname]
                        header_text = " and ".join(all_aliases)

                report_lines.append(f"### {header_text}")
                report_lines.append(
                    f"**Activity:** {acc.frequency} occurrences ({acc.emails_sent} sent, {acc.emails_received} received)"
                )
                report_lines.append("")

                if acc.user_info:
                    if acc.user_info.display_name:
                        report_lines.append(
                            f"- **Display Name:** {acc.user_info.display_name}"
                        )
                    if acc.user_info.job_title:
                        report_lines.append(
                            f"- **Job Title:** {acc.user_info.job_title}"
                        )
                    if acc.user_info.department:
                        report_lines.append(
                            f"- **Department:** {acc.user_info.department}"
                        )
                    if acc.user_info.company_name:
                        report_lines.append(
                            f"- **Company:** {acc.user_info.company_name}"
                        )
                    if acc.user_info.manager:
                        report_lines.append(f"- **Manager:** {acc.user_info.manager}")
                    if acc.user_info.office_location:
                        report_lines.append(
                            f"- **Office Location:** {acc.user_info.office_location}"
                        )

                if acc.group_meaning:
                    report_lines.append(f"- **Purpose:** {acc.group_meaning}")

                if acc.potential_members:
                    method = "LLM-suggested"
                    report_lines.append(
                        f"- **{method} members ({len(acc.potential_members)}):** {', '.join(acc.potential_members)}"
                    )

                report_lines.append("")

        # Save report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"Comprehensive analysis complete! Report saved to: {output_file}")

        # Console summary
        logger.info("QUICK SUMMARY:")
        logger.info(f"  Total accounts: {total_accounts}")
        logger.info(f"  User accounts: {len(results['user'])}")
        logger.info(f"  Bot accounts: {len(results['bot'])}")
        logger.info(f"  Group accounts: {len(results['group'])}")
        logger.info(f"  Unknown accounts: {len(results['unknown'])}")

    def get_current_date(self) -> str:
        """Get current date for reporting"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_user_email_mapping(
        self, output_file: str = "results/user_email_mapping.json"
    ) -> None:
        """Generate comprehensive user email mapping including group memberships"""
        if not self.emails_data or not self.users_by_nickname:
            self.load_data()

        logger.info("Creating comprehensive user email mapping...")

        # Create comprehensive user email mapping
        user_email_data = {}

        # Process individual users and their email aliases
        for nickname, user_data in self.users_by_nickname.items():
            if user_data:
                # Determine account type
                account_type = "user"  # Default to user
                if hasattr(user_data, "job_title") and user_data.job_title:
                    job_title_lower = user_data.job_title.lower()
                    if (
                        "bot" in job_title_lower
                        or "ci" in job_title_lower
                        or "automation" in job_title_lower
                    ):
                        account_type = "bot"

                # Check if this is actually a bot based on nickname patterns
                if any(
                    pattern in nickname.lower()
                    for pattern in [
                        "bot",
                        "ci.",
                        "alert",
                        "monitor",
                        "jenkins",
                        "automation",
                        "alerts",
                        "reporter",
                        "vulnerability",
                        "optimizer",
                        "pipeline",
                    ]
                ):
                    account_type = "bot"

                # Get display name as unique identifier
                display_name = getattr(user_data, "display_name", nickname)

                # If this person already exists, add this email alias
                if display_name in user_email_data:
                    if nickname not in user_email_data[display_name]["email_aliases"]:
                        user_email_data[display_name]["email_aliases"].append(nickname)
                else:
                    # Create new user entry
                    user_email_data[display_name] = {
                        "account_type": account_type,
                        "primary_email": nickname,
                        "email_aliases": [nickname],
                        "personal_info": {
                            "display_name": display_name,
                            "job_title": getattr(user_data, "job_title", ""),
                            "department": getattr(user_data, "department", ""),
                            "company": getattr(user_data, "company_name", ""),
                            "manager": getattr(user_data, "manager", ""),
                            "office_location": getattr(
                                user_data, "office_location", ""
                            ),
                        },
                        "group_memberships": [],
                    }

        # Sort email aliases for each user
        for user_key in user_email_data:
            user_email_data[user_key]["email_aliases"].sort()

        logger.info(f"Processed {len(user_email_data)} unique users/bots")

        # Get all group accounts for analysis
        all_groups = [
            "devops.team",
            "devops-team",
            "devops.leads",
            "devteam",
            "security.ops",
            "security.review",
            "urgent-security",
            "pm.team",
            "quantum.support",
            "it.support",
            "emb-sys-leads",
            "compliance.team",
            "compliance-team",
            "dbadmin.team",
            "db.admin",
            "engineering.director",
            "architecture.board",
            "arch.review.team",
            "platform.ops",
            "ml-ops-team",
            "network-ops",
        ]

        logger.info(f"Analyzing {len(all_groups)} group accounts")

        # Group purposes for LLM analysis
        group_purposes = {
            "devops.team": "Development Operations team responsible for CI/CD, infrastructure, and deployment",
            "devops-team": "Development Operations team responsible for CI/CD, infrastructure, and deployment",
            "devops.leads": "Technical leads or team leadership group",
            "devteam": "Team distribution list for devteam group",
            "security.ops": "Security team responsible for cybersecurity, compliance, and threat management",
            "security.review": "Security team responsible for cybersecurity, compliance, and threat management",
            "urgent-security": "Security team responsible for cybersecurity, compliance, and threat management",
            "pm.team": "Product Management team",
            "quantum.support": "Technical support team for user assistance",
            "it.support": "Technical support team for user assistance",
            "emb-sys-leads": "Technical leads or team leadership group",
            "compliance.team": "Compliance and regulatory team",
            "compliance-team": "Compliance and regulatory team",
            "dbadmin.team": "Database administrator team",
            "db.admin": "Database administrator group",
            "engineering.director": "Engineering team for software development",
            "architecture.board": "Architecture review team for technical design",
            "arch.review.team": "Architecture review team for technical design",
            "platform.ops": "Platform operations team for infrastructure and platform engineering",
            "ml-ops-team": "Machine Learning operations team for ML infrastructure and model deployment",
            "network-ops": "Operations team responsible for system operations and maintenance",
        }

        # Create mapping from email aliases to user display names
        email_to_user = {}
        for display_name, user_info in user_email_data.items():
            for email_alias in user_info["email_aliases"]:
                email_to_user[email_alias] = display_name

        # Process each group
        logger.info("Analyzing group memberships with LLM...")
        for group_email in all_groups:
            purpose = group_purposes.get(
                group_email, "General team or distribution list"
            )

            try:
                # Get LLM analysis for this group
                members = self.find_group_members_llm(group_email, purpose)

                # Add group membership to relevant users
                for member_email in members:
                    if member_email in email_to_user:
                        user_display_name = email_to_user[member_email]
                        if (
                            group_email
                            not in user_email_data[user_display_name][
                                "group_memberships"
                            ]
                        ):
                            user_email_data[user_display_name][
                                "group_memberships"
                            ].append(group_email)

                logger.info(f"  {group_email}: {len(members)} members")

            except Exception as e:
                logger.error(f"  Error analyzing {group_email}: {e}")

        # Sort group memberships for each user
        for user_key in user_email_data:
            user_email_data[user_key]["group_memberships"].sort()

        # Create final JSON structure
        final_data = {
            "metadata": {
                "generated_date": self.get_current_date().split()[
                    0
                ],  # Just the date part
                "total_users": len(
                    [u for u in user_email_data.values() if u["account_type"] == "user"]
                ),
                "total_bots": len(
                    [u for u in user_email_data.values() if u["account_type"] == "bot"]
                ),
                "total_groups_analyzed": len(all_groups),
                "description": "Comprehensive mapping of individual users/bots with their email aliases and group memberships",
            },
            "users": user_email_data,
        }

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\nGenerated comprehensive user email mapping: {output_file}")
        logger.info(f'Total users: {final_data["metadata"]["total_users"]}')
        logger.info(f'Total bots: {final_data["metadata"]["total_bots"]}')
        logger.info(
            f'Total groups analyzed: {final_data["metadata"]["total_groups_analyzed"]}'
        )


def generate_comprehensive_report(
    emails_file: str = "data/emails.config.json",
    users_file: str = "data/users.config.json",
    output_file: str = "results/comprehensive_account_analysis.md",
):
    """Generate comprehensive account classification report.

    Args:
        emails_file: Path to the emails JSON file
        users_file: Path to the users JSON file
        output_file: Path to the output markdown report file
    """
    classifier = EmailAccountClassifier(emails_file=emails_file, users_file=users_file)
    classifier.generate_comprehensive_report(output_file=output_file)
    logger.info(f"Report generated: {output_file}")


def list_by_type(
    account_type: str = "all",
    emails_file: str = "data/emails.config.json",
    users_file: str = "data/users.config.json",
):
    """List accounts by type (user, bot, group, or all).

    Args:
        account_type: Type of accounts to list (user, bot, group, or all)
        emails_file: Path to the emails JSON file
        users_file: Path to the users JSON file
    """
    classifier = EmailAccountClassifier(emails_file=emails_file, users_file=users_file)
    results = classifier.classify_all_accounts()

    if account_type.lower() == "all":
        for type_name, accounts in results.items():
            logger.info(f"\n{type_name.upper()}:")
            for account in accounts:
                logger.info(f"  {account.email}")
    elif account_type.lower() in results:
        accounts = results[account_type.lower()]
        logger.info(f"\n{account_type.upper()}:")
        for account in accounts:
            logger.info(f"  {account.email}")
    else:
        logger.error(f"Invalid account type: {account_type}")
        logger.info("Valid types: user, bot, group, all")


def generate_user_email_mapping(
    emails_file: str = "data/emails.config.json",
    users_file: str = "data/users.config.json",
    output_file: str = "results/user_email_mapping.json",
):
    """Generate comprehensive user email mapping with group memberships.

    Args:
        emails_file: Path to the emails JSON file
        users_file: Path to the users JSON file
        output_file: Path to the output JSON mapping file
    """
    classifier = EmailAccountClassifier(emails_file=emails_file, users_file=users_file)
    classifier.generate_user_email_mapping(output_file=output_file)
    logger.info(f"User email mapping generated: {output_file}")


if __name__ == "__main__":
    fire.Fire()
