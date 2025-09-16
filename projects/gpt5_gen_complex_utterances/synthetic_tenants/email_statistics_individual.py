#!/usr/bin/env python3
"""
Individual Email Account Statistics Module

This module collects email statistics for individual accounts without merging or grouping.
Each email account is treated as completely independent - no alias merging, no group expansion.
It provides statistics on how many emails each account has sent and received.

Key Features:
- Each account treated independently (no merging of alias    collector = IndividualEmailStatistics(
        emails_file=emails_file, users_file=users_file
    )

    result = collector.collect_email_statistics(min_emails=0)

    # Print activity summary at the top
    summary = result["summary"]
    print(f"\n📊 ACCOUNT ACTIVITY SUMMARY")
    print("=" * 50)
    print(f"📁 Total unique accounts: {summary['total_unique_accounts']:,}")
    print(f"⚡ Accounts with email activity: {summary['accounts_with_email_activity']:,}")
    print(f"💤 Accounts with no email activity: {summary['accounts_with_no_activity']:,}")
    print("=" * 50)- Group accounts are standalone entities (not expanded to members)
- Uses users.config.json directly (no preprocessing required)
- Simple email count statistics per account
- Detailed breakdowns by account type (user/group/bot/other)

Usage:
    python email_statistics_individual.py collect_statistics
    python email_statistics_individual.py collect_statistics --output_file="results/individual_stats.json"
    python email_statistics_individual.py collect_statistics --min_emails=5
"""

import json
import os
import logging
import fire
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class IndividualEmailStatistics:
    """Collect email statistics for individual accounts without merging or grouping"""

    def __init__(
        self,
        emails_file: str,
        users_file: str,
    ):
        """Initialize the individual email statistics collector

        Args:
            emails_file: Path to the emails database JSON file
            users_file: Path to the users directory JSON file (users.config.json)
        """
        self.emails_file = emails_file
        self.users_file = users_file
        self.emails_data = None
        self.users_data = None

    def load_data(self) -> None:
        """Load emails database and users directory"""
        # Load emails database
        if not os.path.exists(self.emails_file):
            raise FileNotFoundError(
                f"✗ Error: Could not find emails file: {self.emails_file}"
            )

        try:
            with open(self.emails_file, "r", encoding="utf-8") as f:
                self.emails_data = json.load(f)
            logger.info(
                f"Loaded {len(self.emails_data)} emails from {self.emails_file}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"✗ Error: Invalid JSON in emails file {self.emails_file}: {e}"
            )

        # Load users directory
        if not os.path.exists(self.users_file):
            raise FileNotFoundError(
                f"✗ Error: Could not find users file: {self.users_file}"
            )

        try:
            with open(self.users_file, "r", encoding="utf-8") as f:
                self.users_data = json.load(f)
            logger.info(f"Loaded users directory from {self.users_file}")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"✗ Error: Invalid JSON in users file {self.users_file}: {e}"
            )

    def _extract_recipient_emails(self, recipients: List) -> List[str]:
        """Extract email addresses from recipient list"""
        emails = []
        if isinstance(recipients, list):
            for recipient in recipients:
                if isinstance(recipient, dict):
                    email = recipient.get("Recipient", "")
                    if email:
                        emails.append(email)
                elif isinstance(recipient, str):
                    emails.append(recipient)
        return emails

    def get_account_type(self, account_name: str) -> str:
        """Get account type from users directory"""
        if not self.users_data:
            return "unknown"

        # Handle both dictionary and list formats
        if isinstance(self.users_data, dict):
            account_info = self.users_data.get(account_name, {})
            return account_info.get("Type", "unknown").lower()
        elif isinstance(self.users_data, list):
            # Search for account in list format
            for user in self.users_data:
                if isinstance(user, dict):
                    # Use MailNickName as primary field for individual accounts
                    user_email = user.get("MailNickName")
                    if user_email == account_name:
                        return user.get("Type", "unknown").lower()
            return "unknown"
        else:
            return "unknown"

    def get_account_info(self, account_name: str) -> Dict:
        """Get account information from users directory"""
        if not self.users_data:
            return {}

        # Handle both dictionary and list formats
        if isinstance(self.users_data, dict):
            return self.users_data.get(account_name, {})
        elif isinstance(self.users_data, list):
            # Search for account in list format
            for user in self.users_data:
                if isinstance(user, dict):
                    # Use MailNickName as primary field for individual accounts
                    user_email = user.get("MailNickName")
                    if user_email == account_name:
                        return user
            return {}
        else:
            return {}

    def collect_email_statistics(
        self, min_emails: int = 0, output_file: Optional[str] = None
    ) -> Dict:
        """Collect email statistics for all individual accounts

        Args:
            min_emails: Minimum number of emails to include account in results
            output_file: Optional path to save JSON results

        Returns:
            Dict containing email statistics for all accounts
        """
        logger.info("Collecting email statistics for individual accounts...")

        # Load data if not already loaded
        if not self.emails_data or not self.users_data:
            self.load_data()

        # Initialize counters
        sent_counts = defaultdict(int)
        received_counts = defaultdict(int)
        all_accounts = set()

        # Process each email
        for email in self.emails_data or []:
            sender = email.get("Sender", "")
            to_recipients = email.get("ToRecipients", [])
            cc_recipients = email.get("CcRecipients", [])
            bcc_recipients = email.get("BccRecipients", [])

            # Extract recipient email addresses
            to_emails = self._extract_recipient_emails(to_recipients)
            cc_emails = self._extract_recipient_emails(cc_recipients)
            bcc_emails = self._extract_recipient_emails(bcc_recipients)
            all_recipients = to_emails + cc_emails + bcc_emails

            # Count sent emails
            if sender:
                sent_counts[sender] += 1
                all_accounts.add(sender)

            # Count received emails
            for recipient in all_recipients:
                if recipient:
                    received_counts[recipient] += 1
                    all_accounts.add(recipient)

        logger.info(f"Found {len(all_accounts)} unique email accounts in email data")

        # Add all accounts from users directory (including those with zero email activity)
        users_added = 0
        if isinstance(self.users_data, dict):
            logger.info(
                f"Users data is dictionary format with {len(self.users_data)} entries"
            )
            for account_name in self.users_data.keys():
                if account_name not in all_accounts:
                    users_added += 1
                all_accounts.add(account_name)
        elif isinstance(self.users_data, list):
            logger.info(
                f"Users data is list format with {len(self.users_data)} entries"
            )
            # Debug: Check the first user to see available fields
            if self.users_data and isinstance(self.users_data[0], dict):
                sample_keys = list(self.users_data[0].keys())
                logger.info(f"Sample user keys: {sample_keys}")

            for i, user in enumerate(self.users_data):
                if isinstance(user, dict):
                    user_email = user.get(
                        "MailNickName"
                    )  # Primary field for individual accounts
                    if user_email:
                        if user_email not in all_accounts:
                            users_added += 1
                            logger.info(f"Adding inactive user: {user_email}")
                        all_accounts.add(user_email)
                    else:
                        # Debug: Show what fields this user has if no email found
                        if i < 3:  # Only log first 3 for debugging
                            logger.warning(
                                f"User {i} has no MailNickName field: {list(user.keys())}"
                            )
        else:
            logger.warning(f"Users data is unexpected type: {type(self.users_data)}")

        logger.info(f"Added {users_added} users from directory with no email activity")
        logger.info(
            f"Total unique accounts (including zero activity): {len(all_accounts)}"
        )

        # Build account statistics
        account_stats = {}
        user_accounts = []
        group_accounts = []
        bot_accounts = []
        unknown_accounts = []

        accounts_with_email_activity = 0
        accounts_with_no_activity = 0

        for account in all_accounts:
            sent = sent_counts[account]
            received = received_counts[account]
            total = sent + received

            # Track activity statistics
            if total > 0:
                accounts_with_email_activity += 1
            else:
                accounts_with_no_activity += 1

            # Skip accounts below minimum threshold
            if total < min_emails:
                continue

            account_type = self.get_account_type(account)

            # Get additional info from users directory
            account_info = self.get_account_info(account)
            description = account_info.get("Description", "")
            tags = account_info.get("Tags", [])

            stat_entry = {
                "account": account,
                "type": account_type,
                "description": description,
                "tags": tags,
                "emails_sent": sent,
                "emails_received": received,
                "total_emails": total,
                "ratio_sent_to_received": round(
                    sent / received if received > 0 else float("inf"), 2
                ),
                "activity_level": self._classify_activity_level(total),
            }

            account_stats[account] = stat_entry

            # Categorize by type
            if account_type == "user":
                user_accounts.append(stat_entry)
            elif account_type == "group":
                group_accounts.append(stat_entry)
            elif account_type == "bot":
                bot_accounts.append(stat_entry)
            else:
                unknown_accounts.append(stat_entry)

        # Sort by total email count (descending)
        user_accounts.sort(key=lambda x: x["total_emails"], reverse=True)
        group_accounts.sort(key=lambda x: x["total_emails"], reverse=True)
        bot_accounts.sort(key=lambda x: x["total_emails"], reverse=True)
        unknown_accounts.sort(key=lambda x: x["total_emails"], reverse=True)

        # Calculate summary statistics
        total_accounts_with_activity = len(account_stats)
        total_sent = sum(sent_counts.values())
        total_received = sum(received_counts.values())

        # Create result structure
        result = {
            "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "emails_file": self.emails_file,
            "users_file": self.users_file,
            "min_emails_threshold": min_emails,
            "summary": {
                "total_unique_accounts": len(all_accounts),
                "accounts_with_email_activity": accounts_with_email_activity,
                "accounts_with_no_activity": accounts_with_no_activity,
                "accounts_above_threshold": total_accounts_with_activity,
                "total_emails_sent": total_sent,
                "total_emails_received": total_received,
                "total_email_transactions": total_sent,  # Note: sent = received for internal emails
                "account_type_counts": {
                    "users": len(user_accounts),
                    "groups": len(group_accounts),
                    "bots": len(bot_accounts),
                    "unknown": len(unknown_accounts),
                },
            },
            "accounts_by_type": {
                "users": user_accounts,
                "groups": group_accounts,
                "bots": bot_accounts,
                "unknown": unknown_accounts,
            },
            "all_accounts": account_stats,
        }

        logger.info(
            f"Collected statistics for {total_accounts_with_activity} accounts (min {min_emails} emails)"
        )
        logger.info(
            f"Account types: {len(user_accounts)} users, {len(group_accounts)} groups, {len(bot_accounts)} bots, {len(unknown_accounts)} unknown"
        )

        # Save to file if specified
        if output_file:
            self._save_results(result, output_file)

        return result

    def _classify_activity_level(self, total_emails: int) -> str:
        """Classify activity level based on email count"""
        if total_emails >= 100:
            return "very_high"
        elif total_emails >= 50:
            return "high"
        elif total_emails >= 20:
            return "medium"
        elif total_emails >= 5:
            return "low"
        else:
            return "very_low"

    def _save_results(self, result: Dict, output_file: str) -> None:
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Email statistics saved to: {output_file}")

    def print_summary(self, stats: Dict) -> None:
        """Print a formatted summary of email statistics"""
        summary = stats["summary"]
        accounts_by_type = stats["accounts_by_type"]

        print("\n" + "=" * 70)
        print("📊 EMAIL STATISTICS SUMMARY (Individual Accounts)")
        print("=" * 70)

        print(f"📅 Collection Date: {stats['collection_date']}")
        print(f"📧 Total Email Transactions: {summary['total_emails_sent']:,}")
        print(f"👤 Unique Accounts Found: {summary['total_unique_accounts']:,}")
        print(f"⚡ Accounts with Activity: {summary['accounts_with_activity']:,}")
        print(f"🎯 Minimum Emails Threshold: {stats['min_emails_threshold']}")

        print(f"\n📈 ACCOUNT TYPE BREAKDOWN:")
        type_counts = summary["account_type_counts"]
        print(f"  👤 Users: {type_counts['users']:,}")
        print(f"  👥 Groups: {type_counts['groups']:,}")
        print(f"  🤖 Bots: {type_counts['bots']:,}")
        print(f"  ❓ Unknown: {type_counts['unknown']:,}")

        # Top accounts by type
        def print_top_accounts(
            title: str, accounts: List[Dict], emoji: str, limit: int = 10
        ):
            if accounts:
                print(f"\n{emoji} TOP {title.upper()} (by total emails):")
                print("-" * 50)
                for i, account in enumerate(accounts[:limit], 1):
                    ratio_str = (
                        f"{account['ratio_sent_to_received']:.1f}"
                        if account["ratio_sent_to_received"] != float("inf")
                        else "∞"
                    )
                    activity = account["activity_level"].replace("_", " ").title()
                    print(
                        f"{i:2d}. {account['account']:<30} "
                        f"📤{account['emails_sent']:4d} 📥{account['emails_received']:4d} "
                        f"📊{account['total_emails']:4d} "
                        f"⚡{activity}"
                    )

        print_top_accounts("User Accounts", accounts_by_type["users"], "👤")
        print_top_accounts("Group Accounts", accounts_by_type["groups"], "👥")
        print_top_accounts("Bot Accounts", accounts_by_type["bots"], "🤖", 5)

        print(f"\n💡 Legend: 📤 Sent | 📥 Received | 📊 Total | ⚡ Activity Level")
        print("=" * 70)


# Command-line interface functions
def individual_account_statistics(
    emails_file: str = "data/emails.config.json",
    users_file: str = "data/users.config.json",
    output_file: Optional[str] = None,
    limit: int = 20,
):
    """Show individual account email statistics.

    Args:
        emails_file: Path to the emails JSON file
        users_file: Path to the users directory JSON file
        output_file: Optional path to save results to JSON file
        limit: Number of top accounts to display (use 0 or negative number for all accounts)
    """
    collector = IndividualEmailStatistics(
        emails_file=emails_file, users_file=users_file
    )

    result = collector.collect_email_statistics(min_emails=0)

    # Print activity summary at the top
    summary = result["summary"]
    print(f"\n📊 ACCOUNT ACTIVITY SUMMARY")
    print("=" * 50)
    print(f"📁 Total unique accounts: {summary['total_unique_accounts']:,}")
    print(
        f"⚡ Accounts with email activity: {summary['accounts_with_email_activity']:,}"
    )
    print(
        f"💤 Accounts with no email activity: {summary['accounts_with_no_activity']:,}"
    )
    print("=" * 50)

    # Show all accounts sorted by total emails
    all_accounts = []
    for accounts in result["accounts_by_type"].values():
        all_accounts.extend(accounts)

    # Filter to show only accounts with at least 1 email for the table
    accounts_with_activity = [acc for acc in all_accounts if acc["total_emails"] > 0]
    accounts_with_activity.sort(key=lambda x: x["total_emails"], reverse=True)

    # If limit is 0 or negative, show all active accounts
    if limit <= 0:
        accounts_to_show = accounts_with_activity
        title = f"ALL {len(accounts_with_activity)} ACCOUNTS WITH EMAIL ACTIVITY"
    else:
        accounts_to_show = accounts_with_activity[:limit]
        title = f"TOP {limit} ACCOUNTS WITH EMAIL ACTIVITY"

    print(f"\n📊 {title}")
    print("=" * 70)
    print(f"{'Rank':<4} {'Account':<35} {'Sent':<6} {'Rcvd':<6} {'Total'}")
    print("-" * 70)

    for i, account in enumerate(accounts_to_show, 1):
        print(
            f"{i:<4} {account['account']:<35} "
            f"{account['emails_sent']:<6} {account['emails_received']:<6} "
            f"{account['total_emails']:<6}"
        )

    # Save to file if specified
    if output_file:
        # Create simplified account objects without type and activity_level
        simplified_accounts = []
        for account in accounts_to_show:
            simplified_account = {
                "account": account["account"],
                "description": account.get("description", ""),
                "tags": account.get("tags", []),
                "emails_sent": account["emails_sent"],
                "emails_received": account["emails_received"],
                "total_emails": account["total_emails"],
                "ratio_sent_to_received": account["ratio_sent_to_received"],
            }
            simplified_accounts.append(simplified_account)

        top_accounts_result = {
            "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "emails_file": emails_file,
            "users_file": users_file,
            "limit": limit,
            "total_unique_accounts": summary["total_unique_accounts"],
            "accounts_with_email_activity": summary["accounts_with_email_activity"],
            "accounts_with_no_activity": summary["accounts_with_no_activity"],
            "accounts_shown": len(accounts_to_show),
            "title": title,
            "top_accounts": simplified_accounts,
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(top_accounts_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_file}")

    # Don't return anything to avoid Fire printing the result
    return None


if __name__ == "__main__":
    fire.Fire()
