#!/usr/bin/env python3
"""
Email Extraction Module

This module rapidly collects all sent and received emails for a specified user. It reads the email
database (emails.config.json) and a precomputed user-to-account mapping (user_email_mapping.json)
that links a person’s individual addresses (e.g., mchen, marion.chen) and related group accounts
including the person. The module can retrieve all emails of the person (a list of persons) across
all of the person's accessible accounts (including group accounts).

Usage:
    python email_extractor.py extract_emails john.doe
    python email_extractor.py extract_emails marion.chen --output_file="marion_emails.json"
    python email_extractor.py list_accounts
"""

import json
import os
import logging
import fire
from typing import Dict, List, Optional, Set
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EmailExtractor:
    """Fast email extraction without LLM dependency"""

    def __init__(
        self,
        emails_file: str,
        user_mapping_file: str,
    ):
        """Initialize the email extractor

        Args:
            emails_file: Path to the emails database JSON file
            user_mapping_file: Path to the user email mapping JSON file (output from email_account_analysis.py)
        """
        self.emails_file = emails_file
        self.user_mapping_file = user_mapping_file
        self.emails_data = None
        self.user_mapping_data = None

    def load_data(self) -> None:
        """Load emails database and user mapping data"""
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

        # Load user mapping
        if not os.path.exists(self.user_mapping_file):
            raise FileNotFoundError(
                f"✗ Error: Could not find user mapping file: {self.user_mapping_file}\n"
                f"Please run email_account_analysis.py generate_user_email_mapping first to create this file."
            )

        try:
            with open(self.user_mapping_file, "r", encoding="utf-8") as f:
                self.user_mapping_data = json.load(f)
            logger.info(f"Loaded user mapping data from {self.user_mapping_file}")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"✗ Error: Invalid JSON in user mapping file {self.user_mapping_file}: {e}"
            )

    def find_user_by_email(self, target_email: str) -> Optional[Dict]:
        """Find user information by any of their email aliases

        Args:
            target_email: Email address to search for

        Returns:
            User information dict if found, None otherwise
        """
        if not self.user_mapping_data:
            self.load_data()

        target_email_lower = target_email.lower()

        # Search through all users to find one with matching email alias
        for user_name, user_info in self.user_mapping_data.get("users", {}).items():
            email_aliases = user_info.get("email_aliases", [])
            primary_email = user_info.get("primary_email", "")

            # Check if target email matches any alias or primary email
            if target_email_lower == primary_email.lower() or target_email_lower in [
                alias.lower() for alias in email_aliases
            ]:
                return {
                    "user_name": user_name,
                    "primary_email": primary_email,
                    "email_aliases": email_aliases,
                    "group_memberships": user_info.get("group_memberships", []),
                    "account_type": user_info.get("account_type", "user"),
                    "personal_info": user_info.get("personal_info", {}),
                }

        return None

    def extract_emails_for_account(
        self, target_account: str, output_file: Optional[str] = None
    ) -> Dict:
        """Extract all emails sent and received by a specific account, including group emails.

        Args:
            target_account: Email account to extract emails for
            output_file: Optional path to save JSON results

        Returns:
            Dict containing all emails for the account
        """
        logger.info(f"Extracting emails for account: {target_account}")

        # Load data if not already loaded
        if not self.emails_data or not self.user_mapping_data:
            self.load_data()

        # Find user information
        user_info = self.find_user_by_email(target_account)
        if not user_info:
            # Create minimal result for unknown account
            result = {
                "target_account": target_account,
                "account_aliases": [target_account],
                "group_memberships": [],
                "extracted_date": datetime.now().strftime("%Y-%m-%d"),
                "statistics": {"total_sent": 0, "total_received": 0, "total_emails": 0},
                "sent_emails": [],
                "received_emails": [],
                "error": f"Account '{target_account}' not found in user mapping",
            }
            logger.warning(f"Account {target_account} not found in user mapping")

            if output_file:
                self._save_results(result, output_file)
            return result

        # Get all email identifiers for this account
        account_aliases = user_info["email_aliases"]
        group_memberships = user_info["group_memberships"]
        all_account_identifiers = set(account_aliases + group_memberships)

        logger.info(f"Found account with aliases: {account_aliases}")
        logger.info(f"Group memberships: {group_memberships}")
        logger.info(f"Searching for emails involving: {all_account_identifiers}")

        # Extract emails
        sent_emails = []
        received_emails = []

        for email in self.emails_data or []:
            # Get email metadata
            email_id = email.get("EmailId")
            sender = email.get("Sender", "")
            to_recipients = email.get("ToRecipients", [])
            cc_recipients = email.get("CcRecipients", [])
            bcc_recipients = email.get("BccRecipients", [])

            # Extract recipient email addresses
            to_emails = self._extract_recipient_emails(to_recipients)
            cc_emails = self._extract_recipient_emails(cc_recipients)
            bcc_emails = self._extract_recipient_emails(bcc_recipients)
            all_recipients = to_emails + cc_emails + bcc_emails

            # Check if this account sent the email
            if sender.lower() in [
                identifier.lower() for identifier in all_account_identifiers
            ]:
                sent_emails.append(
                    self._create_email_record(
                        email, "sent", sender, to_emails, cc_emails, bcc_emails
                    )
                )

            # Check if this account received the email (direct or via group)
            matching_recipients = [
                r
                for r in all_recipients
                if r.lower()
                in [identifier.lower() for identifier in all_account_identifiers]
            ]
            if matching_recipients:
                received_emails.append(
                    self._create_email_record(
                        email,
                        "received",
                        sender,
                        to_emails,
                        cc_emails,
                        bcc_emails,
                        matching_recipients,
                    )
                )

        # Create result structure
        result = {
            "target_account": target_account,
            "account_aliases": account_aliases,
            "group_memberships": group_memberships,
            "extracted_date": datetime.now().strftime("%Y-%m-%d"),
            "statistics": {
                "total_sent": len(sent_emails),
                "total_received": len(received_emails),
                "total_emails": len(sent_emails) + len(received_emails),
            },
            "sent_emails": sent_emails,
            "received_emails": received_emails,
        }

        logger.info(
            f"Extracted {len(sent_emails)} sent emails and {len(received_emails)} received emails"
        )

        # Save to file if specified
        if output_file:
            self._save_results(result, output_file)

        return result

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

    def _create_email_record(
        self,
        email: Dict,
        email_type: str,
        sender: str,
        to_emails: List[str],
        cc_emails: List[str],
        bcc_emails: List[str],
        matched_accounts: Optional[List[str]] = None,
    ) -> Dict:
        """Create standardized email record"""
        record = {
            "email_id": email.get("EmailId"),
            "email_type": email_type,
            "timestamp": email.get("Timestamp"),
            "subject": email.get("Subject"),
            "sender": sender,
            "to_recipients": to_emails,
            "cc_recipients": cc_emails,
            "bcc_recipients": bcc_emails,
            "body": email.get("Body", ""),
            "attachments": email.get("Attachments", []),
            "importance": email.get("Importance"),
            "folder": email.get("Folder"),
            "email_action": email.get("EmailAction"),
            "reference_email_id": email.get("ReferenceEmailId"),
        }

        if email_type == "sent":
            record["matched_account"] = sender
        else:
            record["matched_accounts"] = matched_accounts or []

        return record

    def _save_results(self, result: Dict, output_file: str) -> None:
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Email extraction results saved to: {output_file}")

    def list_accounts(self) -> None:
        """List all available accounts that can be used for email extraction"""
        if not self.user_mapping_data:
            self.load_data()

        print("\n=== Available Accounts for Email Extraction ===\n")

        users = (
            self.user_mapping_data.get("users", {}) if self.user_mapping_data else {}
        )
        user_accounts = []
        bot_accounts = []

        for user_name, user_info in users.items():
            account_type = user_info.get("account_type", "user")
            email_aliases = user_info.get("email_aliases", [])
            group_memberships = user_info.get("group_memberships", [])

            account_summary = {
                "name": user_name,
                "aliases": email_aliases,
                "groups": group_memberships,
                "total_access": len(email_aliases) + len(group_memberships),
            }

            if account_type == "bot":
                bot_accounts.append(account_summary)
            else:
                user_accounts.append(account_summary)

        # Sort by number of total accessible accounts (aliases + groups)
        user_accounts.sort(key=lambda x: x["total_access"], reverse=True)
        bot_accounts.sort(key=lambda x: x["total_access"], reverse=True)

        print(f"USER ACCOUNTS ({len(user_accounts)}):")
        print("=" * 50)
        for account in user_accounts:
            aliases_str = ", ".join(account["aliases"])
            groups_str = ", ".join(account["groups"]) if account["groups"] else "None"
            print(f"• {account['name']}")
            print(f"  Email aliases: {aliases_str}")
            print(f"  Group access: {groups_str}")
            print(f"  Total email access: {account['total_access']} accounts")
            print()

        if bot_accounts:
            print(f"BOT ACCOUNTS ({len(bot_accounts)}):")
            print("=" * 50)
            for account in bot_accounts:
                aliases_str = ", ".join(account["aliases"])
                groups_str = (
                    ", ".join(account["groups"]) if account["groups"] else "None"
                )
                print(f"• {account['name']}")
                print(f"  Email aliases: {aliases_str}")
                print(f"  Group access: {groups_str}")
                print(f"  Total email access: {account['total_access']} accounts")
                print()

        print(f"Total accounts available: {len(user_accounts) + len(bot_accounts)}")
        print("\nUsage examples:")
        if user_accounts:
            first_user = user_accounts[0]
            first_alias = (
                first_user["aliases"][0]
                if first_user["aliases"]
                else "email@domain.com"
            )
            print(f"  python email_extractor.py extract_emails {first_alias}")
            print(
                f"  python email_extractor.py extract_emails {first_alias} --output_file='results/{first_alias}_emails.json'"
            )


# Command-line interface functions
def extract_emails(
    target_account: str,
    emails_file: str = "data/emails.config.json",
    user_mapping_file: str = "results/user_email_mapping.json",
    output_file: Optional[str] = None,
):
    """Extract all emails for a specific account, including group emails.

    Args:
        target_account: Email account to extract emails for
        emails_file: Path to the emails JSON file
        user_mapping_file: Path to the user email mapping JSON file
        output_file: Optional path to save JSON results
    """
    extractor = EmailExtractor(
        emails_file=emails_file, user_mapping_file=user_mapping_file
    )
    result = extractor.extract_emails_for_account(
        target_account=target_account, output_file=output_file
    )

    print(f"\n=== Email Extraction Summary ===")
    print(f"Target account: {result['target_account']}")
    print(f"Account aliases: {result['account_aliases']}")
    print(f"Group memberships: {result['group_memberships']}")
    print(f"Extraction date: {result['extracted_date']}")
    print(f"Total emails: {result['statistics']['total_emails']}")
    print(f"  - Sent: {result['statistics']['total_sent']}")
    print(f"  - Received: {result['statistics']['total_received']}")

    if output_file:
        print(f"Results saved to: {output_file}")

    return result


def list_accounts(
    emails_file: str = "data/emails.config.json",
    user_mapping_file: str = "results/user_email_mapping.json",
):
    """List all available accounts that can be used for email extraction.

    Args:
        emails_file: Path to the emails JSON file
        user_mapping_file: Path to the user email mapping JSON file
    """
    extractor = EmailExtractor(
        emails_file=emails_file, user_mapping_file=user_mapping_file
    )
    extractor.list_accounts()


if __name__ == "__main__":
    fire.Fire()
