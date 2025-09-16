#!/usr/bin/env python3
"""
Individual Email Account Extraction Module

This module extracts emails for individual accounts without merging or grouping. Each email account
is treated as completely independent - no alias merging, no group expansion. It reads the raw user
directory (users.config.json) and email database (emails.config.json) directly.

Key Differences from Merged Accounts Module:
- Each account is treated independently (no merging of aliases)
- Group accounts are standalone entities (not expanded to members)
- Uses users.config.json directly (no preprocessing required)
- Simple 1:1 account-to-email mapping
- Emails within sent/received groups are sorted chronologically by timestamp

Key Features:
- Chronological ordering: Emails within sent and received groups are sorted by timestamp (earliest to latest)
- Comprehensive email metadata: Preserves all email fields including attachments, importance, folders
- Robust timestamp handling: Handles missing or malformed timestamps gracefully
- Detailed logging: Provides comprehensive extraction progress and statistics
- Prompt generation: Can generate filled prompt templates with extracted email data for LLM evaluation

Usage:
    python email_extractor_individual_accounts.py extract_emails john.doe
    python email_extractor_individual_accounts.py extract_emails devops.leads
    python email_extractor_individual_accounts.py extract_emails_with_prompt marion.chen
    python email_extractor_individual_accounts.py extract_emails_with_prompt marion.chen --prompt_output="results/marion_utterance_prompt.md"
    python email_extractor_individual_accounts.py list_accounts
"""

import json
import os
import logging
import fire
from typing import Dict, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class IndividualEmailExtractor:
    """Extract emails for individual accounts without merging or grouping"""

    def __init__(
        self,
        emails_file: str,
        users_file: str,
    ):
        """Initialize the individual email extractor

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

    def find_account_info(self, target_account: str) -> Optional[Dict]:
        """Find account information in users directory

        Args:
            target_account: Email account to search for

        Returns:
            Account information dict if found, None otherwise
        """
        if not self.users_data:
            self.load_data()

        target_account_lower = target_account.lower()

        # Search through all accounts to find exact match
        for account_info in self.users_data or []:
            account_name = account_info.get("MailNickName", "")
            if account_name.lower() == target_account_lower:
                return {
                    "account_name": account_name,
                    "account_info": account_info,
                    "description": account_info.get("Description", ""),
                    "tags": account_info.get("Tags", []),
                }

        return None

    def extract_emails_for_account(
        self, target_account: str, output_file: Optional[str] = None
    ) -> Dict:
        """Extract all emails sent and received by a specific individual account.

        Args:
            target_account: Email account to extract emails for (exact match)
            output_file: Optional path to save JSON results

        Returns:
            Dict containing all emails for the account
        """
        logger.info(f"Extracting emails for individual account: {target_account}")

        # Load data if not already loaded
        if not self.emails_data or not self.users_data:
            self.load_data()

        # Find account information
        account_info = self.find_account_info(target_account)
        if not account_info:
            # Create minimal result for unknown account
            result = {
                "target_account": target_account,
                "extracted_date": datetime.now().strftime("%Y-%m-%d"),
                "statistics": {"total_sent": 0, "total_received": 0, "total_emails": 0},
                "sent_emails": [],
                "received_emails": [],
                "error": f"Account '{target_account}' not found in users directory",
            }
            logger.warning(f"❌ Account {target_account} not found in users directory")
            logger.info(f"📊 Email extraction completed for account: {target_account}")
            logger.info(f"📧 Total emails extracted: 0")
            logger.info(f"📤 Sent emails: 0")
            logger.info(f"📥 Received emails: 0")

            # Generate default output file name if not provided
            if not output_file:
                output_file = f"results/{target_account}_emails.json"
                logger.info(
                    f"💾 No output file specified, using default: {output_file}"
                )

            self._save_results(result, output_file)
            return result

        logger.info(f"Found account: {account_info['account_name']}")
        logger.info(f"Description: {account_info['description']}")

        # Extract emails (exact account match only)
        sent_emails = []
        received_emails = []

        for email in self.emails_data or []:
            # Get email metadata
            sender = email.get("Sender", "")
            to_recipients = email.get("ToRecipients", [])
            cc_recipients = email.get("CcRecipients", [])
            bcc_recipients = email.get("BccRecipients", [])

            # Extract recipient email addresses
            to_emails = self._extract_recipient_emails(to_recipients)
            cc_emails = self._extract_recipient_emails(cc_recipients)
            bcc_emails = self._extract_recipient_emails(bcc_recipients)
            all_recipients = to_emails + cc_emails + bcc_emails

            # Check if this account sent the email (exact match)
            if sender.lower() == target_account.lower():
                sent_emails.append(
                    self._create_email_record(
                        email, "sent", sender, to_emails, cc_emails, bcc_emails
                    )
                )

            # Check if this account received the email (exact match)
            if target_account.lower() in [r.lower() for r in all_recipients]:
                received_emails.append(
                    self._create_email_record(
                        email,
                        "received",
                        sender,
                        to_emails,
                        cc_emails,
                        bcc_emails,
                        [target_account],
                    )
                )

        # Sort emails by timestamp (earliest to latest)
        logger.info(f"📅 Sorting {len(sent_emails)} sent emails by timestamp...")
        sent_emails.sort(
            key=lambda email: email.get("timestamp", "1900-01-01T00:00:00Z")
        )

        logger.info(
            f"📅 Sorting {len(received_emails)} received emails by timestamp..."
        )
        received_emails.sort(
            key=lambda email: email.get("timestamp", "1900-01-01T00:00:00Z")
        )

        # Create result structure
        result = {
            "target_account": target_account,
            "description": account_info["description"],
            "tags": account_info["tags"],
            "extracted_date": datetime.now().strftime("%Y-%m-%d"),
            "statistics": {
                "total_sent": len(sent_emails),
                "total_received": len(received_emails),
                "total_emails": len(sent_emails) + len(received_emails),
            },
            "sent_emails": sent_emails,
            "received_emails": received_emails,
        }

        # Log summary information
        total_emails = len(sent_emails) + len(received_emails)
        logger.info(f"📊 Email extraction completed for account: {target_account}")
        logger.info(f"📧 Total emails extracted: {total_emails}")
        logger.info(f"📤 Sent emails: {len(sent_emails)}")
        logger.info(f"📥 Received emails: {len(received_emails)}")
        if account_info["description"]:
            logger.info(f"📝 Description: {account_info['description']}")

        # Generate default output file name if not provided
        if not output_file:
            output_file = f"results/{target_account}_emails.json"
            logger.info(f"💾 No output file specified, using default: {output_file}")

        # Save to file
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
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Email extraction results saved to: {output_file}")
            logger.info(f"✅ Extraction completed successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to save results to {output_file}: {str(e)}")
            raise

    def _generate_prompt_with_data(
        self, email_data: Dict, prompt_template_file: str, output_file: str
    ) -> None:
        """Generate a prompt file with email data filled into the template

        Args:
            email_data: Email extraction results dictionary
            prompt_template_file: Path to the prompt template file
            output_file: Path to save the generated prompt
        """
        try:
            # Read the prompt template
            if not os.path.exists(prompt_template_file):
                raise FileNotFoundError(
                    f"✗ Prompt template file not found: {prompt_template_file}"
                )

            with open(prompt_template_file, "r", encoding="utf-8") as f:
                template_content = f.read()

            # Prepare the email data for insertion - simplified structure
            email_account = email_data["target_account"]

            # Create simplified email data with only statistics and emails
            simplified_email_data = {
                "statistics": email_data["statistics"],
                "sent_emails": email_data["sent_emails"],
                "received_emails": email_data["received_emails"],
            }

            email_json_data = json.dumps(
                simplified_email_data, indent=2, ensure_ascii=False
            )

            # Replace placeholders in the template
            filled_prompt = template_content.replace("{email_account}", email_account)
            filled_prompt = filled_prompt.replace("{email_json_data}", email_json_data)

            # Create output directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save the filled prompt
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(filled_prompt)

            logger.info(f"📝 Generated prompt file: {output_file}")
            logger.info(f"📧 Email account: {email_account}")
            logger.info(
                f"📊 Total emails included: {email_data['statistics']['total_emails']}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to generate prompt file: {str(e)}")
            raise

    def extract_emails_with_prompt(
        self,
        target_account: str,
        output_file: Optional[str] = None,
        prompt_template: str = "prompt_gen_complex_utterance.md",
        prompt_output: Optional[str] = None,
    ) -> Dict:
        """Extract emails and generate a prompt file with the data filled in

        Args:
            target_account: Email account to extract emails for (exact match)
            output_file: Optional path to save JSON results
            prompt_template: Path to the prompt template file
            prompt_output: Optional path to save the generated prompt (defaults to results/{account}_prompt.md)

        Returns:
            Email extraction results
        """
        logger.info(
            f"🚀 Extracting emails and generating prompt for account: {target_account}"
        )

        # Extract emails first
        email_data = self.extract_emails_for_account(
            target_account=target_account, output_file=output_file
        )

        # Generate default prompt output file if not provided
        if not prompt_output:
            prompt_output = f"results/{target_account}_prompt.md"
            logger.info(
                f"📝 No prompt output file specified, using default: {prompt_output}"
            )

        # Generate the prompt with data
        self._generate_prompt_with_data(email_data, prompt_template, prompt_output)

        return email_data

    def list_accounts(self) -> None:
        """List all available individual accounts from users directory"""
        if not self.users_data:
            self.load_data()

        logger.info("\n=== Available Individual Accounts for Email Extraction ===\n")

        users = self.users_data if self.users_data else []
        user_accounts = []

        for account_info in users:
            account_name = account_info.get("MailNickName", "")
            description = account_info.get("Description", "")
            tags = account_info.get("Tags", [])

            account_summary = {
                "name": account_name,
                "description": description,
                "tags": tags,
            }

            user_accounts.append(account_summary)

        # Sort alphabetically
        user_accounts.sort(key=lambda x: x["name"])

        def print_account_group(title: str, accounts: List[Dict], color_emoji: str):
            if accounts:
                logger.info(f"{color_emoji} {title} ({len(accounts)}):")
                logger.info("=" * 50)
                for account in accounts:
                    tags_str = ", ".join(account["tags"]) if account["tags"] else "None"
                    logger.info(f"• {account['name']}")
                    logger.info(
                        f"  Description: {account['description'] or 'No description'}"
                    )
                    logger.info(f"  Tags: {tags_str}")
                    logger.info("")

        print_account_group("INDIVIDUAL ACCOUNTS", user_accounts, "👤")

        total_accounts = len(user_accounts)
        logger.info(f"Total individual accounts available: {total_accounts}")

        logger.info("\n📋 Usage examples:")
        if user_accounts:
            first_user = user_accounts[0]["name"]
            logger.info(
                f"  python email_extractor_individual_accounts.py extract_emails {first_user}"
            )
            logger.info(
                f"  python email_extractor_individual_accounts.py extract_emails {first_user} --output_file='results/{first_user}_individual_emails.json'"
            )
            logger.info(
                f"  python email_extractor_individual_accounts.py extract_emails_with_prompt {first_user}"
            )
            logger.info(
                f"  python email_extractor_individual_accounts.py extract_emails_with_prompt {first_user} --prompt_output='results/{first_user}_utterances.md'"
            )

        logger.info(
            "\n💡 Note: Each account is treated independently - no merging or group expansion."
        )
        logger.info(
            "💡 Use extract_emails_with_prompt to generate filled prompt templates for LLM evaluation."
        )


# Command-line interface functions
def extract_emails(
    target_account: str,
    emails_file: str = "data/emails.config.json",
    users_file: str = "data/users.config.json",
    output_file: Optional[str] = None,
):
    """Extract all emails for a specific individual account (no merging or grouping).

    Args:
        target_account: Email account to extract emails for (exact match)
        emails_file: Path to the emails JSON file
        users_file: Path to the users directory JSON file
        output_file: Optional path to save JSON results
    """
    extractor = IndividualEmailExtractor(emails_file=emails_file, users_file=users_file)
    extractor.extract_emails_for_account(
        target_account=target_account, output_file=output_file
    )


def extract_emails_with_prompt(
    target_account: str,
    emails_file: str = "data/emails.config.json",
    users_file: str = "data/users.config.json",
    output_file: Optional[str] = None,
    prompt_template: str = "prompt_email_gen_complex_utterance.md",
    prompt_output: Optional[str] = None,
):
    """Extract emails and generate a prompt file with the email data filled in.

    Args:
        target_account: Email account to extract emails for (exact match)
        emails_file: Path to the emails JSON file
        users_file: Path to the users directory JSON file
        output_file: Optional path to save JSON results
        prompt_template: Path to the prompt template file (default: prompt_gen_complex_utterance.md)
        prompt_output: Optional path to save the generated prompt (default: results/{account}_prompt.md)
    """
    extractor = IndividualEmailExtractor(emails_file=emails_file, users_file=users_file)
    extractor.extract_emails_with_prompt(
        target_account=target_account,
        output_file=output_file,
        prompt_template=prompt_template,
        prompt_output=prompt_output,
    )


def list_accounts(
    emails_file: str = "data/emails.config.json",
    users_file: str = "data/users.config.json",
):
    """List all available individual accounts from users directory.

    Args:
        emails_file: Path to the emails JSON file
        users_file: Path to the users directory JSON file
    """
    extractor = IndividualEmailExtractor(emails_file=emails_file, users_file=users_file)
    extractor.list_accounts()


if __name__ == "__main__":
    fire.Fire()
