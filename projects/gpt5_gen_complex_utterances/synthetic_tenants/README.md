# Synthetic Tenants - Complex Utterance Generation Suite

A comprehensive toolkit for generating, processing, and validating complex utterances for BizChat Copilot testing using synthetic tenant data. This suite extracts and analyzes emails, files, Teams messages, meetings, and events from synthetic Microsoft 365 tenant data to create realistic test utterances with proper timestamps and metadata.

## Overview

This toolkit processes synthetic tenant data (emails, files, Teams chats, meetings, calendar events) to generate complex natural language queries for testing BizChat Copilot. It provides end-to-end capabilities from data extraction and analysis to utterance generation, validation, and formatting for SEVAL testing.

**Key Capabilities:**
- Email account classification and extraction (individual and merged accounts)
- File access analysis and content extraction
- Teams message and channel analytics
- Meeting and calendar event analysis
- Comprehensive TSV processing and validation
- Timestamp generation for temporal testing
- SEVAL format conversion

---

## 📚 Module Reference

### 1. Email Account Analysis (`email_account_analysis.py`)

**Purpose:** Comprehensive email account classification system that identifies individual users, bots, groups, and unknown accounts using pattern matching and LLM-based analysis.

**Key Features:**
- Extracts all email accounts from emails database
- Classifies accounts as: user, bot, group, or unknown
- Uses LLM to identify group membership
- Generates user-to-email mapping for downstream processing

**Main Commands:**
```bash
# Extract and classify all accounts
python email_account_analysis.py extract_accounts_from_emails

# Generate user email mapping (required for merged account extraction)
python email_account_analysis.py generate_user_email_mapping

# Analyze specific accounts
python email_account_analysis.py classify_remaining_accounts
```

**Output:** JSON files with account classifications and user-email mappings

---

### 2. Individual Email Extractor (`email_extractor_individual_accounts.py`)

**Purpose:** Extracts emails for individual accounts without merging aliases or expanding groups. Each account is treated independently.

**Key Features:**
- Chronological ordering by timestamp
- No alias merging (treats each email address separately)
- Group accounts remain as entities (not expanded to members)
- Comprehensive email metadata preservation
- Prompt generation for LLM-based utterance creation

**Main Commands:**
```bash
# Extract emails for a specific account
python email_extractor_individual_accounts.py extract_emails john.doe

# Extract with prompt generation
python email_extractor_individual_accounts.py extract_emails_with_prompt marion.chen

# List all available accounts
python email_extractor_individual_accounts.py list_accounts
```

**Use Case:** When you need independent account analysis without cross-account relationships

---

### 3. Merged Email Extractor (`email_extractor_merged_accounts.py`)

**Purpose:** Fast email extraction using precomputed user-to-account mappings. Links a person's individual addresses and related group accounts for comprehensive email access.

**Key Features:**
- Uses `user_email_mapping.json` from `email_account_analysis.py`
- Merges aliases (e.g., mchen, marion.chen) into single user view
- Includes group account emails where user is a member
- Rapid extraction without LLM dependency

**Main Commands:**
```bash
# Extract all emails for a user (across all their accounts)
python email_extractor_merged_accounts.py extract_emails john.doe

# Custom output file
python email_extractor_merged_accounts.py extract_emails marion.chen --output_file="marion_emails.json"

# List available users
python email_extractor_merged_accounts.py list_accounts
```

**Prerequisites:** Run `email_account_analysis.py generate_user_email_mapping` first

---

### 4. Email Statistics (`email_statistics_individual.py`)

**Purpose:** Collects email statistics for individual accounts showing sent/received counts without merging or grouping.

**Key Features:**
- Independent account treatment (no alias merging)
- Sent and received email counts per account
- Account type breakdowns (user/group/bot/other)
- Activity summaries and rankings

**Main Commands:**
```bash
# Collect statistics for all accounts
python email_statistics_individual.py collect_statistics

# Filter by minimum email count
python email_statistics_individual.py collect_statistics --min_emails=5

# Custom output
python email_statistics_individual.py collect_statistics --output_file="results/stats.json"
```

**Output:** JSON with comprehensive email activity statistics

---

### 5. Files Extractor (`files_extractor.py`)

**Purpose:** Reads all files in a folder and generates JSON output with folder statistics and individual file contents.

**Key Features:**
- Recursive folder scanning
- Multiple file format support (PDF, Word, Excel, PowerPoint, Text)
- Configurable content preservation
- Comprehensive folder statistics
- Robust error handling

**Main Commands:**
```bash
# Analyze entire folder with subdirectories
python files_extractor.py analyze_folder --input_folder="data" --output_file="folder_analysis.json"

# Exclude subdirectories
python files_extractor.py analyze_folder --input_folder="documents" --include_subdirs=False

# With prompt generation
python files_extractor.py analyze_folder --input_folder="data" --prompt_file="prompt_template.md"
```

**Supported Formats:** PDF, DOCX, XLSX, PPTX, TXT, CSV, JSON, and more

---

### 6. File Permissions Analyzer (`files_individual_extractor.py`)

**Purpose:** Analyzes `files.config.json` to extract file access information for users and channels.

**Key Features:**
- File sharing permissions analysis (view/edit)
- User access pattern statistics
- Ordered by accessible file count
- Handles both shared files and ownership
- Prompt generation for file-based utterances

**Main Commands:**
```bash
# Analyze all file permissions
python files_individual_extractor.py analyze_file_access --config_file="data/files.config.json"

# Get accessible files for specific user
python files_individual_extractor.py get_user_accessible_files --user_email="johndoe"

# Extract file content with prompt generation
python files_individual_extractor.py extract_user_file_content_with_prompt --user_email="johndoe"
```

**Output:** Permission analysis reports and user-specific file extractions

---

### 7. Meetings & Events Analytics (`meetings_events_analytics.py`)

**Purpose:** Specialized analysis tool for Microsoft Teams meetings, calendar events, and transcripts. Focuses on formal meeting discussions for LLM training.

**Key Features:**
- Meeting and event statistics
- Transcript content extraction with speaker analysis
- Meeting-type chat message correlation
- Export in Markdown format optimized for LLM training
- Complex query generation for formal meeting contexts

**Main Commands:**
```bash
# Generate comprehensive meetings/events analytics
python meetings_events_analytics.py generate_meetings_events_analytics --data_folder=data

# Extract meetings for specific participant
python meetings_events_analytics.py extract_meetings_for_complex_queries --participant_filter="alex.khan"

# Custom output
python meetings_events_analytics.py generate_meetings_events_analytics --output_file="results/meetings.md"
```

**Input Requirements:** `onlinemeetings.config.json`, `events.config.json`, `transcripts/*.vtt`, `chats.config.json`

---

### 8. Teams Message Analytics (`teams_message_analytics.py`)

**Purpose:** Comprehensive analysis tool for Microsoft Teams chat and channel data. Focuses on daily collaboration patterns and excludes meeting-type chats from LLM prompt generation.

**Key Features:**
- User messaging statistics (sent/received, conversation participation)
- Message access analysis for specific users
- File reference extraction (URLs, attachments, file paths)
- Export in Markdown format
- Chat type breakdown (Group, OneOnOne, Meeting)
- Top user rankings by various metrics
- Channel message accessibility analysis

**Main Commands:**
```bash
# Generate comprehensive analytics (includes all chat types for statistics)
python teams_message_analytics.py generate_comprehensive_analytics --chats_file="data/chats.config.json"

# Extract user messages with files and prompt (excludes meeting chats)
python teams_message_analytics.py extract_user_teams_messages_with_files_and_prompt --user_email="alex.khan"

# Include channel messages
python teams_message_analytics.py extract_user_teams_messages_with_files_and_prompt --user_email="alex.khan" --include_channel_messages=true
```

**Note:** For meeting-focused queries, use `meetings_events_analytics.py` instead. This separation enables domain-specific optimization.

---

### 9. TSV Processing Suite (`tsv_processor.py`)

**Purpose:** Comprehensive TSV file processing with 8 major capabilities for transformation, validation, and formatting.

**Part 1: TSV Transformation (`process_file`)**
- Remove last column (Complexity_Reason)
- Add 'Segment' column with specified value
- Convert email_account by appending domain suffix
- Add 'Source' column with "synthetic" value
- Supports append mode

```bash
python tsv_processor.py process_file --input_file="input.tsv" --segment="email"
```

**Part 2: LLM TSV to SEVAL Format (`llm_tsv_to_seval_tsv`)**
- Direct conversion from LLM output to SEVAL format
- Adds metadata: Segment, Environment, Grounding Data Source
- Transforms Email_Account → user_id, Query_Timestamp → timestamp

```bash
python tsv_processor.py llm_tsv_to_seval_tsv --input_file="llm_output.tsv" --segment="email"
```

**Part 3: TSV Cleanup (`cleanup_tsv`)**
- Removes enclosing quotes from utterances
- Replaces 80+ non-ASCII characters with ASCII equivalents
- Validates quote and non-ASCII placement
- Comprehensive character replacement: mathematical symbols, fractions, currency, etc.

```bash
python tsv_processor.py cleanup_tsv --input_file="input.tsv" --output_file="cleaned.tsv"
```

**Part 4: TSV Splitting (`split_utterances`)**
- Splits by segment and email account
- Groups utterances by segment (email, file)
- Creates files: `<segment>_<email_prefix>_utterances.tsv`

```bash
python tsv_processor.py split_utterances --input_file="utterances.tsv"
```

**Part 5: Column Filtering (`filter_by_column`, `analyze_column`)**
- SQL-like filtering: `WHERE column_name = 'value'`
- Analyze value distribution in columns
- Case-insensitive matching

```bash
python tsv_processor.py analyze_column --input_file="data.tsv" --column_name="Segment"
python tsv_processor.py filter_by_column --input_file="data.tsv" --column_name="Segment" --value="email"
```

**Part 6: Field Validation (`validate_file`)**
- Validates all fields for completeness
- Identifies missing or empty fields
- Generates detailed validation reports

```bash
python tsv_processor.py validate_file --input_file="data.tsv"
```

**Part 7: Duplicate Detection (`check_duplicates`, `remove_duplicates`)**
- Detects duplicate utterances
- Removes duplicates with statistics

```bash
python tsv_processor.py check_duplicates --input_file="data.tsv"
python tsv_processor.py remove_duplicates --input_file="data.tsv"
```

**Part 8: TSV File Merging (`merge_tsv_files`)**
- Validates headers match (same columns in same order)
- Merges data rows from both files
- Detailed error reporting for header mismatches

```bash
python tsv_processor.py merge_tsv_files --file1="part1.tsv" --file2="part2.tsv" --output_file="merged.tsv"
```

---

### 10. Timestamp Prompt Generator (`timestamp_prompt_generator.py`)

**Purpose:** Generates prompts for adding timestamps to utterances based on email accessibility.

**Key Features:**
- Processes split utterance files (email_*.tsv)
- Extracts corresponding email account from filename
- Retrieves all accessible emails for context
- Creates comprehensive prompts for LLM timestamp addition
- Ensures emails are accessible before utterance timestamp

**Main Commands:**
```bash
# Generate timestamp prompts for all split files
python timestamp_prompt_generator.py generate_timestamp_prompts

# Custom directories
python timestamp_prompt_generator.py generate_timestamp_prompts --split_results_dir="custom_split"

# Process remaining untimestamped utterances
python timestamp_prompt_generator.py generate_remaining_timestamp_prompts
```

**Workflow:** This creates prompts → Manual LLM processing → `utterance_timestamper.py` applies timestamps

---

### 11. Utterance Timestamper (`utterance_timestamper.py`)

**Purpose:** Adds timestamps to utterances for SEVAL testing. Handles both file-based and email-based utterances.

**Key Features:**
- **File utterances:** Uses file modification times from folder scan
- **Email utterances:** Reads LLM-generated timestamps from result files
- Handles duplicate utterances gracefully
- Generates timestamps a few hours after latest relevant file/email
- Supports both JSON and TSV formats

**Main Commands:**
```bash
# Timestamp file-based utterances
python utterance_timestamper.py timestamp_file_utterances --input_file="file_utterances.tsv"

# Timestamp email-based utterances from LLM results
python utterance_timestamper.py timestamp_email_utterances --input_file="email_utterances.tsv" --llm_results_folder="results/timestamps"

# Custom files folder for modification times
python utterance_timestamper.py timestamp_file_utterances --files_folder="data/custom_files"
```

**Input Formats:** TSV or JSON with 'Segment' field distinguishing file vs email utterances

---

## 🔄 Common Workflows

### Workflow 1: Email-Based Utterance Generation (Merged Accounts)

```bash
# Step 1: Classify accounts and generate mapping
python email_account_analysis.py generate_user_email_mapping --emails_file="data/emails.config.json"

# Step 2: Extract emails for target user
python email_extractor_merged_accounts.py extract_emails marion.chen

# Step 3: Generate timestamp prompts
python timestamp_prompt_generator.py generate_timestamp_prompts

# Step 4: Process prompts with LLM (manual step - use prompts from results/)

# Step 5: Apply timestamps from LLM results
python utterance_timestamper.py timestamp_email_utterances --input_file="utterances.tsv"

# Step 6: Convert to SEVAL format
python tsv_processor.py llm_tsv_to_seval_tsv --input_file="timestamped.tsv" --segment="email"

# Step 7: Cleanup and validate
python tsv_processor.py cleanup_tsv --input_file="seval.tsv" --output_file="final.tsv"
python tsv_processor.py validate_file --input_file="final.tsv"
```

### Workflow 2: File-Based Utterance Generation

```bash
# Step 1: Analyze file permissions
python files_individual_extractor.py analyze_file_access --config_file="data/files.config.json"

# Step 2: Extract files for specific user with prompt
python files_individual_extractor.py extract_user_file_content_with_prompt --user_email="johndoe"

# Step 3: Process prompts with LLM (manual step)

# Step 4: Add timestamps based on file modification times
python utterance_timestamper.py timestamp_file_utterances --input_file="file_utterances.tsv"

# Step 5: Convert to SEVAL format
python tsv_processor.py llm_tsv_to_seval_tsv --input_file="timestamped.tsv" --segment="file"

# Step 6: Validate and cleanup
python tsv_processor.py cleanup_tsv --input_file="seval.tsv"
python tsv_processor.py validate_file --input_file="seval.tsv"
```

### Workflow 3: Teams/Meetings Analytics

```bash
# Step 1: Generate Teams message analytics (excludes meeting chats for prompts)
python teams_message_analytics.py generate_comprehensive_analytics --data_folder=data

# Step 2: Extract user-specific Teams messages with files
python teams_message_analytics.py extract_user_teams_messages_with_files_and_prompt --user_email="alex.khan"

# Step 3: Generate meetings/events analytics (formal meeting contexts)
python meetings_events_analytics.py generate_meetings_events_analytics --data_folder=data

# Step 4: Extract meeting content for specific participant
python meetings_events_analytics.py extract_meetings_for_complex_queries --participant_filter="alex.khan"
```

### Workflow 4: TSV Processing Pipeline

```bash
# Step 1: Transform raw LLM output
python tsv_processor.py process_file --input_file="raw.tsv" --segment="email"

# Step 2: Convert to SEVAL format
python tsv_processor.py llm_tsv_to_seval_tsv --input_file="transformed.tsv"

# Step 3: Clean up non-ASCII and quotes
python tsv_processor.py cleanup_tsv --input_file="seval.tsv"

# Step 4: Validate all fields
python tsv_processor.py validate_file --input_file="cleaned.tsv"

# Step 5: Check for duplicates
python tsv_processor.py check_duplicates --input_file="cleaned.tsv"

# Step 6: Remove duplicates if found
python tsv_processor.py remove_duplicates --input_file="cleaned.tsv"

# Step 7: Split by segment and account
python tsv_processor.py split_utterances --input_file="cleaned.tsv"

# Step 8: Merge split files if needed
python tsv_processor.py merge_tsv_files --file1="part1.tsv" --file2="part2.tsv" --output_file="final.tsv"
```

---

## 📁 Data Requirements

### Required Input Files

**Email Processing:**
- `data/emails.config.json` - Email database with messages
- `data/users.config.json` - User directory with account information

**File Processing:**
- `data/files.config.json` - File metadata and permissions
- `data/files/` - Actual file contents (various formats)

**Teams/Meetings:**
- `data/chats.config.json` - Teams chat messages
- `data/onlinemeetings.config.json` - Meeting metadata
- `data/events.config.json` - Calendar events
- `data/transcripts/*.vtt` - Meeting transcript files

### Generated Output Files

**Intermediate Files:**
- `user_email_mapping.json` - User-to-account mappings
- `account_classifications.json` - Account type classifications
- `*_emails.json` - Extracted emails per user
- `*_analysis.json` - Statistics and analytics results

**TSV Files:**
- `*_transformed.tsv` - Transformed utterances
- `*_seval.tsv` - SEVAL format utterances
- `*_cleaned.tsv` - Cleaned and validated
- `*_utterances.tsv` - Split by segment/account
- `*_timestamped.tsv` - With added timestamps

**Prompt Files:**
- `results/*_prompt.md` - Generated prompts for LLM processing

---

## 🚀 Quick Start Examples

### Example 1: Extract Marion Chen's Emails (Merged)
```bash
python email_account_analysis.py generate_user_email_mapping
python email_extractor_merged_accounts.py extract_emails marion.chen
```

### Example 2: Analyze John Doe's File Access
```bash
python files_individual_extractor.py get_user_accessible_files --user_email="johndoe"
```

### Example 3: Get Alex Khan's Teams Activity
```bash
python teams_message_analytics.py extract_user_teams_messages_with_files_and_prompt --user_email="alex.khan"
```

### Example 4: Clean and Validate TSV
```bash
python tsv_processor.py cleanup_tsv --input_file="data.tsv" --output_file="cleaned.tsv"
python tsv_processor.py validate_file --input_file="cleaned.tsv"
```

### Example 5: Generate Meeting Analytics
```bash
python meetings_events_analytics.py generate_meetings_events_analytics --data_folder=data --output_file="meetings.md"
```

---

## 🛠️ VSCode Launch Configurations

The workspace includes 11 pre-configured launch configurations for TSV processing tasks:

- **TSV - Transform File** - Add Segment and Source columns
- **TSV - LLM to SEVAL** - Convert LLM output to SEVAL format
- **TSV - Cleanup** - Remove quotes and normalize non-ASCII
- **TSV - Split Utterances** - Split by segment and email
- **TSV - Analyze Column** - Show value distribution
- **TSV - Filter by Column** - SQL-like WHERE filtering
- **TSV - Validate File** - Check field completeness
- **TSV - Check Duplicates** - Find duplicate utterances
- **TSV - Remove Duplicates** - Remove duplicates with stats
- **TSV - Validate Specific Fields** - Custom field validation
- **TSV - Merge Two Files** - Merge with header validation

Access via **Run and Debug** panel (Ctrl+Shift+D)

---

## 📝 Command Line Interface

All modules use Python Fire for CLI:

```bash
# General pattern
python <module>.py <command> --parameter1=value1 --parameter2=value2

# Get help
python <module>.py --help
python <module>.py <command> --help

# Boolean flags
--flag=true   # Enable
--flag=false  # Disable
```

---

## 🔍 Tips & Best Practices

1. **Email Workflows:**
   - Use `email_account_analysis.py` first to generate mappings
   - Choose individual extractor for independent accounts, merged for comprehensive user view
   - Always validate email counts with statistics module

2. **File Processing:**
   - Run `analyze_file_access` before extracting user files
   - Check file existence in `data/files/` folder
   - Use prompt generation for LLM-based utterance creation

3. **TSV Processing:**
   - Always run `cleanup_tsv` before validation
   - Check duplicates before merging files
   - Use `validate_file` as final step before SEVAL testing
   - Merge files only after verifying headers match

4. **Timestamp Generation:**
   - Generate prompts with `timestamp_prompt_generator.py`
   - Process prompts manually with LLM
   - Apply results with `utterance_timestamper.py`
   - File timestamps use modification times, email timestamps from LLM

5. **Analytics:**
   - Use `teams_message_analytics.py` for daily collaboration patterns (excludes meeting chats in prompts)
   - Use `meetings_events_analytics.py` for formal meeting contexts
   - Export to Markdown for easy LLM consumption

---

## 📊 Output Formats

**JSON Files:**
- Account classifications with type, frequency, and group membership
- Email extraction results with chronological ordering
- File access analysis with permission details
- Statistics reports with counts, breakdowns, and rankings

**TSV Files:**
- SEVAL format: `Utterance, Segment, Environment, Grounding Data Source, user_id, timestamp`
- Cleaned and validated with ASCII-only characters
- Split files by segment and account for targeted testing

**Markdown Files:**
- Analytics reports for Teams messages, meetings, and events
- Prompt templates filled with context data for LLM processing
- Comprehensive summaries optimized for human readability

---

## 🐛 Troubleshooting

**Issue:** "Could not find user mapping file"
- **Solution:** Run `email_account_analysis.py generate_user_email_mapping` first

**Issue:** "Invalid JSON in config file"
- **Solution:** Validate JSON syntax using online validator or `python -m json.tool file.json`

**Issue:** "No emails found for user"
- **Solution:** Check user exists with `list_accounts` command, verify spelling

**Issue:** "Header mismatch when merging TSV files"
- **Solution:** Ensure both files have identical column headers in same order

**Issue:** "Non-ASCII characters still appearing"
- **Solution:** Run `cleanup_tsv` module with latest character replacement dictionary (80+ mappings)

**Issue:** "Empty timestamp field"
- **Solution:** Verify LLM results folder path, check timestamp format in LLM outputs

---

## 📚 Additional Documentation

- **Prompt Templates:** See `prompts_manual/` directory for LLM prompt examples
- **LLM Usage Guide:** `docs/llm_usage_and_auth_guide.md`
- **Universal File Reader:** `tools/README_UniversalFileReader.md`
- **TSV Processing Details:** Docstrings in `tsv_processor.py` (2245 lines, 8 parts)

---

## 🔗 Related Tools

**In BizChatScripts Framework:**
- `utils/file_reader.py` - UniversalFileReader for multiple file formats
- `llms/` - LLM API integration and authentication
- `tools/` - Additional data processing utilities

---

## 📞 Support

For issues, questions, or contributions:
1. Check module docstrings for detailed usage: `python <module>.py --help`
2. Review VSCode launch configurations for example parameters
3. Examine prompt templates in `prompts_manual/` directory
4. Check logs in output directories for detailed error messages

---

## 🎯 Summary

This synthetic tenants toolkit provides end-to-end capabilities for generating complex, realistic test utterances from Microsoft 365 synthetic tenant data. With 11 specialized modules covering emails, files, Teams, meetings, and comprehensive TSV processing, you can extract, analyze, timestamp, validate, and format utterances for SEVAL testing and BizChat Copilot evaluation.
