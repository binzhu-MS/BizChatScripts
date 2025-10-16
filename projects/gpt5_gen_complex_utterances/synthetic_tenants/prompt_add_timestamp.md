# **Timestamp Addition for Complex Utterances**

## **Objective**
Add appropriate timestamps to each utterance based on the email context provided. The timestamp represents when the utterance would be asked to **BizChat Copilot**, ensuring that all relevant emails needed to answer the utterance are accessible.

---

## **BizChat Copilot Context**
BizChat Copilot is an AI assistant that:
- Receives user utterances and routes them to either a **chat model** or **reasoning model**
- Uses available tools including **email search tool** to retrieve relevant information
- Can only retrieve emails that were **sent BEFORE the utterance timestamp**
- May call tools multiple times to gather all necessary information
- Generates grounded responses based on retrieved data

**Critical Constraint:** The email search tool can only access emails sent before the utterance timestamp. Emails sent after the timestamp are not accessible.

---

## **Timestamp Requirements**

### **Format**
All timestamps must use this exact format: `YYYY-MM-DDTHH:MM:SSZ` (ISO 8601 UTC)
Examples: `2024-11-21T17:00:00Z`, `2025-06-15T09:30:00Z`

### **Timing Strategy**
For each utterance, the timestamp should be set to ensure:
1. **All relevant emails** needed to answer the utterance are accessible (sent before the timestamp)
2. **Realistic timing** - the timestamp should make sense in the context of when someone would ask such a question
3. **Email dependency analysis** - consider which emails the utterance references or requires

### **Analysis Process**
For each utterance:
1. **Identify required emails**: Which emails from the provided dataset would be needed to answer this utterance?
2. **Find latest email**: What is the timestamp of the most recent email needed?
3. **Add buffer time**: Add appropriate time (hours, days, weeks) after the latest required email
4. **Consider realistic timing**: When would someone realistically ask this question?

---

## **Email Data Context**
**Email Account:** `{email_account}`

**Available Emails:**
The following emails are accessible by this account and sorted chronologically:

### **Sent Emails:** {sent_count} emails
{sent_emails_summary}

### **Received Emails:** {received_count} emails  
{received_emails_summary}

**Full Email Data:**
```json
{email_json_data}
```

---

## **Utterances to Process**

{full_utterances_list}

---

## **Instructions**

**IMPORTANT:** Process EXACTLY the utterances listed above. Do not add, remove, or duplicate any utterances. The output must contain the same number of utterances as provided in the input.

1. **Analyze each utterance** to understand what information it's seeking
2. **Identify required emails** from the provided email dataset that would be needed to answer the utterance
3. **Determine the latest email timestamp** that the utterance depends on
4. **Calculate appropriate timestamp** by adding realistic buffer time after the latest required email
5. **Ensure realistic timing** - consider when someone would realistically ask such a question
6. **Process each utterance exactly once** - do not create variations or duplicates

## **Output Format**
**CRITICAL:** Your output must contain exactly {utterance_count} rows (one for each utterance provided above). Do not add extra utterances or create duplicates.

Provide your response as a TSV (Tab-Separated Values) format with the following columns:
```
Utterance	Segment	Email_Account	Source	Timestamp	Reasoning
```

For each utterance, include:
- **Utterance**: The exact utterance text (unchanged)
- **Segment**: The segment value (unchanged) 
- **Email_Account**: The email account (unchanged)
- **Source**: The source value (unchanged)
- **Timestamp**: The calculated timestamp in format YYYY-MM-DDTHH:MM:SSZ
- **Reasoning**: Brief explanation of why this timestamp was chosen (which emails were considered, what buffer time was added)

## **Example Output Format**
```
Utterance	Segment	Email_Account	Source	Timestamp	Reasoning
Search my inbox for emails about Q3 budget from Sarah	email	alex.johnson@VertexEdgeLabs@SyntheticTenant	synthetic	2024-11-21T17:00:00Z	Latest budget email from Sarah was 2024-11-20T14:30:00Z, added 1 day buffer for realistic timing
```

---

## **Important Notes**
- Do not modify the utterance text, segment, email_account, or source fields
- Only add the timestamp and reasoning columns
- Ensure all timestamps are in UTC format with Z suffix
- Consider the chronological order of emails when determining dependencies
- Be conservative with buffer times to ensure all relevant emails are accessible
- If an utterance doesn't clearly depend on specific emails, use a reasonable timestamp based on the general email timeline
