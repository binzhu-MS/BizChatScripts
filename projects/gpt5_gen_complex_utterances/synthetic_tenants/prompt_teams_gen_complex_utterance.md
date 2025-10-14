# **Complex Utterance Generator for Teams Messages**

## **Objective**
Generate concise, diverse, and **high-complexity user utterances** based on the provided Teams messages and file content. Each utterance should:
- Explicitly indicate the need to **search Teams messages** and access files (either directly from message references or through additional file searches when URLs/locations aren't provided).
- Simulate real-world enterprise scenarios using provided Teams conversations and shared document content.
- Require reasoning steps where reasoning models outperform standard chat models, as tested in **BizChat Copilot**.
- Use natural, human-like temporal expressions (like *yesterday's chat*, *last week's discussion*) instead of precise timestamps.

---

## **Testing Context**  
BizChat Copilot answers user queries using either a **chat model** or a **reasoning model**. Typically, **simple utterances** are handled by the chat model, while **complex utterances** invoke the reasoning model. Our goal is to generate utterances that strongly steer the system toward using the reasoning model.  

Complex utterances generally:  
- Require **Chain-of-Thought (CoT) reasoning** to interpret and respond effectively  
- Require **strategic multi-step search planning** for Teams messages and intelligent file access (direct fetch vs. search)
- Require **query refinement for accurate Teams message retrieval** and determining optimal file access methods
- Require **multi-step reasoning to integrate insights** from conversations and documents
- Require **temporal reasoning** about message timelines and file modifications

---

## **Environment & Constraints**  
- **Search Tools:** Dedicated tools are available for Teams messages, files, emails, calendars, and web content. For these utterances, the model must correctly select **Teams message search** and determine whether additional file searches are needed, or if files can be accessed directly from Teams message references.
- **File Access Distinction:**
  - **Direct Access:** Files referenced in Teams messages (URLs, attachments, file IDs) can be **fetched directly** without separate file search
  - **Search Required:** Files not explicitly mentioned in messages require **separate file search** with strategic keyword planning
- **Search Behavior:**  
  - Each query can retrieve up to **10 results**; multiple queries are often necessary for comprehensive coverage.  
  - Both models have equal access to these tools but must plan and adjust searches iteratively based on intermediate results.
- **Temporal Constraints:** Queries can only find Teams messages and files that exist **before the query timestamp**. Generate realistic query times that are **at least a few hours after** the latest Teams messages and file content mentioned.
- **Response Generation:** The model must synthesize answers solely from retrieved content. Raw Teams and file content is **not directly accessible to the model during input processing in BizChat Copilot**, so it must infer search strategies from the user's utterance. 

---

## **Critical Considerations**  

1. **Syntactic Variety Mandate - AVOID TEMPLATE PATTERNS**
   - **FORBIDDEN:** Starting every utterance with "Search our Teams messages..." or similar formulaic openings
   - **REQUIRED:** Use diverse sentence structures, question formats, imperatives, conditionals, comparatives
   - **Pattern Breaking:** If you notice repetitive phrasing, immediately switch to a completely different grammatical structure
   - **Creativity Over Formula:** Prioritize natural, varied language over predictable templates

2. **Teams Message Search Optimization**  
   - **Explicit Teams Search Intent:** Every utterance must clearly instruct a **Teams message search**, using phrases like *search our Teams conversations*, *find our chat about*, *look through our Teams discussions*, *review our messages about*. 
   - **Search Engine Intelligence:** Provide **rich contextual clues** and let the search engine find the right conversations rather than specifying exact timestamps:
     - **Participants:** *conversations with Marion*, *messages from John*, *our discussion with the team*
     - **Topics/themes:** *about jitter metrics*, *regarding the alert tuning*, *when we discussed performance*
     - **Chat context:** *Backoff Metrics Discussion*, *SensorDriverHotfixReview chat*, *the post-incident review*
     - **Natural time references:** *yesterday's meeting*, *last week's discussion*, *our recent chat*, *after the deployment*
     - **Technical anchors:** *edgeos_exporter*, *jitter metrics*, *backoff multiplier*, *P95 latency*
   - **Trust Search Capability:** Modern search engines can correlate participants + topics + timeframes without needing precise minute ranges
   - **Encourage Cross-Reference:** Signal need to connect Teams conversations with files: *find the documents mentioned in our chat about...*, *locate files referenced in our Teams discussion on...*

3. **File Content Integration**  
   - **Direct File Access:** When files are explicitly referenced in Teams messages (URLs, file IDs, attachments), they can be **fetched directly** without search:
     - *"Review the smoke-test script at https://edgeosdocs.com/scripts/staging_smoke.sh that Marion mentioned in our jitter discussion"*
     - *"Check the performance data in the Excel file at https://edgeosdocs.com/share/perf-tests/retry_backoff_metrics_20250529.xlsx from Alex's backoff analysis"*
   - **File Search Required:** When files are not explicitly mentioned but are contextually relevant:
     - Complete filenames (*find documents named ZeroCopyFilter_Performance_IntegrationGuide*)
     - File types (*search Excel spreadsheets with performance data*, *locate PowerPoint presentations about architecture*)  
     - Content themes (*find documents about backoff metrics*, *search files containing jitter analysis*)  
   - **Hybrid Approach:** Combine Teams message references with additional file searches for comprehensive analysis

4. **Reality Gap:** During generation, you have full visibility into Teams messages and files. However, during BizChat's execution, both models see only the user's utterance—they have **no direct access to the underlying content**. The model must infer **what to search for, which participants to focus on, and how to connect Teams discussions with file content** based on the original utterance and retrieved snippets.

5. **Reasoning Requirement:** Utterances should demand **beyond single-lookups**, requiring multiple searches and reasoning steps such as:  
   - Correlating Teams discussions with referenced files
   - Building timelines from message sequences and file modifications  
   - Detecting discrepancies between planned actions (in Teams) and documented results (in files)
   - Synthesizing insights from multiple conversations and documents

---

## **Natural Temporal References in Utterances**
Generate utterances with **natural, human-like temporal expressions** instead of precise timestamps:

### **Realistic Time Expressions:**
- **Recent references:** *yesterday's chat*, *this morning's discussion*, *our conversation earlier today*
- **Relative periods:** *last week's meeting*, *the discussion we had on Monday*, *our recent sprint planning*
- **Event-anchored:** *before the deployment*, *after the incident*, *during the code review*, *when we discussed performance*
- **Seasonal/contextual:** *the planning session*, *our quarterly review*, *the post-mortem discussion*

### **Search Engine Reliance:**
Users provide **contextual clues** rather than exact timestamps, trusting search engines to find relevant conversations:
- **Participants:** *in our chat with Marion and Troy*  
- **Topics/keywords:** *about jitter metrics*, *regarding the alert tuning*
- **Chat names:** *in the Backoff Metrics discussion*, *from the SensorDriver review*
- **Technical context:** *when we troubleshot the spike*, *during the configuration changes*

### **Query Timestamp Logic:**
Each utterance still needs a **realistic query timestamp** for the TSV output that:
1. **Comes after** the latest referenced content (minimum 3+ hours gap)
2. **Uses ISO 8601 format** for consistency: `2025-07-05T14:00:00Z`
3. **But the utterance text itself uses natural language** for temporal references

---

## **Success Criteria**
A valid complex utterance:
1. Clearly **demands Teams message search** and appropriate file access (direct fetch from messages OR separate file search).
2. Includes **search-friendly cues** (participant names, chat topics, file names, technical terms).
3. Requires **iterative steps** (can't be answered with one simple lookup).
4. Forces reasoning like **temporal logic, correlation, integration**, or **causal deduction**.
5. **Strategically combines** direct file access from Teams references with additional file searches when needed.
6. Includes **realistic query timestamp** after referenced content.
7. Engages reasoning skills like **timeline deduction, cross-source correlation, dependency analysis**.

---

## **Targeted Reasoning Advantages**
Prioritize scenarios involving:
- **Teams-File Cross-Reference** (connecting conversations to documents)
- **Temporal Logic & Dependencies** (project timelines from messages and files)
- **Multi-Participant Analysis** (understanding team dynamics and file sharing)
- **Technical Correlation** (linking discussed metrics to documented results)
- **Decision Tracking** (following decisions from Teams to implementation in files)
- **Discrepancy Detection** (comparing planned vs. actual results)

---

## **Generation Rules**
1. **Strategic Content Access:** Combine Teams message search with appropriate file access:
   - **Direct Reference:** *"Check the smoke-test script at https://edgeosdocs.com/scripts/staging_smoke.sh that Marion shared in our jitter chat and compare the results with..."*
   - **Hybrid Approach:** *"Find our Teams conversations about deployment and also search for related configuration files not mentioned in our chats"*
   - **Search Extension:** *"Review our messages with Marion about performance and locate any additional Excel files with similar metrics data"*

2. **Syntactic Diversity - CRITICAL:** **Vary sentence structures dramatically** to avoid template-like repetition:
   - **Question Formats:** *"What were the exact changes Marion proposed in our jitter discussion, and how do they align with..."*
   - **Imperative Variations:** *"Pull up our backoff metrics chat and cross-check..."*, *"Locate the Teams thread about..."*, *"Retrieve messages from..."*
   - **Conditional Structures:** *"If the alert rules we discussed in Teams match the current config, then analyze..."*
   - **Comparative Frames:** *"Compare what we decided in our May 30th Teams chat versus what actually shipped..."*
   - **Causal Inquiries:** *"Determine why the jitter spike occurred by examining both our Teams discussion and..."*
   - **Timeline Constructions:** *"Trace the evolution from our initial Teams conversation through the final implementation..."*
   - **Problem-Solution Formats:** *"Given the discrepancies between our Teams planning session and the actual results..."*
   - **Hypothesis Testing:** *"Verify whether our Teams conversation assumptions about performance held true by..."*

3. **Opening Variation Examples:**
   - Instead of always "Search our Teams messages..." use:
     - *"What did we conclude in yesterday's chat about..."*
     - *"Pull up the conversation where Marion mentioned..."*
     - *"Trace back through our recent discussion on..."*
     - *"Reconstruct the decision-making from our planning session..."*
     - *"Check what we decided in our post-incident review..."*
     - *"Cross-reference our Teams conversation from last week with..."*
     - *"Verify the implementation against our planning session..."*
     - *"Compare the performance data with our discussion during the sprint review..."*
     - *"Look back at our conversation about the deployment and..."*
     - *"Review what the team agreed on during our architecture discussion..."*

4. **Participant Context:** Reference actual participants from the Teams data:
   - *messages with Marion about deployment*
   - *our discussion with Troy about performance*  
   - *team conversations involving Felecia*

5. **Technical Specificity:** Use actual technical terms from the data:
   - *edgeos_exporter configurations*
   - *jitter metrics and backoff analysis*
   - *zero-copy filter performance*

6. **Temporal Precision:** Include realistic timestamps and time references:
   - *conversations from May 29-30 and related documentation*
   - *messages before the June 1st deployment*

7. **Structural Variety:** **Avoid formulaic patterns**. Mix:
   - **Short, direct queries** (20-30 words)
   - **Complex, multi-clause requests** (60-80 words)
   - **Conditional/hypothetical structures**
   - **Comparative analysis requests**
   - **Root-cause investigation formats**

8. **Diversity & Quantity:** Generate **as many unique, high-quality utterances as possible**
9. **Conciseness:** Keep utterances focused but intentionally complex
10. **Enterprise Realism:** Use actual project names, technical terms, and scenarios from provided data
11. **Plain Text Format:** Write all utterances in plain text **without any quotation marks, backticks, or wrapping**. Use single quotes (') sparingly and only when referencing specific file names or technical terms within the utterance.

---

## **Example Utterances - Demonstrating Syntactic Diversity & Natural Temporal References**

✅ **Question Format** - *What were the exact jitter thresholds Marion proposed in yesterday's chat about alert tuning, and do they match the current alert rules at https://edgeosdocs.com/alerting/alert_rules_v2.yml?* [Query Time: 2025-07-05T16:00:00Z]

✅ **Comparative Structure** - *Compare our recent Teams planning session conclusions about backoff multipliers with the actual implementation in the performance metrics Excel file that Alex shared* [Query Time: 2025-07-05T18:30:00Z]

✅ **Causal Investigation** - *Determine why the jitter spike wasn't caught by examining both our Teams discussion about alert tuning last week and the smoke-test results Troy mentioned* [Query Time: 2025-07-05T14:00:00Z]

✅ **Conditional Analysis** - *If the evaluationInterval settings we agreed on in our recent Teams chat match the deployed config, analyze whether they would have prevented the performance regression Alex documented* [Query Time: 2025-07-05T15:30:00Z]

✅ **Timeline Reconstruction** - *Trace the evolution from our initial Teams conversation about retry logic through the final configuration by cross-referencing our chat history with the deployment artifacts* [Query Time: 2025-07-05T17:00:00Z]

✅ **Hypothesis Testing** - *Verify whether our Teams assumption that 1.8x multiplier would reduce latency actually held true based on the performance data Marion shared in the follow-up discussion* [Query Time: 2025-07-05T19:00:00Z]

✅ **Direct Imperative** - *Pull up our backoff metrics discussion from earlier this week and cross-check the S14 scenario results against what we predicted in the planning phase* [Query Time: 2025-07-05T16:45:00Z]

✅ **Natural Event Anchoring** - *Look back at our post-incident discussion about the jitter alerts and verify whether the changes we decided on actually made it into the configuration Marion deployed* [Query Time: 2025-07-05T17:30:00Z]

❌ **Avoid These Anti-Patterns:**
- ❌ **Precise timestamps:** *"Pull together the Secure-Boot Metrics Definition (2025-06-02 10:15–10:20Z)"* - too specific
- ❌ **Repetitive openings:** *"Search our Teams messages..."* - overused 
- ❌ **Formulaic structure:** *"Find our chat about... then fetch... and correlate..."* - template-like
- ❌ **Overly generic:** *"What happened with the deployment?"* - no source indication
- ❌ **Missing Teams context:** *"Tell me about jitter metrics"* - doesn't indicate Teams/file access strategy
- ❌ **Robot-like precision:** *"on 2025-05-30 08:30–09:00Z"* - humans don't talk this way

✅ **Use Natural Alternatives:**
- ✅ *"our recent security metrics discussion"* instead of precise timestamps
- ✅ *"yesterday's chat with Marion"* instead of exact time ranges  
- ✅ *"the planning session where we decided on thresholds"* instead of minute-level precision  

---

## **Output Format (TSV)**
Output generated utterances in TSV format with these headers (no empty rows):
```
Utterance	Email_Account	Query_Timestamp	Complexity_Reason
```
- **Utterance:** Complex Teams+file search request in **plain text with NO quotes or backticks**. If you need to reference specific file names or technical terms within the utterance, use single quotes (') only when absolutely necessary.
- **Email_Account:** Provided account (e.g., `alex.khan`)  
- **Query_Timestamp:** ISO 8601 timestamp after referenced content (e.g., `2025-07-05T16:00:00Z`)
- **Complexity_Reason:** Why multi-step reasoning across Teams+files is needed

**CRITICAL:** Do NOT wrap utterances in double quotes ("), backticks (`), or any other quotation marks. Write each utterance as plain, natural text.

---

## **Assigned User Email Account**

**Target User**: `{email_account}`

Generate utterances as if this user is asking questions that require searching through their Teams message history and accessible file content.

---

## **Input Data - Teams Messages and Files Accessible to Target User**

The following data represents **ONLY** the Teams messages and files that the target user `{email_account}` has access to. Generate utterances based exclusively on this accessible content - the user can only ask about conversations they participated in and files they can access.

{teams_messages_files_json_data}

---

## **Goal**
Produce **as many unique, high-quality utterances as possible** that:
- **Demonstrate maximum syntactic diversity** - NO two utterances should follow the same grammatical pattern or opening structure
- Explicitly direct the model to **search Teams messages** and access relevant files (direct fetch when referenced, or file search when locations aren't provided).
- Require **iterative reasoning** and **strategic cross-source planning**.
- Cannot be answered by simple keyword matching or single-source summarization.
- Include **realistic temporal constraints** and **technical specificity**.
- **Challenge reasoning models** with multi-step analysis across Teams conversations and document content.
- **Break away from formulaic language** - use questions, imperatives, conditionals, comparisons, causal investigations, timeline reconstructions, hypothesis testing, and problem-solution formats.