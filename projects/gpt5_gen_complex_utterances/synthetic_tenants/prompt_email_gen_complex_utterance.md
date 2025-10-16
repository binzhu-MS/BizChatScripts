# **Complex Utterance Generator for Email Data**

## **Objective**
Generate concise, diverse, and **high-complexity user utterances** based on the provided enterprise email data. Each utterance should:
- Explicitly indicate the need to **search emails**.
- Simulate real-world enterprise scenarios using provided content.
- Require reasoning steps where reasoning models outperform standard chat models, as tested in **BizChat Copilot**.
- Use natural, human-like temporal expressions (like *yesterday's email*, *last week's thread*) instead of precise timestamps in the utterance text.

---

## **Testing Context**  
BizChat Copilot answers user queries using either a **chat model** or a **reasoning model**. Typically, **simple utterances** are handled by the chat model, while **complex utterances** invoke the reasoning model. Our goal is to generate utterances that strongly steer the system toward using the reasoning model.  

Complex utterances generally:  
- Require **Chain-of-Thought (CoT) reasoning** to interpret and respond effectively  
- Require **strategic multi-step search planning** for email retrieval
- Require **query refinement for accurate email retrieval**  
- Require **multi-step reasoning to integrate insights across multiple emails**
- Require **temporal reasoning** about email sequences and timelines

---

## **Environment & Constraints**  
- **Search Tools:** Dedicated tools are available for emails, chats, files, calendars, and web content. For these utterances, the model must correctly select the **email search tool** and craft **optimized queries** that retrieve the most relevant emails to produce a grounded, accurate response.  
- **Search Behavior:**  
  - Each query can retrieve up to **10 results**; multiple queries are often necessary for comprehensive coverage.  
  - Both models have equal access to these tools but must plan and adjust searches iteratively based on intermediate results.  
- **Temporal Constraints:** Queries can only find emails that exist **before the query timestamp**. Generate realistic query times that are **at least a few hours after** the latest emails mentioned.
- **Response Generation:** The model must synthesize answers solely from retrieved content. Raw email text is **not directly accessible to the model during input processing in BizChat Copilot**, so it must infer search strategies from the user's utterance and retrieved snippets.

---

## **Critical Considerations**  

1. **Syntactic Variety Mandate - AVOID TEMPLATE PATTERNS**
   - **FORBIDDEN:** Starting every utterance with "Search my inbox..." or similar formulaic openings
   - **REQUIRED:** Use diverse sentence structures, question formats, imperatives, conditionals, comparatives
   - **Pattern Breaking:** If you notice repetitive phrasing, immediately switch to a completely different grammatical structure
   - **Creativity Over Formula:** Prioritize natural, varied language over predictable templates

2. **Email Search Optimization**  
   - **Explicit Search Intent:** Every utterance must clearly instruct an **email search**, using varied phrases like *search my inbox*, *find emails*, *look through email threads*, *pull up messages*, *review emails about*, *check correspondence*.
   - **Search Engine Intelligence:** Provide **rich contextual clues** and let the search engine find relevant emails rather than specifying exact timestamps:
     - **Senders/Recipients:** *emails from Sarah*, *messages John sent*, *correspondence with Finance*, *emails I received from the team*
     - **Subjects/Projects:** *Q3 Budget review*, *Azure migration plan*, *Compliance Review thread*, *about the deployment*
     - **Natural time references:** *yesterday's email*, *last week's thread*, *recent messages*, *before the deadline*, *after the meeting*
     - **Technical/Topic anchors:** *about audit findings*, *regarding deployment issues*, *concerning budget approval*, *mentioning the compliance requirements*
   - **Trust Search Capability:** Modern search engines can correlate senders + topics + timeframes without needing precise minute-by-minute timestamps
   - **Avoid Weak Keywords:** Generic references like *audit issues* without context lead to poor matches. Instead: *emails about audit findings from Finance last quarter* or *in the Compliance Review thread*
   - **Encourage Iterative Retrieval:** Signal multi-step needs: *first check Sarah's audit emails, then find related compliance follow-ups*
   - **Mailbox Scoping:** Use phrases like *in my inbox*, *from Sent Items*, *in my threads*, *through my email history*

3. **Reality Gap:** During generation, you have full visibility into the email dataset. However, during BizChat's execution, both models see only the user's utterance—they have **no direct access to the underlying email content**. The model must infer **what to search for, which keywords to prioritize, and how to refine its queries iteratively** based on the original utterance and the snippets retrieved from prior searches.

4. **Reasoning Requirement:** Utterances should demand **beyond single-lookups**, requiring multiple searches and additional reasoning steps such as:  
   - Correlating decisions from separate email threads  
   - Building timelines or detecting dependencies across messages
   - Combining partial facts from different senders into one answer
   - Detecting discrepancies between planned actions and reported outcomes
   - Synthesizing insights from multi-party conversations
   - Tracking decision-making processes across email exchanges

5. **Temporal Constraints:** Queries can only find emails that exist **before the query timestamp**. Generate realistic query times that are **at least a few hours after** the latest emails mentioned.

---

## **Natural Temporal References in Utterances**
Generate utterances with **natural, human-like temporal expressions** instead of precise timestamps:

### **Realistic Time Expressions:**
- **Recent references:** *yesterday's email*, *this morning's message*, *today's correspondence*
- **Relative periods:** *last week's thread*, *the email from Monday*, *recent exchanges*, *our conversation earlier*
- **Event-anchored:** *before the meeting*, *after the deadline*, *during the review*, *when we discussed the project*
- **Contextual:** *the recent compliance email*, *latest budget update*, *the most recent status report*

### **Search Engine Reliance:**
Users provide **contextual clues** rather than exact timestamps, trusting search engines to find relevant emails:
- **Senders/Recipients:** *in my emails with Sarah*, *messages from the Finance team*
- **Topics/keywords:** *about the Q3 budget*, *regarding audit findings*, *concerning migration plans*
- **Subjects:** *in the Compliance Review thread*, *from the Azure Migration discussion*
- **Time context:** *before the project kickoff*, *after the incident*, *during last month's planning*

### **Query Timestamp Logic:**
Each utterance still needs a **realistic query timestamp** for the TSV output that:
1. **Comes after** the latest referenced email content (minimum 3+ hours gap)
2. **Uses ISO 8601 format** for consistency: `2025-07-05T14:00:00Z`
3. **But the utterance text itself uses natural language** for temporal references

---

## **Success Criteria**
A valid complex utterance:
1. Clearly **demands email searching** with explicit search-related verbs.
2. Includes **search-friendly cues** (people's names, projects, subjects, natural time frames).
3. Requires **iterative steps** (can't be answered with one simple lookup).
4. Forces reasoning like **temporal logic, correlation, integration**, or **causal deduction**.
5. Forces **search refinement** if naive keywords fail.
6. Includes **realistic query timestamp** (ISO 8601 format) after referenced content.
7. Engages reasoning skills like **timeline deduction, cross-thread correlation, dependency analysis**.

---

## **Targeted Reasoning Advantages**
Prioritize scenarios involving:
- **Temporal Logic & Dependencies** (project timelines, decision sequences, deadline tracking).
- **Cross-Thread Integration** (decisions spread across separate email conversations).
- **Causal Inference** (e.g., "What led to the decision...", "Why was the project delayed...").
- **Pattern Recognition** (trends across multiple email threads or audits).
- **Multi-Party Synthesis** (combining insights from emails across different senders).
- **Discrepancy Detection** (comparing planned actions vs. reported outcomes).
- **Decision Tracking** (following decisions from initial proposal through implementation).

---

## **Generation Rules**

1. **Syntactic Diversity - CRITICAL:** **Vary sentence structures dramatically** to avoid template-like repetition:
   - **Question Formats:** *"What were the exact compliance requirements Sarah mentioned in yesterday's email, and how do they align with..."*
   - **Imperative Variations:** *"Pull up our Q3 budget thread and cross-check..."*, *"Locate emails from..."*, *"Retrieve messages about..."*
   - **Conditional Structures:** *"If the audit findings from Finance match our compliance review, then analyze..."*
   - **Comparative Frames:** *"Compare what we decided in last week's email thread versus what actually shipped..."*
   - **Causal Inquiries:** *"Determine why the project was delayed by examining emails between..."*
   - **Timeline Constructions:** *"Trace the evolution from the initial proposal email through the final approval..."*
   - **Problem-Solution Formats:** *"Given the discrepancies between our planning emails and the actual results..."*
   - **Hypothesis Testing:** *"Verify whether our email assumptions about the migration timeline held true by..."*

2. **Opening Variation Examples:**
   - Instead of always "Search my inbox..." use:
     - *"What did we conclude in yesterday's email about..."*
     - *"Pull up the thread where Sarah mentioned..."*
     - *"Trace back through recent messages on..."*
     - *"Reconstruct the decision-making from our email exchange about..."*
     - *"Check what we decided in the compliance review thread..."*
     - *"Cross-reference last week's emails about the budget with..."*
     - *"Verify the implementation against our planning emails..."*
     - *"Compare the timeline from John's emails with..."*
     - *"Look back at our correspondence about the audit and..."*
     - *"Review what the team agreed on in our email discussion of..."*

3. **Email-Focused Language:** Always include email-related actions but vary the phrasing:
   - Direct search: *find emails*, *search my inbox*, *review emails*, *look through messages*
   - Contextual: *in my emails*, *from my inbox*, *through email interactions*, *via email*
   - Action-focused: *emails mentioning [topic]*, *emails between [dates]*, *emails with attachments*
   - Analytical: *based on my emails*, *according to our email thread*, *emails where I'm mentioned*
   - Workflow: *catch me up on emails*, *summarize the email thread*, *list emails from [sender]*

4. **Sender/Recipient Context:** Reference actual senders and recipients from the email data:
   - *emails from Sarah about compliance*
   - *messages John sent regarding the deployment*  
   - *correspondence with the Finance team*

5. **Technical Specificity:** Use actual technical terms, projects, and topics from the data:
   - *Q3 Budget approval process*
   - *Azure migration timeline*
   - *compliance audit findings*

6. **Temporal Precision:** Include realistic timestamps and natural time references:
   - Utterance text: *emails from last week* or *yesterday's message*
   - Query_Timestamp: ISO 8601 format (e.g., `2025-07-05T14:00:00Z`)

7. **Structural Variety:** **Avoid formulaic patterns**. Mix:
   - **Short, direct queries** (20-30 words)
   - **Complex, multi-clause requests** (60-80 words)
   - **Conditional/hypothetical structures**
   - **Comparative analysis requests**
   - **Root-cause investigation formats**

8. **Diversity & Quantity:** Generate **as many unique, high-quality utterances as possible**
9. **Conciseness:** Keep utterances focused but intentionally complex
10. **Enterprise Realism:** Use actual sender names, project names, and scenarios from provided email data
11. **Plain Text Format:** Write all utterances in plain text **without any quotation marks, backticks, or wrapping**. Use single quotes (') sparingly and only when referencing specific subjects or terms within the utterance.

---

## **Example Utterances - Demonstrating Syntactic Diversity & Natural Temporal References**

✅ **Question Format** - *What were the exact budget figures Sarah proposed in yesterday's email about Q3 planning, and do they align with the constraints Finance mentioned in their earlier message?* [Query_Timestamp: 2025-07-05T16:00:00Z]

✅ **Comparative Structure** - *Compare the compliance requirements we agreed on in last week's email thread with what was actually implemented according to the audit team's recent feedback* [Query_Timestamp: 2025-07-05T18:30:00Z]

✅ **Causal Investigation** - *Determine why the Azure migration was delayed by examining emails between John and the infrastructure team from the past two weeks* [Query_Timestamp: 2025-07-05T14:00:00Z]

✅ **Conditional Analysis** - *If the deployment schedule Sarah outlined in Monday's email matches what the DevOps team confirmed, analyze whether the timeline is still realistic given the recent issues mentioned by John* [Query_Timestamp: 2025-07-05T15:30:00Z]

✅ **Timeline Reconstruction** - *Trace the evolution of our Q3 budget decision from the initial proposal in Finance's email through the final approval by examining all related correspondence* [Query_Timestamp: 2025-07-05T17:00:00Z]

✅ **Hypothesis Testing** - *Verify whether the cost savings assumptions in our planning emails actually materialized based on the financial reports mentioned in this week's correspondence* [Query_Timestamp: 2025-07-05T19:00:00Z]

✅ **Direct Imperative** - *Pull up our compliance review thread from last month and cross-check the audit findings against what we originally planned in the project kickoff email* [Query_Timestamp: 2025-07-05T16:45:00Z]

✅ **Natural Event Anchoring** - *Look back at our post-incident emails about the deployment failure and verify whether the root causes identified match the mitigation steps we discussed in yesterday's meeting followup* [Query_Timestamp: 2025-07-05T17:30:00Z]

❌ **Avoid These Anti-Patterns:**
- ❌ **Precise timestamps in utterance:** *"Find emails from 2025-06-02 10:15-10:20"* - too specific in text
- ❌ **Repetitive openings:** *"Search my inbox for..."* appearing in every utterance - overused 
- ❌ **Generic queries:** *"What happened with the project?"* - no search intent, no context
- ❌ **No temporal context:** Missing natural time references like *yesterday* or *last week*
- ❌ **Weak anchors:** *"Find emails about issues"* - too vague, needs specifics

---

## **Output Format (TSV)**
Output generated utterances in the TSV file format with the following headers without any empty rows.
```
Utterance    Email_Account    Query_Timestamp    Complexity_Reason
```
- **Utterance:** A complex email-search request written in plain text using natural temporal expressions (do **not** use quotation marks).
- **Email_Account:** Provided account (e.g., `marion.chen`).
- **Query_Timestamp:** Realistic timestamp in ISO 8601 format (e.g., `2025-07-05T14:00:00Z`) that is at least 3+ hours after the latest email referenced in the utterance.
- **Complexity_Reason:** Short note on why multi-step reasoning is needed (e.g., "Requires correlating decisions across 3 email threads with temporal sequencing").

---

## **Input Data**
Email Account: `{email_account}`  
Email Data:  
```
{email_json_data}
```

---

## **Goal**
Produce **as many unique, high-quality utterances as possible** that:
- Explicitly direct the model to **search emails**.
- Use **diverse sentence structures** (avoid template patterns).
- Use **natural temporal expressions** in utterance text.
- Include **realistic query timestamps** in ISO 8601 format.
- Require **iterative reasoning** and **strategic keyword planning**.
- Cannot be answered by simple keyword matching or summarization.
- Leverage actual email content (senders, subjects, topics, timelines) from provided data.
