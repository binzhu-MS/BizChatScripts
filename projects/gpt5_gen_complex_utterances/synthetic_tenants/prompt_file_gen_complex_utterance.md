# **Complex Utterance Generator for File Data**

## **Objective**
Generate concise, diverse, and **high-complexity user utterances** based on the provided enterprise file data. Each utterance should:
- Explicitly indicate the need to **search files**.
- Simulate real-world enterprise scenarios using provided content.
- Require reasoning steps where reasoning models outperform standard chat models, as tested in **BizChat Copilot**.
- Use natural, human-like temporal expressions (like *yesterday's document*, *last week's report*) instead of precise timestamps in the utterance text.

---

## **Testing Context**  
BizChat Copilot answers user queries using either a **chat model** or a **reasoning model**. Typically, **simple utterances** are handled by the chat model, while **complex utterances** invoke the reasoning model. Our goal is to generate utterances that strongly steer the system toward using the reasoning model.  

Complex utterances generally:  
- Require **Chain-of-Thought (CoT) reasoning** to interpret and respond effectively  
- Require **strategic multi-step search planning** for file retrieval
- Require **query refinement for accurate file retrieval**  
- Require **multi-step reasoning to integrate insights across multiple files**
- Require **temporal reasoning** about file modifications and document sequences

---

## **Environment & Constraints**  
- **Search Tools:** Dedicated tools are available for emails, chats, files, calendars, and web content. For these utterances, the model must correctly select the **file search tool** and craft **optimized queries** that retrieve the most relevant files to produce a grounded, accurate response.  
- **Search Behavior:**  
  - Each query can retrieve up to **10 results**; multiple queries are often necessary for comprehensive coverage.  
  - Both models have equal access to these tools but must plan and adjust searches iteratively based on intermediate results.  
- **Temporal Constraints:** Queries can only find files that exist **before the query timestamp**. Generate realistic query times that are **at least a few hours after** the latest file modifications mentioned.
- **Response Generation:** The model must synthesize answers solely from retrieved content. Raw file content is **not directly accessible to the model during input processing in BizChat Copilot**, so it must infer search strategies from the user's utterance and retrieved snippets.

---

## **Critical Considerations**  

1. **Syntactic Variety Mandate - AVOID TEMPLATE PATTERNS**
   - **FORBIDDEN:** Starting every utterance with "Search my files..." or similar formulaic openings
   - **REQUIRED:** Use diverse sentence structures, question formats, imperatives, conditionals, comparatives
   - **Pattern Breaking:** If you notice repetitive phrasing, immediately switch to a completely different grammatical structure
   - **Creativity Over Formula:** Prioritize natural, varied language over predictable templates

2. **File Search Optimization**  
   - **Explicit Search Intent:** Every utterance must clearly instruct a **file search**, using varied phrases like *search my files*, *find documents*, *look through my files*, *review documents*, *locate files*, *pull up documents*.
   - **Search Engine Intelligence:** Provide **rich contextual clues** and let the search engine find relevant files rather than specifying exact timestamps:
     - **Complete filenames:** *the Q3_Budget.xlsx file*, *document named Azure_Migration_Plan.docx*
     - **Filename keywords:** *files with 'budget' in the name*, *documents containing 'migration'*
     - **File types:** *Excel spreadsheets*, *PDF reports*, *PowerPoint presentations*, *Word documents*
     - **Natural time references:** *files modified yesterday*, *documents from last week*, *recent reports*, *files created before the deadline*
     - **Technical/Topic anchors:** *budget analysis*, *deployment plans*, *compliance documentation*, *performance metrics*
     - **Folder/location hints:** *from the Finance folder*, *in the Project directory*, *shared documents*
   - **Trust Search Capability:** Modern search engines can correlate filenames + file types + topics + timeframes without needing precise timestamps
   - **Avoid Weak Keywords:** Generic references like *audit issues* without context lead to poor matches. Instead: *files containing audit findings from the Finance folder last quarter* or *PDF reports about compliance review*
   - **Encourage Iterative Retrieval:** Signal multi-step needs: *first check Excel files for budget data, then find related PowerPoint presentations*
   - **Folder Context:** Use phrases like *in my files*, *from my documents*, *in shared folders*, *from the project directory*

3. **Reality Gap:** During generation, you have full visibility into file data. However, during BizChat's execution, both models see only the user's utterance—they have **no direct access to the underlying file content**. The model must infer **what to search for, which keywords to prioritize, and how to refine its queries iteratively** based on the original utterance and the snippets retrieved from prior searches.

4. **Reasoning Requirement:** Utterances should demand **beyond single-lookups**, requiring multiple searches and additional reasoning steps such as:  
   - Correlating data from separate files across different folders  
   - Building timelines or detecting dependencies from document sequences
   - Combining partial information from different file types into one answer
   - Detecting discrepancies between planned documents and actual results
   - Synthesizing insights from multiple document versions

5. **Temporal Constraints:** Queries can only find files that exist **before the query timestamp**. Generate realistic query times that are **at least a few hours after** the latest file modifications mentioned.

---

## **Natural Temporal References in Utterances**
Generate utterances with **natural, human-like temporal expressions** instead of precise timestamps:

### **Realistic Time Expressions:**
- **Recent references:** *yesterday's document*, *this morning's report*, *today's spreadsheet*
- **Relative periods:** *last week's presentation*, *files from Monday*, *recent documents*, *files updated earlier*
- **Event-anchored:** *before the meeting*, *after the review*, *during the project*, *files created for the Q3 analysis*
- **Contextual:** *the recent compliance report*, *latest budget update*, *the most recent version*

### **Search Engine Reliance:**
Users provide **contextual clues** rather than exact timestamps, trusting search engines to find relevant files:
- **Filenames:** *the Q3_Budget file*, *Azure_Migration_Plan document*
- **File types:** *Excel spreadsheets with budget data*, *PDF reports about audits*
- **Topics:** *files about compliance*, *documents regarding migration plans*
- **Time context:** *files modified before the deadline*, *documents created during the project*, *recent reports*

### **Query Timestamp Logic:**
Each utterance still needs a **realistic query timestamp** for the TSV output that:
1. **Comes after** the latest referenced file modification (minimum 3+ hours gap)
2. **Uses ISO 8601 format** for consistency: `2025-07-05T14:00:00Z`
3. **But the utterance text itself uses natural language** for temporal references

---

## **Success Criteria**
A valid complex utterance:
1. Clearly **demands file searching** with explicit search-related verbs.
2. Includes **search-friendly cues** (filenames, file types, topics, natural time frames).
3. Requires **iterative steps** (can't be answered with one simple lookup).
4. Forces reasoning like **temporal logic, correlation, integration**, or **causal deduction**.
5. Forces **search refinement** if naive keywords fail.
6. Includes **realistic query timestamp** (ISO 8601 format) after referenced content.
7. Engages reasoning skills like **timeline deduction, cross-file correlation, dependency analysis**.

---

## **Targeted Reasoning Advantages**
Prioritize scenarios involving:
- **Temporal Logic & Dependencies** (document timelines, version sequences, deadline tracking).
- **Cross-File Integration** (data spread across separate documents or file types).
- **Causal Inference** (e.g., "What led to changes in...", "Why was the approach modified...").
- **Pattern Recognition** (trends across multiple reports or document versions).
- **Multi-Source Synthesis** (combining insights from Excel, PowerPoint, PDF, Word documents).
- **Discrepancy Detection** (comparing planned documents vs. actual deliverables).
- **Version Tracking** (following document evolution across modifications).

---

## **Generation Rules**

1. **Syntactic Diversity - CRITICAL:** **Vary sentence structures dramatically** to avoid template-like repetition:
   - **Question Formats:** *"What were the exact budget figures in yesterday's Q3_Budget.xlsx file, and how do they align with..."*
   - **Imperative Variations:** *"Pull up the Excel files about budget analysis and cross-check..."*, *"Locate documents from..."*, *"Retrieve files containing..."*
   - **Conditional Structures:** *"If the deployment plan in the PowerPoint matches the actual configuration, then analyze..."*
   - **Comparative Frames:** *"Compare the compliance requirements documented last week with the audit results in..."*
   - **Causal Inquiries:** *"Determine why the project was delayed by examining files in the project folder..."*
   - **Timeline Constructions:** *"Trace the evolution from the initial proposal document through the final implementation files..."*
   - **Problem-Solution Formats:** *"Given the discrepancies between our planning documents and the actual deliverables..."*
   - **Hypothesis Testing:** *"Verify whether the cost assumptions in our budget spreadsheets held true by examining..."*

2. **Opening Variation Examples:**
   - Instead of always "Search my files..." use:
     - *"What figures appear in yesterday's budget spreadsheet about..."*
     - *"Pull up documents where the migration plan was outlined..."*
     - *"Trace back through recent files on..."*
     - *"Reconstruct the project timeline from documents in..."*
     - *"Check what was documented in the compliance report..."*
     - *"Cross-reference last week's presentation with..."*
     - *"Verify the implementation against our planning documents..."*
     - *"Compare the performance data in Excel files with..."*
     - *"Look back at files created during the project and..."*
     - *"Review what was documented in the architecture diagrams..."*

3. **File-Focused Language:** Always include file-related actions but vary the phrasing:
   - Direct search: *find files*, *search my documents*, *review files*, *look through documents*
   - Contextual: *in my files*, *from my documents*, *through my file system*, *among my files*
   - Action-focused: *files containing [topic]*, *documents modified between [dates]*, *files with specific extensions*
   - Analytical: *based on my files*, *according to the document*, *files where [criteria] appears*
   - Workflow: *find all versions of*, *summarize documents*, *list files from [folder]*

4. **Filename/Type Context:** Reference actual filenames and file types from the data:
   - *the Q3_Budget.xlsx file*
   - *PowerPoint presentations about deployment*  
   - *PDF compliance reports*
   - *Word documents in the project folder*

5. **Technical Specificity:** Use actual topics, projects, and content from the data:
   - *budget approval process*
   - *Azure migration timeline*
   - *compliance audit findings*
   - *performance metrics analysis*

6. **Temporal Precision:** Include realistic timestamps and natural time references:
   - Utterance text: *files from last week* or *yesterday's document*
   - Query_Timestamp: ISO 8601 format (e.g., `2025-07-05T14:00:00Z`)

7. **Structural Variety:** **Avoid formulaic patterns**. Mix:
   - **Short, direct queries** (20-30 words)
   - **Complex, multi-clause requests** (60-80 words)
   - **Conditional/hypothetical structures**
   - **Comparative analysis requests**
   - **Root-cause investigation formats**

8. **Diversity & Quantity:** Generate **as many unique, high-quality utterances as possible**
9. **Conciseness:** Keep utterances focused but intentionally complex
10. **Enterprise Realism:** Use actual filenames, file types, and scenarios from provided file data
11. **Plain Text Format:** Write all utterances in plain text **without any quotation marks, backticks, or wrapping**. Use single quotes (') sparingly and only when referencing specific filenames or technical terms within the utterance.

---

## **Example Utterances - Demonstrating Syntactic Diversity & Natural Temporal References**

✅ **Question Format** - *What were the exact budget figures in yesterday's Q3_Budget.xlsx spreadsheet for the marketing department, and do they align with the constraints mentioned in the Finance_Guidelines.pdf document?* [Query_Timestamp: 2025-07-05T16:00:00Z]

✅ **Comparative Structure** - *Compare the deployment timeline outlined in last week's Azure_Migration_Plan.pptx with the actual milestones documented in the recent Project_Status.docx file* [Query_Timestamp: 2025-07-05T18:30:00Z]

✅ **Causal Investigation** - *Determine why the project was delayed by examining Excel files with budget data from the Finance folder and Word documents about resource allocation* [Query_Timestamp: 2025-07-05T14:00:00Z]

✅ **Conditional Analysis** - *If the compliance requirements documented in the Audit_Findings.pdf match what was planned in our Compliance_Strategy.docx, analyze whether the implementation was sufficient* [Query_Timestamp: 2025-07-05T15:30:00Z]

✅ **Timeline Reconstruction** - *Trace the evolution of our Q3 budget from the initial proposal in Budget_Draft_v1.xlsx through all subsequent versions to the final Budget_Final.xlsx* [Query_Timestamp: 2025-07-05T17:00:00Z]

✅ **Hypothesis Testing** - *Verify whether the cost savings assumptions in our planning spreadsheets actually materialized based on the financial reports in recent PDF documents* [Query_Timestamp: 2025-07-05T19:00:00Z]

✅ **Direct Imperative** - *Pull up PowerPoint presentations about the deployment strategy from last month and cross-check the technical requirements against the actual configuration documented in recent files* [Query_Timestamp: 2025-07-05T16:45:00Z]

✅ **Natural Event Anchoring** - *Look back at files created during the compliance review period and verify whether the audit recommendations were addressed in subsequent documentation* [Query_Timestamp: 2025-07-05T17:30:00Z]

❌ **Avoid These Anti-Patterns:**
- ❌ **Precise timestamps in utterance:** *"Find files modified on 2025-06-02 at 10:15"* - too specific in text
- ❌ **Repetitive openings:** *"Search my files for..."* appearing in every utterance - overused 
- ❌ **Generic queries:** *"What happened with the project?"* - no search intent, no file context
- ❌ **No temporal context:** Missing natural time references like *yesterday* or *last week*
- ❌ **Weak anchors:** *"Find files about issues"* - too vague, needs filenames or file types

---

## **Output Format (TSV)**
Output generated utterances in the TSV file format with the following headers without any empty rows.
```
Utterance    Email_Account    Query_Timestamp    Complexity_Reason
```
- **Utterance:** A complex file-search request written in plain text using natural temporal expressions (do **not** use quotation marks).
- **Email_Account:** Provided account (e.g., `marion.chen`).
- **Query_Timestamp:** Realistic timestamp in ISO 8601 format (e.g., `2025-07-05T14:00:00Z`) that is at least 3+ hours after the latest file modification referenced in the utterance.
- **Complexity_Reason:** Short note on why multi-step reasoning is needed (e.g., "Requires correlating data across 3 Excel files and 2 PDF reports with timeline analysis").

---

## **Input Data**
Email Account: `{email_account}`  
File Data:  
```
{file_json_data}
```

---

## **Goal**
Produce **as many unique, high-quality utterances as possible** that:
- Explicitly direct the model to **search files**.
- Use **diverse sentence structures** (avoid template patterns).
- Use **natural temporal expressions** in utterance text.
- Include **realistic query timestamps** in ISO 8601 format.
- Require **iterative reasoning** and **strategic keyword planning**.
- Cannot be answered by simple keyword matching or single-file summarization.
- Leverage actual file content (filenames, file types, topics, modification times) from provided data.
