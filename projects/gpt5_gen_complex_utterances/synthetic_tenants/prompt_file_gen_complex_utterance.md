# **Complex Utterance Generator for Reasoning Model**

## **Objective**
Generate concise, diverse, and **high-complexity user utterances** based on the provided enterprise file data. Each utterance should:
- Explicitly indicate the need to **search files**.
- Simulate real-world enterprise scenarios using provided content.
- Require reasoning steps where reasoning models outperform standard chat models, as tested in **BizChat Copilot**.

---

## **Testing Context**  
BizChat Copilot answers user queries using either a **chat model** or a **reasoning model**. Typically, **simple utterances** are handled by the chat model, while **complex utterances** invoke the reasoning model. Our goal is to generate utterances that strongly steer the system toward using the reasoning model.  

Complex utterances generally:  
- Require **Chain-of-Thought (CoT) reasoning** to interpret and respond effectively  
- Require **strategic multi-step search planning**  
- Require **query refinement for accurate file retrieval**  
- Require **multi-step reasoning to integrate insights across multiple files**  

---

## **Environment & Constraints**  
- **Search Tools:** Dedicated tools are available for emails, chats, files, calendars, and web content. For these utterances, the model must correctly select the **file search tool** and craft **optimized queries** that retrieve the most relevant files to produce a grounded, accurate response.  
- **Search Behavior:**  
  - Each query can retrieve up to **10 results**; multiple queries are often necessary for comprehensive coverage.  
  - Both models have equal access to these tools but must plan and adjust searches iteratively based on intermediate results.  
- **Response Generation:** The model must synthesize answers solely from retrieved content. Raw file content is **not directly accessible to the model during input processing in BizChat Copilot**, so it must infer search strategies from the user's utterance. 

---

## **Critical Considerations**  

1. **Keyword Sensitivity and Search Optimization**  
   - **Explicit Search Intent:** Every utterance must clearly instruct a **file search**, using verbs like *search my files*, *find documents*, *look through my files*, *review documents*. Ambiguous phrasing that could be answered without searching (e.g., *What happened with the Q3 budget?*) must be avoided.  
   - **Metadata Anchors Matter:** File search engines are most sensitive to **keywords within filenames, complete filenames, file types, and modification dates**, which outperform content-only terms. Utterances should therefore include one or more strong anchors such as:  
     - Complete filenames (*the Q3_Budget.xlsx file*, *document named 'Azure_Migration_Plan.docx'*)
     - Filename keywords (*files with 'budget' in the name*, *documents containing 'migration' in filename*)  
     - File types (*Excel spreadsheets*, *PDF reports*, *PowerPoint presentations*)   
     - Time frames (*files modified last week*, *documents created between January and March*)  
   - **Avoid Weak Keywords:** Generic content references like *audit issues* without context lead to poor matches. Instead use: *files containing audit findings from the Finance folder last quarter* or *in the 'Compliance Review' documents*.  
   - **Encourage Iterative Retrieval:** If the question spans multiple file types or folders, signal this explicitly, for example: *first check Excel files for budget data, then find related PowerPoint presentations*.  
   **Bottom Line:** Utterances should resemble well-formed file search instructions rather than casual questions.  

2. **Reality Gap:** During generation of complex utterances, you have full visibility into the file dataset. However, during BizChat's execution, both models see only the user's utterance—they have **no direct access to the underlying file content**. Instead, the model must infer **what to search for, which keywords to prioritize, and how to refine its queries iteratively** based on the original utterance and the snippets retrieved from prior searches. 

3. **Reasoning Requirement:** Utterances should demand **beyond single-lookups**, requiring multiple searches and additional reasoning steps such as:  
   - Correlating data from separate files across different folders  
   - Building timelines or detecting dependencies from document sequences  
   - Combining partial information from different file types into one answer  

4. **Performance Signal:** The strength of the test lies in **how many reasoning steps are required** (temporal sequencing, causal inference, query optimization). The deeper the reasoning chain, the clearer the advantage of reasoning models over chat models.  

---

## **Success Criteria**
A valid complex utterance:
1. Clearly **demands file searching**.
2. Includes **search-friendly cues** (filenames, file types, time frames).
3. Requires **iterative steps** (can't be answered with one simple lookup).
4. Forces reasoning like **temporal logic, correlation, integration**, or **causal deduction**.
5. Forces **search refinement** if naive keywords fail.
5. Engages reasoning skills like **timeline deduction, correlation across sources, constraints handling**.

---

## **Targeted Reasoning Advantages**
Prioritize scenarios involving:
- **Temporal Logic & Dependencies** (project timelines, document versions).
- **Cross-File Integration** (data spread across separate documents).
- **Causal Inference** (e.g., "What led to changes in...").
- **Pattern Recognition** (trends across multiple reports).
- **Search Strategy & Optimization** (choosing critical keywords over generic ones).

---

## **Generation Rules**
1. **File-Focused Language:** Always mention file-related actions to steer toward the file search tool. Use varied phrasings such as:
   - Direct search: *find files*, *search my documents*, *review files*
   - Contextual: *in my files*, *from my documents*, *through my file system*
   - Action-focused: *files containing [topic]*, *documents modified between [dates]*, *files with specific extensions*
   - Analytical: *based on my files*, *documents where [criteria] appears*
   - Workflow: *find all versions of*, *summarize documents*, *list files from [folder]*
2. **Diversity:** Cover a wide range of reasoning challenges.
3. **Conciseness:** Keep utterances short but intentionally complex.
4. **Enterprise Context:** Incorporate real-world elements like project names, compliance documents, reports, budgets, and timelines (from provided file data).
5. **Search-Friendly:** Include filenames, file types, dates for effective keyword extraction.
6. **Challenge the Model:** Make questions beyond pattern matching—require tactical reasoning steps.

---

## **Example Utterances**
✅ Good:
- Find Excel files from the Finance folder about Q3 budget that mention compliance issues  
- Search my documents for PowerPoint presentations discussing Azure migration timeline conflicts  
- Check PDF reports created last week related to audit findings and any follow-up documentation  

❌ Avoid:
- What happened with the Q3 budget compliance issues? *(too generic)*  
- Tell me about Azure migration conflicts *(doesn't indicate a file search)*  

---

## **Output Format (TSV)**
Output generated utterances in the TSV file format with the following headers without any empty rows.
```
Utterance    Email_Account    Complexity_Reason
```
- **Utterance:** A complex file-search request written in plain text (do **not** use quotation marks).
- **Email_Account:** Provided account (e.g., `marion.chen`).
- **Complexity_Reason:** Short note on why multi-step reasoning is needed.
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
- Require **iterative reasoning** and **strategic keyword planning**.
- Cannot be answered by simple keyword matching or summarization.
