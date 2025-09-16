# **Complex Utterance Generator for Reasoning Model**

## **Objective**
Generate concise, diverse, and **high-complexity user utterances** based on the provided enterprise email data. Each utterance should:
- Explicitly indicate the need to **search emails**.
- Simulate real-world enterprise scenarios using provided content.
- Require reasoning steps where reasoning models outperform standard chat models, as tested in **BizChat Copilot**.

---

## **Testing Context**  
BizChat Copilot answers user queries using either a **chat model** or a **reasoning model**. Typically, **simple utterances** are handled by the chat model, while **complex utterances** invoke the reasoning model. Our goal is to generate utterances that strongly steer the system toward using the reasoning model.  

Complex utterances generally:  
- Require **Chain-of-Thought (CoT) reasoning** to interpret and respond effectively  
- Require **strategic multi-step search planning**  
- Require **query refinement for accurate email retrieval**  
- Require **multi-step reasoning to integrate insights across multiple emails**  

---

## **Environment & Constraints**  
- **Search Tools:** Dedicated tools are available for emails, chats, files, calendars, and web content. For these utterances, the model must correctly select the **email search tool** and craft **optimized queries** that retrieve the most relevant emails to produce a grounded, accurate response.  
- **Search Behavior:**  
  - Each query can retrieve up to **10 results**; multiple queries are often necessary for comprehensive coverage.  
  - Both models have equal access to these tools but must plan and adjust searches iteratively based on intermediate results.  
- **Response Generation:** The model must synthesize answers solely from retrieved content. Raw email text is **not directly visible to the model during input processing in BizChat Copilot**, so it must infer search strategies from the user’s utterance.  

---

## **Critical Considerations**  

1. **Keyword Sensitivity and Search Optimization**  
   - **Explicit Search Intent:** Every utterance must clearly instruct an **email search**, using verbs like *search my inbox*, *find emails*, *look through email threads*. Ambiguous phrasing that could be answered without searching (e.g., *What happened with the Q3 budget?*) must be avoided.  
   - **Metadata Anchors Matter:** Email search engines are most sensitive to **subjects, senders, and dates**, which outperform body-only terms. Utterances should therefore include one or more strong anchors such as:  
     - Specific people (*emails from Sarah*, *messages John sent yesterday*)  
     - Subject lines or projects (*Q3 Budget review*, *Azure migration plan*)  
     - Time frames (*last week*, *between January and March*)  
   - **Avoid Weak Keywords:** Generic content references like *audit issues* without context lead to poor matches. Instead use: *emails about audit findings from Finance last quarter* or *in the ‘Compliance Review’ thread*.  
   - **Encourage Iterative Retrieval:** If the question spans multiple threads or steps, signal this explicitly, for example: *first check Sarah’s audit emails, then find related compliance follow-ups*.  
   - **Mailbox Scoping:** Use phrases like *in my inbox*, *from Sent Items*, or *in my threads* to reinforce email-tool invocation.  
   **Bottom Line:** Utterances should resemble well-formed email search instructions rather than casual questions.  

2. **Reality Gap:** During generation of complex utterances, you have full visibility into the email dataset. However, during BizChat’s execution, both models see only the user’s utterance—they have **no direct access to the underlying email content**. Instead, the model must infer **what to search for, which keywords to prioritize, and how to refine its queries iteratively** based on the original utterance and the snippets retrieved from prior searches. 

3. **Reasoning Requirement:** Utterances should demand **beyond single-lookups**, requiring multiple searches and additional reasoning steps such as:  
   - Correlating decisions from separate email threads  
   - Building timelines or detecting dependencies  
   - Combining partial facts from different senders into one answer  

4. **Performance Signal:** The strength of the test lies in **how many reasoning steps are required** (temporal sequencing, causal inference, query optimization). The deeper the reasoning chain, the clearer the advantage of reasoning models over chat models.  

---

## **Success Criteria**
A valid complex utterance:
1. Clearly **demands email searching**.
2. Includes **search-friendly cues** (people’s names, projects, time frames).
3. Requires **iterative steps** (can’t be answered with one simple lookup).
4. Forces reasoning like **temporal logic, correlation, integration**, or **causal deduction**.
5. Forces **search refinement** if naive keywords fail.
5. Engages reasoning skills like **timeline deduction, correlation across sources, constraints handling**.

---

## **Targeted Reasoning Advantages**
Prioritize scenarios involving:
- **Temporal Logic & Dependencies** (project timelines, deadlines).
- **Cross-Thread Integration** (decisions spread across separate chains).
- **Causal Inference** (e.g., “What led to…”).
- **Pattern Recognition** (trend across multiple audits).
- **Search Strategy & Optimization** (choosing critical keywords over generic ones).

---

## **Generation Rules**
1. **Email-Focused Language:** Always mention email-related actions to steer toward the email search tool. Use varied phrasings such as:
   - Direct search: *find emails*, *search my inbox*, *review emails*
   - Contextual: *in my emails*, *from my inbox*, *through email interactions*
   - Action-focused: *emails mentioning [topic]*, *emails between [dates]*, *emails with attachments*
   - Analytical: *based on my emails*, *emails where I’m mentioned*
   - Workflow: *catch me up on emails*, *summarize emails*, *list emails from [sender]*
2. **Diversity:** Cover a wide range of reasoning challenges.
3. **Conciseness:** Keep utterances short but intentually complex.
4. **Enterprise Context:** Incorporate real-world elements like project names, compliance issues, audits, budgets, and timelines (from provided email data).
5. **Search-Friendly:** Include names, subjects, dates for effective keyword extraction.
6. **Challenge the Model:** Make questions beyond pattern matching—require tactical reasoning steps.

---

## **Example Utterances**
✅ Good:
- Find emails from John about the Q3 budget that mention compliance issues  
- Search my inbox for emails discussing the Azure migration timeline conflicts  
- Check emails Sarah sent last week related to the audit findings and any follow-up from Finance  

❌ Avoid:
- What happened with the Q3 budget compliance issues? *(too generic)*  
- Tell me about Azure migration conflicts *(doesn’t indicate an email search)*  

---

## **Output Format (TSV)**
Output generated utterances in the TSV file format with the following headers without any empty rows.
```
Utterance    Email_Account    Complexity_Reason
```
- **Utterance:** A complex email-search request written in plain text (do **not** use quotation marks).
- **Email_Account:** Provided account (e.g., `marion.chen`).
- **Complexity_Reason:** Short note on why multi-step reasoning is needed.
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
- Require **iterative reasoning** and **strategic keyword planning**.
- Cannot be answered by simple keyword matching or summarization.
