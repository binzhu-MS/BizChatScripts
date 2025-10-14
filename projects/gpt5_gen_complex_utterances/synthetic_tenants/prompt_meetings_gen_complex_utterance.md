# **Complex Utterance Generator for Meetings & Events**

## **Objective**
Generate concise, diverse, and **high-complexity user utterances** based on the provided meetings, events, transcripts, and meeting chat data. Each utterance should:
- Explicitly indicate the need to **search meeting transcripts**, **calendar events**, and **meeting-type chats**.
- Simulate real-world enterprise scenarios using formal meeting discussions, transcript content, event details, and related files.
- Require reasoning steps where reasoning models outperform standard chat models in **BizChat Copilot**.

---

## **Testing Context**  
BizChat Copilot answers user queries using either a **chat model** or a **reasoning model**. Typically, **simple utterances** are handled by the chat model, while **complex utterances** invoke the reasoning model. Our goal is to generate utterances that strongly steer the system toward using the reasoning model.  

Complex utterances generally:  
- Require **Chain-of-Thought (CoT) reasoning** to interpret and respond effectively  
- Require **strategic multi-step search planning** for meeting transcripts, events, and meeting chats
- Require **temporal reasoning** about meeting timelines, decisions, and follow-up actions
- Require **cross-source correlation** between transcript discussions, event details, and chat follow-ups
- Require **speaker attribution and sentiment analysis** from meeting transcripts

---

## **Environment & Constraints**  
- **Search Tools:** Dedicated tools are available for meeting transcripts, calendar events, emails, files, and Teams chats. For these utterances, the model must correctly select **meeting transcript search**, **event search**, and **meeting chat search**.
- **Meeting Context:**
  - **Transcripts:** Full VTT transcripts with speaker attribution and timestamps
  - **Events:** Calendar invites with subjects, attendees, attachments, and body content
  - **Meeting Chats:** Chat conversations that occurred during meetings
  - **Correlation:** Many meetings have associated transcripts, events, and chats that can be cross-referenced
- **Search Behavior:**  
  - Each query can retrieve up to **10 results**; multiple queries are often necessary for comprehensive coverage.  
  - Both models have equal access to these tools but must plan and adjust searches iteratively based on intermediate results.
- **Temporal Constraints:** Queries can only find meetings and events that occurred **before the query timestamp**. Generate realistic query times that are **at least a few hours after** the latest meeting or event mentioned.
- **Response Generation:** The model must synthesize answers solely from retrieved content.

---

## **Critical Considerations**  

1. **Syntactic Variety Mandate - AVOID TEMPLATE PATTERNS**
   - **FORBIDDEN:** Starting every utterance with "Search our meetings..." or similar formulaic openings
   - **REQUIRED:** Use diverse sentence structures, question formats, imperatives, conditionals, comparatives
   - **Pattern Breaking:** If you notice repetitive phrasing, immediately switch to a completely different grammatical structure
   - **Creativity Over Formula:** Prioritize natural, varied language over predictable templates

2. **Meeting Search Optimization**  
   - **Explicit Meeting Search Intent:** Every utterance must clearly instruct a **meeting/transcript search**, using phrases like *review our meeting about*, *check the transcript from*, *find our discussion on*, *locate the session where we talked about*. 
   - **Search Engine Intelligence:** Provide **rich contextual clues** rather than precise timestamps:
     - **Participants:** *in Marion's presentation*, *when John discussed*, *during the team review*
     - **Topics/themes:** *about performance metrics*, *regarding the architecture decision*, *when we reviewed the incident*
     - **Meeting context:** *Sprint Planning*, *Post-Mortem Review*, *Architecture Deep Dive*
     - **Natural time references:** *last week's standup*, *yesterday's retrospective*, *our recent planning session*
     - **Technical anchors:** *deployment strategy*, *incident response*, *performance targets*
   - **Trust Search Capability:** Modern search engines can correlate participants + topics + timeframes
   - **Cross-Reference:** Signal need to connect meeting transcripts with events and chats: *find the decisions from our meeting and check if they match the follow-up actions in the chat...*

3. **Event & Calendar Integration**  
   - **Event Search:** Reference calendar events by subject, attendees, or timeframe
   - **Attachment Access:** Many events have attached documents that provide additional context
   - **Meeting-Event Correlation:** Connect transcript discussions with event details and attachments

4. **Meeting Chat Analysis**  
   - **Chat Correlation:** Meeting-type chats often contain follow-up discussions, decisions, and action items
   - **Participant Overlap:** Analyze how meeting participants continued discussions in chat
   - **Decision Tracking:** Follow decisions from meeting transcripts to chat confirmations

5. **Reality Gap:** During generation, you have full visibility into meetings and events. However, during BizChat's execution, models see only the user's utterance—they have **no direct access to the underlying content**. The model must infer **what to search for, which participants to focus on, and how to connect meetings with events** based on the utterance and retrieved snippets.

6. **Reasoning Requirement:** Utterances should demand **beyond single-lookups**, requiring multiple searches and reasoning steps such as:  
   - Correlating meeting discussions with event details
   - Building timelines from multiple meetings and follow-up chats
   - Detecting discrepancies between meeting decisions and actual outcomes
   - Synthesizing insights from transcripts, events, and chats

---

## **Natural Temporal References in Utterances**
Generate utterances with **natural, human-like temporal expressions** instead of precise timestamps:

### **Realistic Time Expressions:**
- **Recent references:** *yesterday's standup*, *this morning's review*, *our meeting earlier today*
- **Relative periods:** *last week's sprint planning*, *Monday's retrospective*, *our recent architecture session*
- **Event-anchored:** *before the release*, *after the incident*, *during the design review*, *when we discussed the roadmap*
- **Seasonal/contextual:** *Q3 planning session*, *annual review*, *post-mortem discussion*

### **Search Engine Reliance:**
Users provide **contextual clues** rather than exact timestamps:
- **Participants:** *in the meeting with Marion and Troy*  
- **Topics/keywords:** *about performance optimization*, *regarding the incident response*
- **Meeting types:** *Sprint Planning*, *Design Review*, *Post-Mortem*
- **Technical context:** *when we analyzed the metrics*, *during the architecture discussion*

### **Query Timestamp Logic:**
Each utterance needs a **realistic query timestamp** for the TSV output that:
1. **Comes after** the latest referenced content (minimum 3+ hours gap)
2. **Uses ISO 8601 format** for consistency: `2025-07-05T14:00:00Z`
3. **But the utterance text itself uses natural language** for temporal references

---

## **Success Criteria**
A valid complex utterance:
1. Clearly **demands meeting/transcript search** and optionally event or chat searches.
2. Includes **search-friendly cues** (participant names, meeting topics, event subjects, speaker names).
3. Requires **iterative steps** (can't be answered with one simple lookup).
4. Forces reasoning like **temporal logic, speaker attribution, decision tracking**, or **cross-source correlation**.
5. **Strategically combines** transcript content with event details and meeting chats.
6. Includes **realistic query timestamp** after referenced content.
7. Engages reasoning skills like **timeline deduction, decision tracking, discrepancy detection**.

---

## **Targeted Reasoning Advantages**
Prioritize scenarios involving:
- **Meeting-Event-Chat Cross-Reference** (connecting formal meetings with calendar events and follow-up chats)
- **Speaker Attribution & Analysis** (identifying who said what, tracking opinions across speakers)
- **Decision Timeline** (tracing decisions from meeting transcripts to implementation)
- **Multi-Meeting Synthesis** (combining insights from multiple related meetings)
- **Action Item Tracking** (following action items from meetings to completion)
- **Discrepancy Detection** (comparing planned actions with actual outcomes)
- **Consensus Building** (analyzing how team reached decisions across multiple sessions)

---

## **Generation Rules**

1. **Strategic Content Access:** Combine meeting transcript search with event and chat searches:
   - **Transcript Focus:** *"Review the transcript from our Sprint Planning meeting where Marion discussed the deployment timeline and compare with..."*
   - **Event Integration:** *"Find the calendar event for our Architecture Review and analyze the attached design documents alongside the transcript discussion"*
   - **Chat Follow-up:** *"Check what decisions we made in yesterday's retrospective and verify if they were confirmed in the meeting chat"*
   - **Hybrid Approach:** *"Trace the evolution of our incident response strategy across multiple post-mortem meetings and related chat discussions"*

2. **Syntactic Diversity - CRITICAL:** **Vary sentence structures dramatically** to avoid template-like repetition:
   - **Question Formats:** *"What was the consensus reached in our planning meeting about the deployment strategy, and did everyone agree?"*
   - **Imperative Variations:** *"Pull up the transcript from our architecture session and identify..."*, *"Locate the meeting where we discussed..."*, *"Retrieve the session notes from..."*
   - **Conditional Structures:** *"If Marion's proposal from Monday's meeting was accepted, then trace how it was implemented..."*
   - **Comparative Frames:** *"Compare what we decided in our Q3 planning versus what we actually delivered..."*
   - **Causal Inquiries:** *"Determine why the project was delayed by examining both planning meetings and follow-up chats"*
   - **Timeline Constructions:** *"Trace the decision-making process from initial brainstorming through final approval..."*
   - **Problem-Solution Formats:** *"Given the concerns raised in our retrospective, what solutions did the team propose?"*
   - **Hypothesis Testing:** *"Verify whether our planning meeting assumptions about performance held true by..."*

3. **Opening Variation Examples:**
   - Instead of always "Search our meetings..." use:
     - *"What did Marion propose in yesterday's architecture review..."*
     - *"Pull up the session where we discussed incident response..."*
     - *"Trace back through our sprint planning meetings to find..."*
     - *"Reconstruct the decision process from our stakeholder review..."*
     - *"Check what consensus we reached in our retrospective..."*
     - *"Cross-reference our planning session with the actual outcomes..."*
     - *"Verify the action items against our meeting decisions..."*

4. **Speaker-Aware Queries:**
   - Reference specific speakers: *"What were John's concerns in the security review?"*
   - Track opinions: *"How did team sentiment shift between the first and second planning sessions?"*
   - Attribution: *"Who proposed the alternative approach in our design meeting?"*

5. **Meeting Type Targeting:**
   - **Sprint Planning:** Focus on roadmap, priorities, capacity planning
   - **Retrospectives:** Emphasize lessons learned, process improvements, team dynamics
   - **Post-Mortems:** Incident analysis, root causes, preventive measures
   - **Architecture Reviews:** Design decisions, technical trade-offs, scalability
   - **Stakeholder Updates:** Project status, risks, deliverables

6. **Complexity Layers:** Each utterance should require at least 2-3 of:
   - Multiple meeting transcript searches
   - Correlation with calendar events
   - Analysis of meeting chat follow-ups
   - Cross-speaker opinion synthesis
   - Timeline construction across sessions
   - Decision-to-outcome validation

7. **Realism & Specificity:**
   - Use actual participant names from the data
   - Reference real meeting topics and subjects
   - Incorporate technical terms from transcripts
   - Mention specific action items or decisions
   - Include natural time expressions

8. **Avoid Repetition:** Generate **at least 10-20 distinct utterances** with:
   - Different meeting types
   - Various reasoning patterns
   - Diverse participant combinations
   - Multiple temporal scopes
   - Varied complexity layers

---

## **Output Format (TSV)**

Generate utterances in TSV format with the following columns:

```
Utterance	Email_Account	Query_Timestamp	Complexity_Reason
```

### **Column Specifications:**

1. **Utterance**: The complex query text in **plain text with NO quotes or backticks**. Use natural temporal expressions like "yesterday's sprint planning" or "last week's review meeting".
2. **Email_Account**: Target user's email (use `{email_account}`)
3. **Query_Timestamp**: ISO 8601 timestamp **after** latest referenced content (format: `2025-07-05T14:00:00Z`)
4. **Complexity_Reason**: Brief explanation of why this requires multi-step reasoning across meetings, events, and transcripts (2-3 sentences)

**CRITICAL:** Do NOT wrap utterances in double quotes ("), backticks (`), or any other quotation marks. Write each utterance as plain, natural text.

### **Example Output:**

```tsv
Utterance	Email_Account	Query_Timestamp	Complexity_Reason
What was the consensus in yesterday's sprint planning about the deployment timeline, and did Marion's concerns about infrastructure readiness get addressed in the follow-up chat?	{email_account}	2025-06-15T14:00:00Z	Requires correlating meeting transcript discussions with chat follow-ups, tracking speaker-specific concerns, and synthesizing team consensus across multiple communication channels.
Compare the incident response strategy we discussed in last week's post-mortem with what actually happened during this week's outage based on our standup transcripts	{email_account}	2025-06-20T10:00:00Z	Demands temporal reasoning across multiple meetings, comparison of planned vs actual responses, and synthesis of lessons learned from different incident discussions.
```

---

## **Assigned User Email Account**

**Target User**: `{email_account}`

Generate utterances as if this user is asking questions that require searching through their meeting transcripts, calendar events, and meeting chats. The user can only ask about:
- Meetings they participated in or attended
- Events where they were an attendee
- Meeting chats they were part of
- Topics, decisions, and discussions from meetings they joined

---

## **Input Data - Meetings, Events, and Transcripts for Target User**

The following data represents **ONLY** the meetings, events, transcripts, and meeting chats that the target user `{email_account}` participated in or has access to. Generate utterances based exclusively on this accessible content.

**Data Context:**
- **Meetings with Transcripts:** Meeting IDs, participants, speaker lists, transcript content excerpts
- **Events with Details:** Event IDs, subjects, attendees, body content, attachments
- **Meeting Chat Correlations:** Chat IDs, message counts, participant overlaps, chat message excerpts
- **Statistics:** Overall counts and participant summaries

```json
{meetings_events_json_data}
```

---

## **Important Reminders**

1. **Natural Language First:** Write utterances as real users would ask them (natural temporal expressions)
2. **Search-Friendly:** Include enough context for search engines to find relevant meetings
3. **Reasoning-Focused:** Ensure each utterance requires multi-step reasoning
4. **Syntactic Variety:** Never repeat the same opening pattern
5. **Data-Grounded:** Use actual participant names, meeting subjects, and topics from the provided data
6. **Realistic Timestamps:** Query timestamps should be logically after the meetings/events referenced
7. **Complexity Justification:** Clearly explain the reasoning steps required

---

## **Begin Generation**

Based on the provided meeting and event data, generate diverse, complex utterances that require reasoning capabilities to answer effectively.
