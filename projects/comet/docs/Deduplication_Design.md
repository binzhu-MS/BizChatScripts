# Search Result Deduplication — Design Document

This document describes the deduplication strategy for identifying duplicate search results across iterations in the Retrieved Good Gain metrics.

---

## **Executive Summary**

The deduplication strategy uses a **two-phase approach**:

**Phase 1: Candidate Generation (High Recall)**
- Extract ALL available individual IDs from each result
- Create separate keys for each: `fileid:`, `eventid:`, `url:`, `snippet:`, `title:`, etc.
- **Aggressive matching:** Any single key match -> candidate pair (high recall; candidates are later filtered in Phase 2)

**Phase 2: Verification (High Precision)**
- Definitive ID component match (type-specific, item-level IDs)
- URL match with minimal normalization (strip known irrelevant params only)
- Field-based verification with exact field comparisons

**Key Design Decisions:**
- Do NOT use `reference_id` (it's a citation marker, not a document ID)
- Aggressive URL normalization for Phase 1, minimal for Phase 2
- All original field values preserved for debugging/traceability

---

## **1. Two-Phase Deduplication Strategy**

### **1.1 Overview**

Deduplication uses a **two-phase approach** to balance high recall (find all duplicates) with high precision (avoid false matches):

Phase 1 is intentionally **aggressive**: any single key match forms a candidate pair.

```
Phase 1: Candidate Generation (High Recall)
    └── Extract ALL available IDs from each result
    └── Generate content-based hashes (snippet, title, content)
    └── Create separate keys for each ID and hash
    └── Any single key match -> candidate pair

Phase 2: Verification (High Precision, rejects false positives from Phase 1)
    └── Strict ID component match OR
    └── Field-based verification (exact field comparisons)
```

### **1.2 Why Two Phases?**

| Approach | Problem |
|----------|---------|
| Single-phase exact matching | Misses duplicates when IDs/URLs differ slightly |
| Single-phase fuzzy matching | Creates false positives |
| **Two-phase** | Balances recall and precision |

### **1.3 Key Principles**

1. **Multi-ID Extraction**: Extract ALL available unique IDs from each result (not just one)
2. **Aggressive Phase 1**: Any single ID match creates a candidate pair
3. **Strict Phase 2**: ID component/URL match OR field-based verification (exact comparisons) required to confirm
4. **Preserve Original Data**: All original field values are kept for debugging/traceability

**⚠️ CRITICAL: Do NOT use `reference_id`**

The `reference_id` field in raw SEVAL results (e.g., "turn1search1", "turn2search3") is a **citation marker**, NOT a document identifier. The same document appearing in different queries will have different `reference_id` values, causing deduplication to fail.

---

## **2. Available Data in Search Results**

### **2.1 Entity Types and Their Fields**

Search results contain entities of various types, each with different fields available:

| Type | Description |
|------|-------------|
| EmailMessage | Email messages from Outlook |
| Event | Calendar events and meetings |
| File | Documents from SharePoint/OneDrive |
| TeamsMessage | Teams chat/channel messages |
| PeopleInferenceAnswer | People directory entries |
| External | External connector results |
| MeetingTranscript | Meeting transcriptions |
| Bookmark | Browser bookmarks |

### **2.2 Fields by Entity Type**

#### **2.2.1. Common Fields (most entity types)**
| Field | Description | Available For |
|-------|-------------|---------------|
| `Id` | Entity identifier | All |
| `Title` | Display title | All |
| `Snippet` | Text snippet/preview | Most |
| `ContentBody` | Full content (usually same as Snippet) | Most |
| `Type` | Entity type name (string, e.g., "Event", "EmailMessage") | All |
| `Url` | Resource URL | File, External, some others |

> **Note on Type field:** 
> - Raw SEVAL data contains a string `type` field (e.g., "Event", "EmailMessage")
> - The transform script preserves this string value directly as `Type`
> - `entity_type_utils.resolve_full_entity_type()` accepts both numeric ("2") and string ("Event") values

#### **2.2.2. EmailMessage Fields**
| Field | Description | Uniqueness |
|-------|-------------|------------|
| `DateTimeReceived` | Timestamp when received | High (with From/To) |
| `From` | Sender address | High |
| `To` | Recipient list | Medium |
| `Subject` | Email subject | Medium |
| `IsMentioned` | User mentioned | Low |
| `IsRead` | Read status | Low |

#### **2.2.3. Event Fields**
| Field | Description | Uniqueness |
|-------|-------------|------------|
| `Start` | Event start time | High (with End+Organizer) |
| `End` | Event end time | High |
| `OrganizerEmail` | Meeting organizer | High |
| `OrganizerName` | Organizer display name | Medium |
| `Invitees` | Attendee list | Low (can change; do not use for dedup decisions) |
| `Subject` | Event title | Medium |
| `IsMeetingTranscribed` | Has transcript | Low |
| `MeetingChat` | Associated chat | Low |
| `MeetingInvitationContent` | Invitation body | Medium |

#### **2.2.4. File Fields**
| Field | Description | Uniqueness |
|-------|-------------|------------|
| `FileName` | File name | Medium |
| `FileType` | File extension/type | Low |
| `Author` | Document author | Medium |
| `LastModifiedBy` | Last editor | Medium |
| `LastModifiedTime` | Modification timestamp | Medium |
| `Url` | SharePoint/OneDrive URL | High |
| `Source` | Contains domain-specific IDs | High |

#### **2.2.5. Source Object (Domain-Specific IDs)**

Some datasets include a `Source` object containing domain-specific identifiers (often the most reliable dedup signals).

##### **2.2.5.1. Typical `Source` Fields**

Typical `Source` fields by entity type (not exhaustive):

| Entity Type | Example `Source` fields |
|---|---|
| File | `Source.FileId`, `Source.UniqueId`, `Source.SiteId`, `Source.WebId` |
| Event | `Source.EventId.Id`, `Source.OriginalId` |
| EmailMessage | `Source.EmailMessage.Url` |
| TeamsMessage | `Source.ItemId.Id`, `Source.Url` |

Example shape (illustrative - shows typical field names; values vary by dataset):

```
"Source": {
    "FileId": "SPO_...",
    "UniqueId": "89f5a370-3b17-4c19-9864-fd17ec2cc5ec",
    "SiteId": "17f010cc-cc0b-4ae7-b62c-60e01dd28f68",
    "WebId": "75203a7b-23a8-4cf5-97c9-6cd8f8f721e7"
}
```

*See Section 2.3 for SEVAL job-specific field availability notes.*

##### **2.2.5.2. Component ID Examples**

The following table shows **real examples** of component ID values and their formats:

| Component ID | Entity Type | Format Description | Example Value |
|--------------|-------------|-------------------|---------------|
| **FileId** | File | Base64-encoded SPO identifier with site/web/item info | `SPO_MTdmMDEwY2MtY2MwYi00YWU3LWI2MmMtNjBlMDFkZDI4ZjY4LDc1MjAzYTdiLTIzYTgtNGNmNS05N2M5LTZjZDhmOGY3MjFlNyw1ZDcwODZhNi1lNzhkLTQ0OWQtOTAxZS1lMTJmYzRhY2JkMDk_01X4XIIU3QUP2YSFZ3DFGJQZH5C7WCZRPM` |
| **UniqueId** | File | GUID identifying the file item | `89f5a370-3b17-4c19-9864-fd17ec2cc5ec` |
| **WebId** | File | GUID identifying the SharePoint web | `75203a7b-23a8-4cf5-97c9-6cd8f8f721e7` |
| **SiteId** | File | GUID identifying the SharePoint site collection | `17f010cc-cc0b-4ae7-b62c-60e01dd28f68` |
| **EventId.Id** | Event | Base64-encoded EWS item ID for specific occurrence | `AAMkADM4OTQ4MjQxLTU0NmQtNGNmOS04MjgxLWNjNDY2MWRjNzg4ZgFRAAgI3lFth4NAAEYAAAAAUtJIna4cykaUNYoMWW1SnwcA9Md4Rs7lk0GL5at7aoMRHgAABJWiVQAA...` |
| **OriginalId** | Event | Base64-encoded EWS item ID for recurring series master | `AAMkADM4OTQ4MjQxLTU0NmQtNGNmOS04MjgxLWNjNDY2MWRjNzg4ZgBGAAAAAABSokidrhzKRpQ1igxZbVKfBwD0x3hGzuWTQYvlq3tqgxEeAAAElZ...` |
| **EmailMessage.Url** | Email | OWA deep link URL to the email | `https://outlook.office365.com/owa/?ItemID=AAMkAGI2...&exvsurl=1&viewmodel=ReadMessageItem` |
| **ItemId.Id** | TeamsMessage | Unique identifier for the Teams message | `1703859234567` |
| **Url** | TeamsMessage | Deep link URL to the Teams message | `https://teams.microsoft.com/l/message/19:meeting_abc123@thread.v2/1703859234567` |
| **EmailAddresses[0].Address** | People | Primary email address | `johndoe@contoso.com` |
| **ArtifactId** | External | Graph connector artifact identifier | `9fe7d6d3-b8fe-4a2c-9d1e-3f4a5b6c7d8e` |

> **📝 ID Format Notes:**
> - **GUIDs** (UniqueId, WebId, SiteId, ArtifactId): Standard 36-character format `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
> - **Base64 IDs** (FileId, EventId, OriginalId): Variable length, may contain `+`, `/`, `=` characters
> - **EWS Item IDs** (EventId, OriginalId): Long Base64 strings typically starting with `AAMk...`
> - **URLs**: Full HTTP/HTTPS URLs with query parameters

#### **2.2.6. PeopleInferenceAnswer Fields**
| Field | Description | Uniqueness |
|-------|-------------|------------|
| `Alias` | User alias (e.g., "johnd") | Very High |
| `UserPrincipalName` | UPN (email format) | Very High |
| `EmailAddresses` | All email addresses | Very High |
| `DisplayName` | Full name | Medium |
| `Department` | Organization unit | Low |
| `CompanyName` | Company | Low |
| `OfficeLocation` | Office location | Low |

#### **2.2.7. TeamsMessage Fields**
| Field | Description | Uniqueness |
|-------|-------------|------------|
| `From` | Sender | High |
| `To` | Channel/recipient | Medium |
| `Participants` | Conversation participants | Medium |
| `Subject` | Thread subject | Medium |

#### **2.2.8. External (Connector) Fields**
| Field | Description | Uniqueness |
|-------|-------------|------------|
| `DisplayName` | Result title | Medium |
| `PluginId` | Connector ID | Low |
| `PluginType` | Connector type | Low |
| `Source` | Source metadata | Medium |

### **2.3 Content Fields Analysis**

> **Data Source:** Analysis based on SEVAL job #144683 results.

This section summarizes **what was (and was not) available** in the job and how content fields behave. It directly informs:
- Phase 1 key formation (Section 3)
- Phase 2 verification fallbacks (Section 4)

#### **2.3.1. Identifier Field Availability (SEVAL Job #144683)**

**Source availability:** In SEVAL job #144683, `Source` was **not present** on File/Event/EmailMessage/TeamsMessage entities. Treat all `Source.*` fields as missing and rely on `Url` / content / other fields.

**URL availability:**

| Entity Type | URL Available | Practical Dedup Fields When `Source` Missing |
|-------------|---------------|---------------------------------------------|
| File | ✅ Yes (SharePoint/Loop URL) | `url`, `title`, `fileName`, `author` |
| Event | ❌ No | `subject`, `start`, `end`, `organizerEmail` |
| EmailMessage | ❌ No | `subject`, `from`, `to`, `dateTimeReceived` |
| TeamsMessage | ❌ No | `title`, `from`, `participants` |

#### **2.3.2. Content Field Characteristics (SEVAL Job #144683)**

Average `ContentBody` length by entity type:

| Entity Type | ContentBody Avg Length | Notes |
|-------------|------------------------|-------|
| EmailMessage | 1,658 chars | Full email body |
| Event | 169 chars | Usually meeting invite or "You have not RSVP'd" |
| File | 1,676 chars | Document snippet/preview |
| TeamsMessage | 1,767 chars | Full message content |
| External | 960 chars | Connector-specific content |
| MeetingTranscript | 3,328 chars | Transcript excerpt |
| PeopleInferenceAnswer | 0 chars | No content (metadata only) |

**Key finding:** `ContentBody` and `Snippet` are identical in ~98% of cases.

#### **2.3.3. Implications for Deduplication**

- When `Source` is missing, many ID components (and canonical composite ID strings) may be unavailable; Phase 1 and Phase 2 must have URL/field/content fallbacks.
- Because `Snippet` ~ `ContentBody` in this job, content-based Phase 1 keys are largely redundant; hashing either works (we keep both for completeness).
- PeopleInferenceAnswer has no meaningful content, so dedup relies on direct identity fields (Alias/UPN/EmailAddresses) rather than content matching.

---

## **3. Phase 1: Candidate Generation (Key Formation)**

This section describes how to generate keys for candidate matching. We'll generate all keys, as long as available, for each result.
Any single key match -> candidate pair.

### **3.1 URL Key (All Entity Types)**

If a result includes a `Url`, generate a `url:` key using **aggressive** normalization.
This key is often the strongest available signal when domain-specific IDs are missing.

Some entity types (and some datasets) do not provide `Url` (e.g., Event/EmailMessage/TeamsMessage in SEVAL job #144683). In those cases, this key is simply omitted and Phase 1 relies on the other keys defined in Section 3.2 (ID keys) and Section 3.3 (content-based keys).

#### **Common (All Entity Types)**
| Key Format | Source Field | Example |
|------------|--------------|---------|
| `url:{netloc}\|{path}` | `Url` (aggressively normalized) | `url:loop.cloud.microsoft\|/p/eyJ1Ijoi...` |

**URL Normalization for Phase 1 (Aggressive):**
1. Parse the URL.
2. Use `netloc` + `path` only (drop the scheme).
3. Strip fragments (everything after `#`).
4. Strip ALL query parameters (everything after `?`).
5. Emit `url:{netloc}|{path}`.

This intentionally maximizes recall by collapsing common URL variants of the same resource.

**Illustrative Example (shows why Phase 1 is aggressive):**
```
Original URL A: https://contoso.sharepoint.com/sites/Team/Shared%20Documents/Doc.docx?web=1&version=5&x-unknown=abc#section
Original URL B: http://contoso.sharepoint.com/sites/Team/Shared%20Documents/Doc.docx?download=1&x-unknown=def

Phase 1 Key (A): url:contoso.sharepoint.com|/sites/Team/Shared%20Documents/Doc.docx
Phase 1 Key (B): url:contoso.sharepoint.com|/sites/Team/Shared%20Documents/Doc.docx

Note: In Phase 1 we strip ALL query parameters (including unknown ones) to maximize recall. Phase 2 is where we handle the hard case: unknown parameters are preserved unless we know they are content-unrelated.
```

### **3.2 Individual ID Keys (by Entity Type)**

For each result, extract **all available IDs** and create separate keys.

> **Note:** In SEVAL job #144683, `Source` was not present for several entity types (see Section 2.3). In those cases, these ID keys will be missing and Phase 1 relies more heavily on `url:` + content-based keys.

#### **File**
| Key Prefix | Field Path | Notes |
|------------|------------|-------|
| `fileid:` | `Source.FileId` | Item-level (definitive if present) |
| `uniqueid:` | `Source.UniqueId` | Item-level (definitive if present) |
| `webid:` | `Source.WebId` | Container-level (context only) |
| `siteid:` | `Source.SiteId` | Container-level (context only) |

#### **Event**
| Key Prefix | Field Path | Notes |
|------------|------------|-------|
| `eventid:` | `Source.EventId.Id` | Occurrence-level ID |
| `originalid:` | `Source.OriginalId` | Series master ID (recurrence-aware) |

#### **EmailMessage**
| Key Prefix | Field Path | Notes |
|------------|------------|-------|
| `emailurl:` | `Source.EmailMessage.Url` | Deep-link URL; behaves like an item ID |

#### **TeamsMessage**
| Key Prefix | Field Path | Notes |
|------------|------------|-------|
| `itemid:` | `Source.ItemId.Id` | Message ID |

#### **PeopleInferenceAnswer**
| Key Prefix | Field Path | Notes |
|------------|------------|-------|
| `email:` | `EmailAddresses[0].Address` | Prefer direct fields when present |
| `alias:` | `Alias` | Typically present directly on entity |
| `upn:` | `UserPrincipalName` | Typically present directly on entity |

#### **External (GraphConnector)**
| Key Prefix | Field Path | Notes |
|------------|------------|-------|
| `artifactid:` | `sourceJson.ArtifactId` | Connector document identifier |

### **3.3 Content-Based Keys**

To capture duplicates that may have different IDs/URLs but identical content, we use content-based hash keys:

**Implementation note:** The prefix lengths used for hashing are intentionally **tunable parameters**. Phase 1 is aggressive (high recall), so we can use shorter prefixes to reduce compute while still producing a useful fingerprint.

#### **Snippet Hash**
```
snippet:{hash(normalized_snippet[:100])}
```

Where `normalized_snippet` = lowercase, whitespace-collapsed snippet text.

**Prefix rationale:** The `snippet:` prefix ensures snippet hashes don't collide with ID-based keys.

#### **Title Hash**
```
title:{hash(normalized_title)}
```

Where `normalized_title` = lowercase, whitespace-collapsed title.

**Note:** Type is NOT included in Phase 1 title hash for aggressive matching. Cross-type collisions (e.g., email subject vs file name) will be rejected in Phase 2 verification.

#### **Content Hash**
```
content:{hash(normalized_content[:500])}
```

Use first 500 characters of ContentBody after normalization.

**Note:** ContentBody is identical to Snippet in ~98% of cases, but we hash both for completeness. This provides redundancy and catches the ~2% of cases where they differ.

### **3.4 Key Registration Summary**

For each entity, generate and register **all available keys** (any may be None):

| Key Type | Prefix | Description |
|----------|--------|-------------|
| Individual IDs | Various | Type-specific IDs (FileId, EventId, etc.) |
| URL | `url:` | Aggressively normalized URL |
| Snippet Hash | `snippet:` | Hash of normalized snippet |
| Content Hash | `content:` | Hash of normalized content body |
| Title Hash | `title:` | Hash of normalized title |

All keys are generated and registered — there is no prioritization. Any single key match creates a candidate pair.

---

## **4. Phase 2: Verification**

When Phase 1 finds candidate pairs, Phase 2 verifies they represent the same document.

**General rule:** Phase 2 applies the strongest available verification signals first (definitive ID components, then URL, then domain-specific fields). If a step requires fields that are missing for either result, that step is skipped and verification falls back to the next step.

### **4.1 ID Components and Canonical Composite IDs**

Phase 2 prefers **definitive item-level identifiers** (e.g., Event occurrence ID, Teams message ID). If a definitive ID component matches, it is sufficient to decide **Duplicate** (see Step 4.4.1).

In addition, when multiple ID components are available, we should build a **canonical composite ID string** for traceability and extra collision resistance in logs/telemetry. Conceptually, this canonical string represents “all available ID components” for an entity.

| Entity Type | Definitive ID Component (Decision) | Canonical Composite ID String (Optional) | Reliability |
|-------------|----------------------------------|------------------------------------------|-------------|
| **File** | `FileId` OR `UniqueId` | `id:File:{fileid}\|{uniqueid}\|{siteid}\|{webid}` (include any available) | Very High |
| **Event** | `EventId.Id` (occurrence-level) | `id:Event:{eventid.id}\|{originalid}` (include `OriginalId` when available) | Very High |
| **Email** | `EmailMessage.Url` | `id:Email:{emailmessage.url}` | High |
| **TeamsMessage** | `ItemId.Id` (message-level) | `id:TeamsMessage:{itemid.id}\|{url}` (include `Url` when available) | High |
| **People** | `Alias` OR `UPN` OR any `EmailAddresses[].Address` | `id:People:{upn}` OR `id:People:{emailaddresses[i].address}` | Very High |
| **External** | `ArtifactId` | `id:External:{artifactid}` | High |
| **MeetingTranscript** | Associated `EventId.Id` (occurrence-level) | `id:MeetingTranscript:{eventid.id}` | Very High |

**Decision semantics (Section 4.4.1 fast path):**

For “composite ID agrees/disagrees”, interpret it as:
- Compare **all comparable ID components** that are present on *both* results.
- Missing components are treated as **unknown** (not a mismatch).

| Outcome | Decision |
|---|---|
| Any comparable ID component differs (composite ID disagrees) | **Different** |
| All comparable ID components match AND at least one **definitive item-level ID component** is comparable and matches | **Duplicate** |
| All comparable ID components match BUT no definitive item-level ID component is comparable (only non-definitive IDs present) | **Inconclusive** (proceed to URL/text/field verification) |

> **TeamsMessage URL note:** In some datasets, `ItemId.Id` may be missing while `Url` is present. In that case, `Url` is verified via Section 4.2 (URL Verification), not as a definitive ID component.

> **📝 File ID Semantics:**
> - **FileId** and **UniqueId** are *definitive item-level* identifiers for a file.
> - If both are present on both results, require both to match; any disagreement implies **Different**.
>
> **📝 Event ID Semantics (recurrence-safe):**
> - **EventId.Id** is an *occurrence-level* identifier and is sufficient to uniquely identify the same event occurrence.
> - **OriginalId** is a *series-level* identifier and is not sufficient on its own; it must be paired with time-based checks (see Section 4.4.3.2).
>
> **📝 TeamsMessage ID Semantics:**
> - **ItemId.Id** is a *message-level* identifier and is sufficient to uniquely identify the same Teams message.
>
> **📝 Container IDs (`SiteId`/`WebId`) Consistency Rule (all entity types when available):**
> - **SiteId** and **WebId** are *container-level* identifiers (site collection / web).
> - They are **not sufficient** to conclude a duplicate on their own.
> - When present on both results, they participate as non-definitive components in the composite-ID agreement check: disagreement => **Different**; agreement alone => **Inconclusive**.

See Section 2.2.5.2 ("Component ID Examples") for example formats/values.

> **⚠️ Source Object Availability:** When `Source` is missing, many ID components may be absent and canonical composite ID strings cannot be formed. See Section 2.3 for SEVAL job #144683 observations.

### **4.2 URL Verification (Minimal Normalization)**

For Phase 2, use minimal URL normalization — strip only known content-unrelated parameters, keep all others:

```
{netloc}{path}?{filtered_query}
```

**Known content-unrelated parameters to strip:**
- `web`, `preview`, `view`, `action` (UI/display params)
- `auth`, `token`, `session` (authentication params)
- `utm_*` (tracking params)

**Known uniqueness-affecting parameters to keep (examples):**
- `id`, `itemid`, `version` (resource selectors / version selectors)

If a parameter is known to affect document uniqueness, we keep it in the Phase 2 normalized URL.
If it is present in both URLs and differs, we reject the pair as **Different**.

**Unknown parameters (hard case):**
- Preserve them by default (conservative).
- Only add them to the strip list after we have high confidence they are content-unrelated.

**Decision semantics:**
- If the Phase 2 normalized URLs match exactly -> **Duplicate**.
- If the URLs differ in domain or path -> **Different**.
- If the URLs match on domain + path but differ only in parameters -> **Inconclusive by URL alone**; proceed to Step 2 (Section 4.4.2) where `Snippet`/`Content` may confirm, otherwise continue to Step 3.

| Original URL | Phase 2 URL |
|--------------|-------------|
| `https://sharepoint.com/doc.docx?version=1&web=1&x-unknown=abc` | `sharepoint.com/doc.docx?version=1&x-unknown=abc` |
| `http://sharepoint.com/doc.docx?id=123&preview=true&x-unknown=def` | `sharepoint.com/doc.docx?id=123&x-unknown=def` |

**Design Rationale:** 
- Unknown parameters are preserved in comparison (conservative approach) for accurate verification
- All original field values are preserved in the data for traceability

### **4.3 Text Normalization and Comparison (Snippet/Content)**

When this document says `Snippet`/`Content` **matches** in Phase 2, it means:
- Normalize both strings (lowercase + whitespace collapse).
- Compare the **full available normalized text** for exact equality.

Rationale: Phase 2 runs on a relatively small set of candidate pairs, so full-string equality is typically cheap and avoids false positives that prefix-only comparisons can introduce.

**If text is truncated or capped (implementation detail):**
- Upstream systems may return only a truncated snippet/content, or an implementation may cap comparison to a maximum length for safety.
- In those cases, treat a prefix-only equality as **weaker evidence** and require at least one additional corroborating field match (e.g., `From` for TeamsMessage, `PluginId` for External, `FileName` for File) before concluding **Duplicate**.

### **4.4 Verification Logic**

#### **4.4.1. Step 1: ID Component Verification**

Form a canonical view of the available ID evidence by considering **all available ID components** for the entity (as listed in Section 4.1).

**Available Fields (in this step):** All ID components present for the entity type (Section 4.1), including container IDs like `SiteId`/`WebId` when available.

**Fields used for dedup decisions (in this step):** All ID components that are present on **both** results (definitive item-level ID components can decide **Duplicate**; any comparable mismatch decides **Different**).

**How comparison works:**
- Compare **all ID components that are present on both results**.
- Missing components are treated as **unknown** (not a mismatch).

**Decision semantics:**

| Outcome | Decision |
|---|---|
| Any comparable ID component differs (composite ID disagrees) | **Different** |
| All comparable ID components match AND the shared components include at least one **definitive item-level ID component** (Section 4.1) | **Duplicate** |
| All comparable ID components match BUT the shared components include no definitive item-level ID component (only non-definitive IDs) | **Inconclusive** -> Step 2 |
| No ID components are comparable (nothing present on both) | -> Step 2 |

---

#### **4.4.2. Step 2: URL Verification**

If a URL is available, compare Phase 2 minimally normalized URLs (Section 4.2).

**Available Fields (in this step):** `Url` (when present).

**Fields used for dedup decisions (in this step):** Phase 2 minimally normalized `Url`.

If the URLs match on domain + path but differ in query parameters, treat the URL as **inconclusive** unless the differing keys include a known uniqueness-affecting parameter (e.g., `id`, `itemid`, `version`), in which case conclude **Different**.

| Scenario | Decision |
|----------|----------|
| Same domain + path (no params or same params) | **Duplicate** |
| Different domain or path | **Different** |
| Same domain + path, params differ -> verify `Snippet`/`Content` (Section 4.3) | Match -> **Duplicate**; Mismatch -> **Different**; Missing -> Step 3 |
| No URL available | -> Step 3 |

---

#### **4.4.3. Step 3: Domain-Specific Verification**

When URL is unavailable or inconclusive, apply domain-specific verification. Each entity type has its own logic combining ID-based and field/content-based verification.

This step assumes Step 1 (ID components) did not decide (i.e., definitive IDs were missing on at least one side) and Step 2 (URL) was unavailable or inconclusive.

This step should not repeat Step 1/2 checks (definitive ID comparison, container-ID conflicts). If those checks were comparable, the pair would have already been decided earlier.

**Immediate Rejection (applies to all entity types):**
- Type mismatch (e.g., File vs Email) -> **Different**

##### **4.4.3.1. File**

**Available Fields (in this step):** FileName, Title, FileType, Author, LastModifiedBy, LastModifiedTime, Snippet/Content

**Fields used for dedup decisions (in this step):** FileName, Title, FileType, Author, LastModifiedBy, LastModifiedTime, Snippet/Content

This section follows the same pattern as canonical composite IDs: treat the available File fields as a “composite” of evidence.

**Step A - Mismatch check (reject on any comparable mismatch):**

Compare all fields that are present on **both** results. If any comparable field differs -> **Different**.

| Field (only if present in both) | If differs -> |
|---|---|
| `FileName` | **Different** (we do not allow renamed-file matches in this design) |
| `FileType` | **Different** |
| `Author` | **Different** |
| `LastModifiedBy` | **Different** |
| `LastModifiedTime` | **Different** |
| `Title` | **Different** |
| `Snippet` | **Different** |
| `Content` | **Different** |

**Step B - Decisive match:**

If `Snippet` **matches** OR `Content` **matches** (as defined in Section 4.3) and Step A found no mismatches -> **Duplicate**.

**Step C - No decisive match (Snippet/Content missing on one or both):**

If neither `Snippet` nor `Content` is comparable-and-matching, we can still conclude **Duplicate** only if we have enough agreeing metadata:

- Require `FileName` to be present on both results and match.
- In addition, require at least **two** matching fields among: `Title`, `Author`, `FileType`, `LastModifiedBy`, `LastModifiedTime`.

If these requirements are not met -> **Inconclusive** (treat as **Different** for dedup; log for analysis).

##### **4.4.3.2. Event**

**Available Fields (in this step):** OriginalId, Start, End, Subject/Title, OrganizerEmail, OrganizerName, Invitees, Snippet/Content

**Fields used for dedup decisions (in this step):** OriginalId, Start, End, Subject/Title, OrganizerEmail, Snippet/Content

**Fields explicitly not used for dedup decisions:** `Invitees`, `OrganizerName` (can vary and are not required for identity).

This section follows the same pattern as canonical composite IDs: treat the available Event fields as a “composite” of evidence.

> **Event recurring-series safety:**
> - `Snippet`/`Content`, `Title`, and `OrganizerEmail` are **not reliable** to *confirm* duplicate across recurring occurrences *without time*.
> - **Only occurrence time (`Start`/`End`) can confirm the same occurrence** when definitive event IDs are missing.
> - **Do not use `Invitees` for verification or rejection** (it can legitimately change across iterations).

**Step A - Mismatch check (reject on stable comparable mismatches):**

Events have several fields that can legitimately vary across iterations (e.g., snippet extraction, title formatting). For this reason, Step A only rejects on mismatches for a small set of **stable** fields.

Compare the stable fields that are present on **both** results. If any comparable stable field differs -> **Different**.

| Field (only if present in both) | If differs -> |
|---|---|
| `OriginalId` | **Different** |
| `Start` | **Different** |
| `End` | **Different** |
| `OrganizerEmail` | **Different** |

**Other comparable fields (do not reject on mismatch):**

If these fields differ, treat it as **non-decisive** (do not return **Different** in Step A). They may be used as corroboration signals in Step B.

`OrganizerName` is tracked as an available field for completeness, but is not used for verification or rejection.

| Field (only if present in both) | If differs -> |
|---|---|
| `Title` / `Subject` | Ignore for rejection |
| `Snippet` | Ignore for rejection |
| `Content` | Ignore for rejection |

**Step B - Decisive match (same occurrence):**

This step intentionally focuses on **Start + End time** as the occurrence identity.

Conclude **Duplicate** only if Step A found no mismatches AND at least one of the following evidence lines matches:

1. `Start` matches AND `End` matches AND `OriginalId` matches
2. `Start` matches AND `End` matches AND `OrganizerEmail` matches AND (`Title`/`Subject` matches OR `Snippet` matches OR `Content` matches)
3. `Start` matches AND `End` matches AND `Title`/`Subject` matches AND (`Snippet` matches OR `Content` matches)

`Title`/`Subject` and `Snippet`/`Content` are treated as corroboration only (they can be unstable and/or shared across a series), and are only allowed to participate in a **Duplicate** decision when `Start` + `End` already match.

**Step C - No decisive match (time missing or not comparable):**

If the decisive match conditions above cannot be satisfied (e.g., `Start`/`End` missing on one side, or no stable corroboration is comparable) -> **Inconclusive** (treat as **Different** for dedup; log for analysis).

##### **4.4.3.3. Email**

**Available Fields (in this step):** DateTimeReceived, From, To, Subject, Snippet/Content, IsRead, IsMentioned

**Fields used for dedup decisions (in this step):** DateTimeReceived, From, To, Subject, Snippet/Content

**Fields explicitly not used for dedup decisions:** `IsRead`, `IsMentioned` (these can change across iterations and are not identity signals).

This section follows the same pattern as canonical composite IDs: treat the available Email fields as a “composite” of evidence.

**Step A - Mismatch check (reject on any comparable mismatch):**

Compare all fields that are present on **both** results. If any comparable field differs -> **Different**.

| Field (only if present in both) | If differs -> |
|---|---|
| `DateTimeReceived` | **Different** |
| `From` | **Different** |
| `To` | **Different** |
| `Subject` | **Different** |
| `Snippet` | **Different** |
| `Content` | **Different** |

**Step B - Decisive match:**

If Step A found no mismatches, conclude **Duplicate** if any of the following evidence patterns match:

1. `DateTimeReceived` matches AND at least one of the following matches: `Subject`, (`Snippet` matches OR `Content` matches), `From`, `To`
2. `Subject` matches AND (`Snippet` matches OR `Content` matches) AND (`From` matches OR `To` matches)
3. (`Subject` matches OR `Snippet` matches OR `Content` matches) AND `From` matches AND `To` matches

**Step C - Otherwise:**

If none of the Step B duplicate conditions can be satisfied (e.g., required fields are missing on one side or comparable fields do not match) -> **Inconclusive** (treat as **Different** for dedup; log for analysis).

##### **4.4.3.4. TeamsMessage**

**Available Fields (in this step):** From, To, Participants, Subject/Title, Snippet/Content

**Fields used for dedup decisions (in this step):** From, To, Participants, Subject/Title, Snippet/Content

This section follows the same pattern as canonical composite IDs: treat the available Teams message fields as a “composite” of evidence.

**Step A - Mismatch check (reject on any comparable mismatch):**

Compare all fields that are present on **both** results. If any comparable field differs -> **Different**.

| Field (only if present in both) | If differs -> |
|---|---|
| `From` | **Different** |
| `To` | **Different** |
| `Participants` | **Different** |
| `Title` | **Different** |
| `Subject` | **Different** |
| `Snippet` | **Different** |
| `Content` | **Different** |

**Step B - Decisive match:**

If Step A found no mismatches, conclude **Duplicate** if any of the following evidence patterns match:

1. (`Snippet` matches OR `Content` matches) AND (`Subject` matches OR `Title` matches) AND at least one of the following matches: `From`, `To`, `Participants`
2. (`Snippet` matches OR `Content` matches OR `Subject` matches OR `Title` matches) AND at least two of the following match (when comparable): `From`, `To`, `Participants`

**Step C - Otherwise:**

If none of the Step B duplicate conditions can be satisfied (e.g., required fields are missing on one side or comparable fields do not match) -> **Inconclusive** (treat as **Different** for dedup; log for analysis).

##### **4.4.3.5. People**

**Available Fields (in this step):** DisplayName, Department, CompanyName, OfficeLocation

**Fields used for dedup decisions (in this step):** DisplayName, Department, CompanyName, OfficeLocation

This section follows the same pattern as canonical composite IDs: treat the available People fields as a “composite” of evidence.

**Step A - Mismatch check (reject on any comparable mismatch):**

Compare all fields that are present on **both** results. If any comparable field differs -> **Different**.

| Field (only if present in both) | If differs -> |
|---|---|
| `DisplayName` | **Different** |
| `Department` | **Different** |
| `CompanyName` | **Different** |
| `OfficeLocation` | **Different** |

**Step B - Decisive match:**

People results typically have no decisive content fields. In this step, treat People as **Duplicate** only when we have sufficiently specific agreeing metadata.

If Step A found no mismatches, conclude **Duplicate** only if:
- `DisplayName` is present on both results and matches.
- In addition, at least **two** of the following match (when comparable): `Department`, `CompanyName`, `OfficeLocation`.

**Step C - Otherwise:**

If none of the Step B duplicate conditions can be satisfied (e.g., required fields are missing on one side or comparable fields do not match) -> **Inconclusive** (treat as **Different** for dedup; log for analysis).

##### **4.4.3.6. External**

**Available Fields (in this step):** DisplayName, PluginId, PluginType, Snippet/Content

**Fields used for dedup decisions (in this step):** DisplayName, PluginId, PluginType, Snippet/Content

This section follows the same pattern as canonical composite IDs: treat the available External fields as a “composite” of evidence.

**Step A - Mismatch check (reject on any comparable mismatch):**

Compare all fields that are present on **both** results. If any comparable field differs -> **Different**.

| Field (only if present in both) | If differs -> |
|---|---|
| `PluginId` | **Different** |
| `PluginType` | **Different** |
| `DisplayName` | **Different** |
| `Snippet` | **Different** |
| `Content` | **Different** |

**Step B - Decisive match:**

If Step A found no mismatches, conclude **Duplicate** if any of the following evidence patterns match:

- `Snippet` matches OR `Content` matches (Section 4.3)

**Step C - Otherwise:**

If none of the Step B duplicate conditions can be satisfied (e.g., required fields are missing on one side or comparable fields do not match) -> **Inconclusive** (treat as **Different** for dedup; log for analysis).

##### **4.4.3.7. MeetingTranscript**

**Available Fields (in this step):** Start, End, Title, Snippet/Content

**Fields used for dedup decisions (in this step):** Start, End, Title, Snippet/Content

This section follows the same pattern as canonical composite IDs: treat the available transcript fields as a “composite” of evidence.

**Step A - Mismatch check (reject on any comparable mismatch):**

Compare all fields that are present on **both** results. If any comparable field differs -> **Different**.

| Field (only if present in both) | If differs -> |
|---|---|
| `Start` | **Different** |
| `End` | **Different** |
| `Title` | **Different** |
| `Snippet` | **Different** |
| `Content` | **Different** |

**Step B - Decisive match:**

If Step A found no mismatches, conclude **Duplicate** if any of the following evidence patterns match:

- `Snippet` matches OR `Content` matches (Section 4.3)
- `Title` matches AND (`Start` matches OR `End` matches)

**Step C - Otherwise:**

If none of the Step B duplicate conditions can be satisfied (e.g., required fields are missing on one side or comparable fields do not match) -> **Inconclusive** (treat as **Different** for dedup; log for analysis).

---

## **5. Implementation Considerations**

### **5.1 Hash Function (Phase 1 Only)**

Use hashes for **Phase 1 candidate generation** (fast lookup):

```python
import hashlib
import re
from urllib.parse import urlparse

# Recommended Phase 1 defaults (tunable):
SNIPPET_PREFIX_LENGTH = 100
CONTENT_PREFIX_LENGTH = 500
TITLE_PREFIX_LENGTH = 100

MIN_TEXT_LENGTH_FOR_HASH_KEY = 50

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text_for_hash(text: str) -> str:
    """Normalize text for Phase 1 hashing (casefold + whitespace collapse)."""
    return _WHITESPACE_RE.sub(" ", text.casefold()).strip()


def hash_text_prefix(text: str, prefix_length: int) -> str:
    """Generate a short stable hash of a normalized text prefix (Phase 1 only)."""
    normalized_prefix = normalize_text_for_hash(text)[:prefix_length]
    return hashlib.sha256(normalized_prefix.encode("utf-8")).hexdigest()[:16]


def normalize_url_aggressive(url: str) -> str:
    """Aggressively normalize a URL for Phase 1 candidate generation.

    Phase 1 prioritizes recall: we drop scheme, query, and fragment.
    Phase 2 uses minimal normalization (see Section 4.2) and is authoritative.
    """
    parsed = urlparse(url)
    netloc = (parsed.netloc or "").casefold()
    path = parsed.path or ""
    return f"{netloc}|{path}"
```

> **Note:** Phase 2 verification uses actual field/text comparison (see Sections 4.3 and 4.4), not hash comparison. Hashes are only for fast candidate lookup in Phase 1.

### **5.2 Key Generation**

```python
def get_all_dedup_keys(entity: dict, verification_data: dict) -> list[str]:
    """Return Phase 1 dedup keys.

    Notes:
    - Keys are namespaced by entity type to avoid cross-type collisions.
    - Do NOT use `reference_id` (it is not stable across iterations/queries).
    - Any single matching key can be used to propose a candidate pair; Phase 2 decides.
    """

    entity_type = (entity.get("EntityType") or entity.get("type") or "unknown").casefold()
    keys: set[str] = set()

    def add_key(key: str) -> None:
        keys.add(f"{entity_type}|{key}")

    # Individual ID keys (all available).
    # Treat every extracted component as an independent key to maximize recall.
    extracted_ids = entity.get("ExtractedIds") or {}
    for id_type, id_value in extracted_ids.items():
        if not id_value:
            continue

        if isinstance(id_value, (list, tuple, set)):
            values = [v for v in id_value if v]
        else:
            values = [id_value]

        for value in values:
            add_key(f"id:{str(id_type).casefold()}:{str(value).casefold()}")

    # Aggressively normalized URL key (Phase 1 only).
    url = verification_data.get("url")
    if url:
        add_key(f"url:{normalize_url_aggressive(url)}")

    # Text-based keys (Phase 1 only).
    snippet = verification_data.get("snippet")
    if isinstance(snippet, str) and len(snippet) >= MIN_TEXT_LENGTH_FOR_HASH_KEY:
        add_key(f"snippet:{hash_text_prefix(snippet, prefix_length=SNIPPET_PREFIX_LENGTH)}")

    content = verification_data.get("content_body") or verification_data.get("content")
    if isinstance(content, str) and len(content) >= MIN_TEXT_LENGTH_FOR_HASH_KEY:
        add_key(f"content:{hash_text_prefix(content, prefix_length=CONTENT_PREFIX_LENGTH)}")

    title = verification_data.get("title")
    if isinstance(title, str) and len(title) >= MIN_TEXT_LENGTH_FOR_HASH_KEY:
        add_key(f"title:{hash_text_prefix(title, prefix_length=TITLE_PREFIX_LENGTH)}")

    return sorted(keys)
```

## **6. Known Limitations**

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **ID Field Availability** | Not all entities have all ID fields | Multi-ID extraction + content-based keys |
| **URL Availability** | Not all types have URLs | Type-specific field matching |
| **Content Variability** | Minor content changes break hash | Use prefix (first N chars) |
| **Empty Content** | People entities have no content | Rely on unique IDs (Alias/UPN) |
| **Cross-Type Collisions** | Title "Meeting" could match multiple types | Phase 2 verification rejects cross-type matches |
