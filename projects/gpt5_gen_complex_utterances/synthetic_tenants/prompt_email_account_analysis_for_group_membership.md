# Group Membership Analysis

**DO NOT16. **DO NOT assume technical skills overlap:** A Software Engineer is NOT a Database Administrator, Network Engineer, or DevOps Engineer.
17. **When a user should be a member, include ALL their email aliases** from the `email_aliases` list in your response.
18. **Include only email addresses from the `email_aliases` fields**; do not list group names, bot accounts, or email addresses not present in the profiles.sume technical skills overlap:** A Software Engineer is NOT a Database Administrator, Network Engineer, or DevOps Engineer.
14. **When a user should be a member, include ALL their email aliases** from the `email_aliases` list in your response.
15. **Include only email addresses from the `email_aliases` fields**; do not list group names, bot accounts, or email addresses not present in the profiles.pt

You are an expert at analyzing organizational structures and determining group membership.

## Group Information
- **Group Email:** {group_email}
- **Group Purpose:** {group_meaning}

## Task
Based on the group email address and purpose, analyze the following individual user profiles and determine which **INDIVIDUAL USERS** (not other groups or bots) would logically be members of this group.

## Individual User Profiles
```json
{user_profiles}
```

## STRICT Membership Criteria
1. **Each user profile contains `email_aliases` (a list of all email addresses for that person) and complete job information.**
2. **BE EXTREMELY SELECTIVE:** Only include users whose job titles, departments, or roles DIRECTLY and SPECIFICALLY match the group's purpose.
3. **Database teams (db.admin, dbadmin.team):** ONLY include Database Administrators, Database Engineers, Data Engineers, or roles with "database" or "data" in the title.
4. **Network teams (network-ops):** ONLY include Network Engineers, Network Administrators, or roles with "network" in the title.
5. **DevOps teams (devops.team, devops-team):** ONLY include DevOps Engineers, Site Reliability Engineers, Platform Engineers, or roles with "DevOps" or "SRE" in the title.
6. **DevOps LEADERSHIP teams (devops.leads, devops.lead):** ONLY include DevOps Leads, DevOps Managers, Platform Engineering Managers, or leadership roles specifically managing DevOps teams.
7. **Embedded Systems LEADERSHIP teams (emb-sys-leads):** ONLY include Technical Leads, Team Leads, or Architects who specifically work on embedded systems, firmware, or hardware. DO NOT include Directors or VPs.
8. **ML Operations teams (ml-ops-team, mlops):** ONLY include Machine Learning Engineers, MLOps Engineers, Data Scientists, AI/ML Engineers, or roles specifically focused on machine learning operations, model deployment, and ML infrastructure. DO NOT include general DevOps engineers.
9. **Platform Operations teams (platform.ops, platform-ops):** ONLY include Platform Engineers, Platform Architects, Infrastructure Engineers, Site Reliability Engineers focused on platform engineering, or roles specifically managing platform infrastructure and services. DO NOT include general DevOps engineers or software developers.
10. **Support teams (*.support, quantum.support):** ONLY include Support Engineers, Customer Support, Technical Support, or roles specifically focused on user assistance and support.
11. **Project Management teams (pm.team):** ONLY include Project Managers, Program Managers, or roles with "Project" or "Program" management in the title.
12. **Security teams:** ONLY include Security Engineers, Security Analysts, or roles with "security" in the title.
12. **Security teams:** ONLY include Security Engineers, Security Analysts, or roles with "security" in the title.
13. **CRITICAL DISTINCTION - Directors vs Technical Leads:** 
    - Directors (Director, VP, Vice President): Executive leadership roles - ONLY include in director-specific groups
    - Technical Leads (Technical Lead, Team Lead, Lead Engineer): Individual contributor leadership - include in technical lead groups
    - DO NOT put Directors/VPs in technical lead groups
14. **Leadership teams (*leads, *.leads):** ONLY include Technical Leads, Team Leads, Lead Engineers, or similar individual contributor leadership roles. DO NOT include Directors, VPs, or Managers.
15. **Executive teams (*director, *.director):** ONLY include Directors, VPs, Vice Presidents, or executive management roles.
16. **DO NOT include individual contributors in leadership teams:** Engineers, Analysts, and individual contributors should NOT be in leadership groups even if they work in the same domain.
17. **DO NOT include software engineers, developers, or technical leads unless they have specific domain expertise in the group's focus area.**
18. **DO NOT assume technical skills overlap:** A Software Engineer is NOT a Database Administrator, Network Engineer, or DevOps Engineer.
19. **Specialized Operations Teams:** MLOps teams are NOT the same as DevOps teams. Platform Operations teams are NOT the same as general DevOps teams. Each requires specific domain expertise.
20. **Empty Groups are Valid:** If no users have the specific expertise required for a specialized team (e.g., no ML specialists for ml-ops-team), the group should be empty.
21. **When a user should be a member, include ALL their email aliases** from the `email_aliases` list in your response.
22. **Include only email addresses from the `email_aliases` fields**; do not list group names, bot accounts, or email addresses not present in the profiles.

## Response Format
**CRITICAL: Your response must be ONLY a valid JSON array. Do not include any explanations, reasoning, markdown formatting, or additional text.**

**REQUIRED OUTPUT FORMAT:**
- Start your response immediately with `[`
- End your response with `]`
- Include email addresses from the `email_aliases` fields of users who should be members
- If a user should be a member, include ALL their email aliases from their profile
- Do not add any text before or after the JSON array
- Do not use markdown code blocks (no ```json or ```)
- Do not provide explanations or rationale

**Valid example response (include all aliases for selected members):**
```
["user1@company.com", "user1.alternate@company.com", "user2@company.com", "user3@company.com", "user3.alt@company.com"]
```

**Another valid example (empty group):**
```
[]
```

**INVALID examples (DO NOT DO THIS):**
- Adding explanations: "Here are the members: ["user1"] because..."
- Using markdown: ```json ["user1"] ```
- Adding text: "The following users should be members: ["user1"]"

**Remember: Output ONLY the JSON array, nothing else.**
