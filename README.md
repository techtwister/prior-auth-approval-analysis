Prior Authorisation Approval Rate Analysis
Tools: SQL · Python (Pandas, NumPy) · Excel
Domain: US Healthcare · Revenue Cycle Management · Insurance Operations

Business Problem
Prior authorisation (PA) denial rates represent one of the most costly inefficiencies in US healthcare operations. High denial rates mean delayed patient care, increased rework costs, and lost revenue. This project set out to answer one question:
What factors most strongly predict whether a prior authorisation request will be approved or denied?

Dataset

2,400+ monthly prior authorisation records
Fields included: submission date, insurer name, clinical specialty, documentation completeness, decision outcome, turnaround time
Source: Operational data from US healthcare RCM workflow


Approach
Step 1 — Data Cleaning (Python / Pandas)

Removed duplicate submissions and incomplete records
Standardised insurer names and specialty categories
Created derived features: submission lag time, document completeness score

Step 2 — Exploratory Analysis (SQL)

Queried approval vs denial rates by insurer, specialty, and submission timing
Identified top denial reasons by payer category
Segmented records by turnaround time buckets

Step 3 — Pattern Identification (Python)

Analysed correlation between submission timing and approval outcome
Compared payer-specific language patterns in approved vs denied requests
Identified document completeness as the single highest predictor of denial


Key Findings
FindingDetailSubmission timingRecords submitted within 24hrs of clinical notes had 34% higher approval ratePayer languageMirroring insurer-specific coverage criteria language increased approvals significantlyIncomplete attachments40% of denials had missing or mismatched supporting documentsBaseline approval rate~72% before workflow redesign

Outcome
Findings were used to redesign the prior authorisation submission workflow:

Introduced a 24-hour submission SLA from clinical note completion
Created payer-specific documentation checklists
Built an attachment verification step before submission

Result: Approval rate improved from ~72% to 95%

Business Impact
This analysis directly informed an operational decision that reduced claim denial rates, decreased rework costs, and improved revenue cycle efficiency across the organisation.

Files in This Repository
FileDescriptionREADME.mdProject overview and findingsprior_auth_analysis.sqlSQL queries for approval pattern analysisprior_auth_analysis.pyPython code for data cleaning and pattern identificationfindings_summary.xlsxSummary of key findings and approval rate trends

Author
Rohit Bhandari
Healthcare Data Analyst | Pharma & RCM Analytics
LinkedIn
