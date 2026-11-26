# Data sources, licences and transformations

Snapshot:2026-11-20. No student submissions or personal study records.

- software: CPython3.11.2 standard-library source, public API docstrings and AST facts. Each source_url points to the versioned file and line. PSF licence: https://docs.python.org/3.11/license.html . Parameter counts describe static declarations, excluding self/cls; class counts are not_applicable. Not runtime signature introspection.
- university: official MIT Learn API, offered_by=ocw, course records only; https://api.learn.mit.edu/api/v1/learning_resources/ . Attribution: MIT OpenCourseWare and the named instructors in each record. Terms: https://ocw.mit.edu/terms/ ; https://creativecommons.org/licenses/by-nc-sa/4.0/ . Derived OCW data and any applicable derived trained models must retain attribution, noncommercial and share-alike obligations. We do not include images or third-party course files.
- research: Crossref REST bibliographic metadata, query machine learning, publications2020–2025. https://www.crossref.org/documentation/retrieve-metadata/ . Factual bibliographic metadata is generally public-domain/CC0; no publisher full texts or abstracts are copied. Original DOI links retained. Count fields describe deposited metadata, not independently verified paper contents.

Changes:500 unique record IDs per option; compact metadata contexts; deterministic withholding of1–2 fields; authored bilingual question templates and derived exact answers; document-identity split350/100/50. Full descriptive text is available for retrieval where supplied. GPT trains on compact contexts, not full source prose. A source URL may contain a withheld field, but the task's answerability is defined by the supplied context only. No claim of500 full research papers, free-form QA or real-world benchmark independence.

Search LSA fits the unlabelled500-document catalogue. GPT vocabulary/weights fit only train QA. Source identity splits prevent the same source's questions crossing partitions; common templates/schema remain shared.
