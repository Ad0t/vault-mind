# ADR 1: Corpus Selection for VaultMind

## Context

VaultMind is a Retrieval-Augmented Generation (RAG) system designed to answer questions from a domain-specific corpus while providing grounded citations.

For the initial version (v1), the corpus should:

* Belong to a single domain.
* Be small enough to process and iterate quickly.
* Contain structured information such as chapters, headings, and page numbers.
* Support citation-based answers.

Several options were considered:

1. Multiple research papers from different domains.
2. Operating Systems course notes.
3. Database Management Systems (DBMS) textbook/notes.

## Decision

Use a single Database Management Systems (DBMS) textbook/course notes (approximately 150–300 pages) as the corpus for VaultMind v1.

## Rationale

* DBMS represents a well-defined academic domain.
* The corpus size is manageable for rapid experimentation and debugging.
* Textbooks contain rich hierarchical structure (chapters, sections, page numbers), enabling accurate retrieval and citations.
* Domain-specific questions make evaluation straightforward.
* Existing familiarity with DBMS concepts reduces the learning overhead during development.

## Consequences

### Positive

* Faster development and experimentation.
* Easier chunking due to clear document structure.
* Reliable citation generation using page numbers and section headings.
* Simplifies evaluation because ground truth answers can be verified manually.

### Negative

* The system will initially be limited to DBMS-related queries.
* Generalization to other domains will require future extensions.

## Future Work

Future versions may expand the corpus to include Operating Systems notes or collections of research papers to evaluate cross-document retrieval capabilities.
