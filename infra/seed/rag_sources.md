# RAG source documents (§7.9)

A curated set of a few dozen documents the RAG ingestion job
(`python/jobs/ingest_docs.py`, not yet built) chunks and embeds into pgvector.
List them here with a source URL before downloading — do not commit the PDFs
themselves without checking each publisher's redistribution terms (`.gitignore`
already excludes `data/rag_sources/*.pdf`).

## NIST — FIPS standards (public domain, US government work)

- [ ] FIPS 203 — Module-Lattice-Based Key-Encapsulation Mechanism Standard (ML-KEM) — https://csrc.nist.gov/pubs/fips/203/final
- [ ] FIPS 204 — Module-Lattice-Based Digital Signature Standard (ML-DSA) — https://csrc.nist.gov/pubs/fips/204/final
- [ ] FIPS 205 — Stateless Hash-Based Digital Signature Standard (SLH-DSA) — https://csrc.nist.gov/pubs/fips/205/final
- [ ] NIST SP 1800-38 — Migration to Post-Quantum Cryptography (practice guide) — https://csrc.nist.gov/pubs/sp/1800/38/final

## ETSI — check licence before redistributing

- [ ] ETSI TR 103 619 — Migration strategies and recommendations to Quantum Safe schemes
- [ ] ETSI TR 103 823 — Quantum-Safe Public-Key Encryption and Key Encapsulation

## Vendor / community migration guidance (secondary sources)

- [ ] Add 2–3 vendor whitepapers here as you find ones with clear, citable
      migration guidance (e.g. cloud provider PQC migration guides).

---

**Ingestion checklist per document** (implemented by `pqc_rag.ingest`, §7.9):
chunk ~500 tokens with overlap → embed with
`sentence-transformers/all-MiniLM-L6-v2` → store in the `rag` schema of the
shared Postgres instance, tagged with `source_title`, `source_url`, `section`.
