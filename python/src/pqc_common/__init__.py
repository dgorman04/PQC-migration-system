"""Shared helpers used by more than one Python package.

Per README.md §8 ("separation-of-concerns quick rules"): this package holds
thin, stateless helpers only — no database connections, no business logic.
If a function here starts needing its own config file or its own tests
beyond a couple of lines, it has outgrown pqc_common and belongs in the
package that actually owns that concern.
"""
