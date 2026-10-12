"""
polars reader descriptors for TSV, BED, and GTF files.
"""

from __future__ import annotations

import polars as pl

# Plain tab-separated values with a header.
TSV_FORMAT = {
    "separator": "\t",
}

# Minimal BED: chrom, start, end (0-based, half-open).
BASIC_BED_SCHEMA = {
    "chrom": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
}

BASIC_BED_FORMAT = {
    "separator": "\t",
    "has_header": False,
    "new_columns": list(BASIC_BED_SCHEMA),
    "schema_overrides": BASIC_BED_SCHEMA,
}

# GTF: nine columns; start/end are 1-based inclusive.
GTF_SCHEMA = {
    "chrom": pl.String,
    "source": pl.String,
    "type": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "score": pl.String,
    "strand": pl.String,
    "phase": pl.String,
    "attributes": pl.String,
}

GTF_FORMAT = {
    "separator": "\t",
    "has_header": False,
    "comment_prefix": "#",
    "new_columns": list(GTF_SCHEMA),
    "schema_overrides": GTF_SCHEMA,
}
