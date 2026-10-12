import io

import polars as pl

from lib.files import BASIC_BED_FORMAT, GTF_FORMAT, GTF_SCHEMA


def test_gtf_format_parses_a_record():
    line = 'chr1\tsrc\tgene\t1\t100\t.\t+\t.\tgene_id "x";\n'
    df = pl.read_csv(io.StringIO(line), **GTF_FORMAT)
    assert df.shape == (1, 9)
    assert list(df.columns) == list(GTF_SCHEMA)
    assert df.schema["start"] == pl.Int64
    assert df.schema["attributes"] == pl.String


def test_basic_bed_format_parses_a_record():
    line = "chr1\t0\t100\n"
    df = pl.read_csv(io.StringIO(line), **BASIC_BED_FORMAT)
    assert df.shape == (1, 3)
    assert list(df.columns) == ["chrom", "start", "end"]
    assert df.schema["start"] == pl.Int64
