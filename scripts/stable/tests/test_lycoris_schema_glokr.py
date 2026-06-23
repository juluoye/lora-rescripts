from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_shared_lycoris_schema_exposes_glokr_for_sdxl():
    shared_schema = (ROOT / "mikazuki" / "schema" / "shared.ts").read_text(encoding="utf-8")
    sdxl_schema = (ROOT / "mikazuki" / "schema" / "sdxl-lora.ts").read_text(encoding="utf-8")

    assert 'lycoris_algo: Schema.union(["locon", "loha", "lokr", "glokr"' in shared_schema
    assert "lycoris_algo: Schema.union(['lokr', 'glokr']).required()" in shared_schema
    assert "LoKr / GLoKr 分解因子" in shared_schema
    assert "SHARED_SCHEMAS.LYCORIS_MAIN" in sdxl_schema
    assert "SHARED_SCHEMAS.LYCORIS_LOKR" in sdxl_schema
