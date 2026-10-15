from flzat_cognitive_pipeline.transcript import is_usable_transcript


def test_transcript_requires_meaningful_content():
    assert is_usable_transcript("get the book")
    assert is_usable_transcript("2")
    assert not is_usable_transcript("   ")
    assert not is_usable_transcript("...?!")
