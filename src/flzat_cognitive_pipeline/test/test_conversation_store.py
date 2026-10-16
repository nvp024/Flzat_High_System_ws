import pytest

from flzat_cognitive_pipeline.conversation_store import ConversationStore


def test_conversation_store_is_bounded_and_returns_recent_turns():
    store = ConversationStore(max_turns=2, max_text_characters=64)
    store.append("social_dialogue", "one", "reply one")
    store.append("social_dialogue", "two", "reply two")
    store.append("social_dialogue", "three", "reply three")

    assert len(store) == 2
    assert [turn["user"] for turn in store.recent()] == ["two", "three"]
    assert store.recent(1)[0]["assistant"] == "reply three"


def test_conversation_store_rejects_empty_turns():
    store = ConversationStore()
    with pytest.raises(ValueError):
        store.append("social_dialogue", "", "reply")

