import pytest
from pydantic import ValidationError

from filinglens.models import Chunk, DocumentPage


def test_document_page_accepts_valid_data() -> None:
    page = DocumentPage(
        filing_id="nvda-2025-10k",
        page_number=1,
        text="  Revenue increased year over year.  ",
    )

    assert page.model_dump() == {
        "filing_id": "nvda-2025-10k",
        "page_number": 1,
        "text": "Revenue increased year over year.",
    }


def test_document_page_rejects_zero_page_number() -> None:
    with pytest.raises(ValidationError):
        DocumentPage(
            filing_id="nvda-2025-10k",
            page_number=0,
            text="Revenue increased.",
        )


def test_document_page_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        DocumentPage(
            filing_id="nvda-2025-10k",
            page_number=1,
            text="   ",
        )


def test_chunk_accepts_valid_data() -> None:
    chunk = Chunk(
        chunk_id="nvda-2025-10k-chunk-0000",
        filing_id="nvda-2025-10k",
        chunk_index=0,
        page_start=12,
        page_end=13,
        text="  Research and development expenses increased.  ",
        section="Item 7. Management's Discussion and Analysis",
    )

    assert chunk.text == "Research and development expenses increased."
    assert chunk.page_start == 12
    assert chunk.page_end == 13


def test_chunk_rejects_reversed_page_range() -> None:
    with pytest.raises(ValidationError, match="page_end"):
        Chunk(
            chunk_id="nvda-2025-10k-chunk-0000",
            filing_id="nvda-2025-10k",
            chunk_index=0,
            page_start=13,
            page_end=12,
            text="Research and development expenses increased.",
        )


def test_chunk_rejects_negative_index() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="nvda-2025-10k-chunk-0000",
            filing_id="nvda-2025-10k",
            chunk_index=-1,
            page_start=12,
            page_end=12,
            text="Research and development expenses increased.",
        )
