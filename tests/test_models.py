import pytest
from pydantic import ValidationError

from filinglens.models import DocumentPage


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
