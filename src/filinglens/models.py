from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentPage(BaseModel):
    """One non-empty, human-readable page extracted from a filing."""

    model_config = ConfigDict(extra="forbid")

    filing_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)

    @field_validator("filing_id", "text")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value
