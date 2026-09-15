from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class Chunk(BaseModel):
    """A contiguous text segment used as a retrieval unit."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    filing_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    section: str | None = None

    @field_validator("chunk_id", "filing_id", "text")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self
