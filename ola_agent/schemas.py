"""Pydantic schemas: crew response format (Task 9) and shared models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Intent = Literal["policy", "ticket_status", "mixed", "out_of_scope", "blocked"]


class SupportResponse(BaseModel):
    """Every crew response must conform to this schema (validated in code after every kickoff)."""

    answer: str = Field(min_length=1, description="Final answer shown to the user.")
    intent: Intent
    sources: list[str] = Field(default_factory=list, description="KB doc_ids the answer is grounded in.")
    record_id: str | None = None
    ticket_status: str | None = None
    escalation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    recommend_escalation: bool | None = None
    grounded: bool = Field(description="True when every claim comes from retrieved context or the ticket record.")
    refused: bool = False
    guardrail_flags: list[str] = Field(default_factory=list)

    @field_validator("record_id")
    @classmethod
    def _record_id_format(cls, v: str | None) -> str | None:
        if v is not None and not (v.startswith("OLA-T-") and v[6:].isdigit()):
            raise ValueError("record_id must look like OLA-T-0001")
        return v
