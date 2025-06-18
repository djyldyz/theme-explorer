"""Module for defining response models for synthetic user feedback"""

from typing import List

from pydantic import BaseModel, Field


class FeedbackItem(BaseModel):
    feedback: str = Field(
        description="Respondent's feedback on the web-site or mobile app."
    )
    sentiment: str = Field(description="Sentiment of the feedback.")


class SyntheticResponse(BaseModel):
    """Model for multiple synthetic feedback responses"""

    responses: List[FeedbackItem]
