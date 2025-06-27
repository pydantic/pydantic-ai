from typing import Annotated, Any

from annotated_types import Ge, Le
from pydantic import BaseModel


### [profile]
class Profile(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    email: str

    def as_prompt(self) -> str:
        from pydantic_ai import format_as_xml

        return format_as_xml(self, root_tag='profile')  ### [/profile]


### [analysis]
class Analysis(BaseModel):
    profile: Profile
    organization_name: str
    organization_domain: str
    job_title: str
    relevance: Annotated[int, Ge(1), Le(5)]
    """Relevance as a sales lead on a scale of 1 to 5"""
    summary: str
    """Short summary of the user and their relevance."""

    def as_slack_blocks(self, include_relevance: bool = False) -> list[dict[str, Any]]:
        profile = self.profile
        relevance = f'({self.relevance}/5)' if include_relevance else ''
        return [
            {
                'type': 'markdown',
                'text': f'[{profile.display_name}](mailto:{profile.email}), {self.job_title} at [**{self.organization_name}**](https://{self.organization_domain}) {relevance}',
            },
            {
                'type': 'markdown',
                'text': self.summary,
            },
        ]  ### [/analysis]


### [unknown]
class Unknown(BaseModel):
    reason: str
    """Reason for why you couldn't find anything."""  ### [/unknown]
