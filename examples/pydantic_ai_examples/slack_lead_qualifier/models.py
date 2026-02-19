from typing import Annotated, Any

from annotated_types import Ge, Le
from pydantic import BaseModel

### [import-format_as_xml]
from pydantic_ai import format_as_xml  ### [/import-format_as_xml]


### [profile,profile-intro]
class Profile(BaseModel):  ### [/profile-intro]
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    email: str  ### [/profile]

    ### [profile-as_prompt]
    def as_prompt(self) -> str:
        return format_as_xml(self, root_tag='profile')  ### [/profile-as_prompt]


### [analysis,analysis-intro]
class Analysis(BaseModel):  ### [/analysis-intro]
    profile: Profile
    organization_name: str
    organization_domain: str
    job_title: str
    relevance: Annotated[int, Ge(1), Le(5)]
    """Estimated fit for Pydantic Logfire: 1 = low, 5 = high"""
    summary: str
    """One-sentence welcome note summarising who they are and how we might help"""  ### [/analysis]

    ### [analysis-as_slack_blocks]
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
        ]  ### [/analysis-as_slack_blocks]
