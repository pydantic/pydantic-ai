import logfire

### [import-modal]
import modal  ### [/import-modal]

from .models import Analysis


### [analysis_store]
class AnalysisStore:
    @classmethod
    @logfire.instrument('Add analysis to store')
    async def add(cls, analysis: Analysis):
        await cls._get_store().put.aio(analysis.profile.email, analysis.model_dump())

    @classmethod
    @logfire.instrument('List analyses from store')
    async def list(cls) -> list[Analysis]:
        return [
            Analysis.model_validate(analysis)
            async for analysis in cls._get_store().values.aio()
        ]

    @classmethod
    @logfire.instrument('Clear analyses from store')
    async def clear(cls):
        await cls._get_store().clear.aio()

    @classmethod
    def _get_store(cls) -> modal.Dict:
        return modal.Dict.from_name('analyses', create_if_missing=True)  # type: ignore ### [/analysis_store]
