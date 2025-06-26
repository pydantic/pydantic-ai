import modal

from .models import Analysis


class AnalysisStore:
    @classmethod
    async def add(cls, analysis: Analysis):
        await cls.get_store().put.aio(analysis.profile.email, analysis.model_dump())

    @classmethod
    async def list(cls) -> list[Analysis]:
        return [
            Analysis.model_validate(analysis)
            async for analysis in cls.get_store().values.aio()
        ]

    @classmethod
    async def clear(cls):
        await cls.get_store().clear.aio()

    @classmethod
    def get_store(cls) -> modal.Dict:
        return modal.Dict.from_name('analyses', create_if_missing=True)  # type: ignore
