from temporalio import workflow


@workflow.defn
class PydanticAIPluginSandboxWorkflow:
    @workflow.run
    async def run(self) -> str:
        return 'sandboxed'
