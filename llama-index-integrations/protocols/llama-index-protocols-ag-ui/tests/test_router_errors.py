import asyncio

import pytest
from ag_ui.core import RunAgentInput

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.protocols.ag_ui.router import AGUIWorkflowRouter

SECRET = "404 Client Error for url: http://10.0.0.5/internal"


class FailingWorkflow(Workflow):
    @step
    async def fail(self, ev: StartEvent) -> StopEvent:
        raise RuntimeError(SECRET)


def test_run_error_does_not_leak_exception_details() -> None:
    async def factory() -> Workflow:
        return FailingWorkflow()

    router = AGUIWorkflowRouter(factory)
    run_input = RunAgentInput(
        thread_id="t",
        run_id="r",
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props={},
    )

    async def collect() -> str:
        response = await router.run(run_input)
        chunks = []
        with pytest.raises(RuntimeError):
            async for chunk in response.body_iterator:
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
        return "".join(chunks)

    text = asyncio.run(collect())
    assert "RUN_ERROR" in text
    assert "INTERNAL_ERROR" in text
    assert SECRET not in text
    assert "RuntimeError" not in text
