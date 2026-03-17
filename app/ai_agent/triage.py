from app.core.config import get_settings, Settings
from app.ai_agent.tools import search_docs

from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI

set_tracing_disabled(True)
settings: Settings = get_settings()

open_router_client = AsyncOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)

open_router_model = OpenAIChatCompletionsModel(
    model="nvidia/nemotron-3-super-120b-a12b:free", openai_client=open_router_client
)

triage_agent = Agent(
    name="SupportBrainAgent",
    instructions=(
        "You are a helpful customer support assistant for SupportBrain. "
        "Use the search_docs tool to find answers from the documentation. "
        "If the tool returns that it lacks enough information, let the user know "
        "and suggest they contact human support."
    ),
    model=open_router_model,
    tools=[search_docs],
)


async def agent(message: str):
    result = await Runner.run(starting_agent=triage_agent, input=message)
    return {"agent_reply": result.final_output}
