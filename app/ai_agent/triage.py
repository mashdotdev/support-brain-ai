from app.core.config import get_settings, Settings

from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI

set_tracing_disabled(True)
settings: Settings = get_settings()

gemini_client = AsyncOpenAI(
    api_key=settings.gemini_api_key, base_url=settings.gemini_base_url
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", openai_client=gemini_client
)

print(settings.gemini_api_key)


async def agent(message: str):
    triage_agent = Agent(
        name="SupportBrainAgent",
        instructions="You are a agent for Support Brain",
        model=gemini_model,
    )

    result = await Runner.run(starting_agent=triage_agent, input=message)

    return {"agent_reply": result.final_output}
