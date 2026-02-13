#!/usr/bin/python3

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

OPENROUTER_API_KEY = ""

SYSTEM_PROMPT = "You are a helpful AI assistant."


# Implementation function (can be called directly)
def _get_weather_impl(city: str) -> str:
    """Mock weather fetching for demo purposes."""
    # Mock weather data for common cities
    mock_weather = {
        "london": "London: Cloudy, 48°F",
        "minneapolis": "Minneapolis: Sunny, 28°F",
        "new york": "New York: Rainy, 42°F",
        "tokyo": "Tokyo: Clear, 59°F",
        "paris": "Paris: Cloudy, 45°F",
    }
    city_lower = city.lower().strip()
    return mock_weather.get(city_lower, f"{city}: Partly cloudy, 50°F (mock data)")


# Tool definition for the LLM
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return _get_weather_impl(city)


def main():
    # Initialize the ChatOpenAI model with OpenRouter
    chat = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    # Define what tools we can use
    tools = [get_weather]

    # Bind the tools to the LLM so it can call them when needed
    llm_with_tools = chat.bind_tools(tools)

    history = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # Add user message to history
        history.append(HumanMessage(content=user_input))

        # Get response from model with tools
        print("Assistant: ", end="", flush=True)
        response = llm_with_tools.invoke(history)

        # Check if model called a tool
        if response.tool_calls:
            # For each tool call, execute the tool and get the result
            for tool_call in response.tool_calls:
                if tool_call["name"] == "get_weather":
                    city = tool_call["args"]["city"]
                    weather = _get_weather_impl(city)
                    history.append(response)
                    history.append(
                        ToolMessage(content=weather, tool_call_id=tool_call["id"])
                    )
                    # Get final response
                    final_response = llm_with_tools.invoke(history)
                    print(final_response.content)
        else:
            # No tool call, just print response
            print(response.content)

        print()
        history.append(response)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
