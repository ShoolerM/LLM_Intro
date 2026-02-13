#!/usr/bin/python3

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

OPENROUTER_API_KEY = (
    "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"
)

# System prompt: set via environment variable `SYSTEM_PROMPT` or default below
SYSTEM_PROMPT = "You are pirate. You are ALWAYS a pirate. You were born a pirate. You will die a pirate. You are a pirate in every way. You are a pirate in every moment. You are a pirate in every thought. You are a pirate in every word. You are a pirate in every action. You are a pirate in every interaction. You are a pirate in every relationship. You are a pirate in every situation. You are a pirate in every circumstance. You are a pirate in every environment. You are a pirate in every context. You are a pirate in every reality. You are a pirate in every universe. You are a pirate in every multiverse."


def main():
    # Initialize the ChatOpenAI model with OpenRouter
    chat = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",  # The model we are using for the chatbot.
        api_key=OPENROUTER_API_KEY,  # Your OpenRouter API key for authentication.
        base_url="https://openrouter.ai/api/v1",  # The URL for the OpenRouter Inference endpoint.
        temperature=0.1,  # Adjust temperature for more creative responses
    )

    # Store conversation history and include system prompt
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    print("Welcome to the LangChain Chatbot!")
    print("Type 'quit' to exit.\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # Add user message to history
        messages.append(HumanMessage(content=user_input))

        # Get response from the model
        response = chat.invoke(messages)

        # Add AI response to history
        messages.append(AIMessage(content=response.content))

        print(f"Assistant: {response.content}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
