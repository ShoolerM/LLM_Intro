#!/usr/bin/python3

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

OPENROUTER_API_KEY = ""


def main():
    # Initialize the ChatOpenAI model with OpenRouter
    chat = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",  # The model we are using for the chatbot.
        api_key=OPENROUTER_API_KEY,  # Your OpenRouter API key for authentication.
        base_url="https://openrouter.ai/api/v1",  # The URL for the OpenRouter Inference endpoint.
    )

    while True:
        # Get user input
        user_input = input("You: ").strip()

        # The user didn't input anything, we don't want to send a blank line to the LLM.
        if not user_input:
            continue

        # Get response from the model
        response = chat.invoke(
            [HumanMessage(content=user_input)],
        )

        print(f"Assistant: {response.content}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
