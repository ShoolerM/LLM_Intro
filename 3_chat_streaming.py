#!/usr/bin/python3

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time

OPENROUTER_API_KEY = (
    "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"
)

SYSTEM_PROMPT = "You are a pirate."


def main():
    # Initialize the ChatOpenAI model with OpenRouter
    chat = ChatOpenAI(
        # model="openai/gpt-oss-120b:free",
        model="meta-llama/llama-3.3-70b-instruct:free",  # The model we are using for the chatbot.
        api_key=OPENROUTER_API_KEY,  # Your OpenRouter API key for authentication.
        base_url="https://openrouter.ai/api/v1",  # The URL for the OpenRouter Inference endpoint.
        stream_usage=True,  # Enable streaming responses from the model
    )

    print("Welcome to the LangChain Chatbot!")
    print("Type 'quit' to exit.\n")

    # conversation history starts with system prompt
    history = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # add user message to history
        history.append(HumanMessage(content=user_input))

        # Init the Assistant reply
        print("Assistant: ", end="", flush=True)

        assistant_text = ""
        # Stream tokens from the model using full history
        for token in chat.stream(history):
            # Print the stream to the terminal and accumulate
            print(f"{token.content}", end="", flush=True)
            assistant_text += token.content

        print("\n")  # Print a newline after the response is complete

        # save assistant reply into history
        history.append(AIMessage(content=assistant_text))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
