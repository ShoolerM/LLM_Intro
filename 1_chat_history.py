#!/usr/bin/python3

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

OPENROUTER_API_KEY = (
    "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"
)


def main():
    # Initialize the ChatOpenAI model with OpenRouter
    chat = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",  # The model we are using for the chatbot.
        api_key=OPENROUTER_API_KEY,  # Your OpenRouter API key for authentication.
        base_url="https://openrouter.ai/api/v1",  # The URL for the OpenRouter Inference endpoint.
        temperature=0.7,  # Adjust temperature for more creative responses
        # Temperature controls the randomness of the model's output.
        # A higher temperature (e.g., 0.8) will make the output more creative and diverse,
        # while a lower temperature (e.g., 0.2) will make it more focused and deterministic.
        # Adjusting the temperature can help you find the right balance between creativity
        # and coherence in the model's responses.
    )

    # Store conversation history
    messages = []

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
