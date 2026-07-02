from math import sqrt

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def square_root(x: float) -> float:
    """Compute the square root of a positive number."""
    return sqrt(x)


tools = {
    "multiply": multiply,
    "square_root": square_root,
}

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
).bind_tools(list(tools.values()))

messages = [
    HumanMessage(
        content=(
            "Give me the answer to 123 * 456, "
            "and if the result is greater than 1000, "
            "give me its square root."
        )
    )
]

while True:

    response = llm.invoke(messages)

    # Show what the model is thinking
    print("\nLLM:")
    print(response)

    # No tool calls? We're done.
    if not response.tool_calls:
        print("\nFinal answer:")
        print(response.content)
        break

    messages.append(response)

    for tool_call in response.tool_calls:

        print(f"\nExecuting {tool_call['name']}({tool_call['args']})")

        tool = tools[tool_call["name"]]

        result = tool.invoke(tool_call["args"])

        print("Result:", result)

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )