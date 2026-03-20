import re
import inspect
from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATION = 10
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"   >> Exceuting get_product_price[product='{product}']")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Available tiers: bronze, silver, gold."""

    price = float(price)
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentage.get(discount_tier, 0)

    final_prize = round(price - (price * discount)/100, 2)

    print(f"Final price is being returned: {final_prize}")

    return final_prize


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_tool_descriptions(tools_dict):
    decription = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        decription.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(decription)

tool_description = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
    "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price."
                "You must call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself applying math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - DO NOT assume one."

    Answer the following questions as best you can. You have access to the following tools:

    {tool_description}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {{question}}
    Thought:
"""



@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=MODEL, messages=messages, options=options)

# ----- Agent Loop ------

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    prompt = react_prompt.format(question = question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"---- Iteration: {iteration} ----")

        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages = [{"role": "user", "content": full_prompt}],
            options={
                "stop": ["\nObservation:", "Observation:", "\n Observation:"],
                "temperature": 0
            }
        )

        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(f"[Parsing] Looking for final answer in LLM output..... ")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "=" * 60)
            print(f"Final answer: {final_answer}")

            return final_answer

        print(f"[Parsing] looking for Action and Action input in LLM output....")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("ERROR: could not parse action or action input from LLM output")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f" [Tool Selected] {tool_name} with args {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f" [Tool Executing] {tool_name} ({args})...")

        if tool_name not in tools:
            observation = f"tool name {tool_name} not found in the tools list. Available tools: {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f" [Tool Results] {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("ERROR: Max iteration received without a final answer")
    return None



if __name__ == "__main__":
    print("Hello LangChain agent")
    
    result = run_agent("What is the price of laptop after applying gold discount")