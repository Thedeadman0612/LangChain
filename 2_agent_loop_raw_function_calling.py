import trace
from dotenv import load_dotenv
from urllib3 import response

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

    discount_percentage = {"broze": 5, "silver": 12, "gold": 23}
    discount = discount_percentage.get(discount_tier, 0)

    return round(price - (price * discount)/100, 2)


tools_for_llm=[
    {
      'type': 'function',
      'function': {
        'name': 'get_product_price',
        'description': 'Look up the price of the product in the catalog',
        'parameters': {
          'type': 'object',
          'properties': {
            'product': {
              'type': 'string',
              'description': "The product name e.g. 'laptop', 'headphones', 'keyborad' ",
            },
          },
          'required': ['product'],
        },
      },
    },
    {
      'type': 'function',
      'function': {
        'name': 'apply_discount',
        'description': 'Apply a discount tier to a price and return the final price Available tiers: bronze, silver, gold.',
        'parameters': {
          'type': 'object',
          'properties': {
            'product': {
              'type': 'string',
              'description': "The discount tier e.g. 'bronze', 'silver', 'gold' ",
            },
          },
          'required': ['price', 'discount_tier'],
        },
      },
    },
  ]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

# ----- Agent Loop ------

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content":
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
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
            ,
        },
        {"role": "user", "content": question},     
    ]

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"---- Iteration: {iteration} ----")

        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_call = ai_message.tool_calls

        if not tool_call:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # process only the first tool call - force one tool per iteration
        tool_call = tool_call[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f" [Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use(**tool_args)

        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation)
            }
        )


    print("ERROR: Max iteration received without a final answer")
    return None



if __name__ == "__main__":
    print("Hello LangChain agent")
    
    result = run_agent("What is the price of laptop after applying gold discount")