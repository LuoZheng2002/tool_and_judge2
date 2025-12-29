import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_gpt4(system_prompt: str, user_prompt: str, model: str = "gpt-4-turbo") -> str:
    """
    Call GPT-4 with system and user prompts.

    Args:
        system_prompt: The system prompt to set the context/behavior
        user_prompt: The user's input/question
        model: The model to use (default: gpt-4-turbo)

    Returns:
        The assistant's response as a string
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling GPT-4: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    system_prompt = (
        "You are a helpful assistant. The user is going to provide you a question, an LLM's response to the question, and two extra answers. "
        "Your task is to merge the style of the LLM's response with the two given answers, and produce two responses that have the same meaning as the given answers but in the style of the LLM's response. "
        "The two synthesized responses must be IDENTICAL except for the very essence of the answers. "
        "The essence of the answers in the two responses should be enclosed in <answer> and </answer> tags. "
        "You may slightly modify the wording of the two answers and the LLM's response to ensure coherence in the synthesized responses. "
        "If the LLM's response contains any content that is related to the decision of the answer, you should discard it in the synthesized responses. "
        "\n\n"
        "Here is an example:\n"
        "Question: Judge the following statements: 1+1=3. All integers are either even or odd.\n"
        "LLM's Response: The first statement is incorrect. 1+1=2. The second statement is **correct**. All integers are either even or odd.\n"
        "Answer 1: True, True\n"
        "Answer 2: False, False\n"
        "Your final output:\n"
        "{\n"
        '  "response_1": "The first statement is <answer>true</answer>. The second statement is **<answer>true</answer>**.",\n'
        '  "response_2": "The first statement is <answer>false</answer>. The second statement is **<answer>false</answer>**."\n'
        "}\n\n"
        "Begin your response with your first trial of generating the two synthesized answers WITHOUT using JSON format. Then check the following:\n"
        "1. Is the essence of the answers enclosed in <answer> and </answer> tags while the rest of the content is in the style of the LLM's response?\n"
        "2. Apart from the content inside the <answer> tags, are the two responses identical?\n"
        "3. Does the content outside the <answer> tags reveal the decision of the answers? If so, it should be removed.\n"
        "4. Are the details from the LLM's response faithfully preserved, including letter cases and special decorations like \"**\" for bold?\n"
        'Finally, output the final version of the two responses in JSON format with keys "response_1" and "response_2".'
    )
    user_prompt = ("Question: Which of the following best describes the structure that collects urine in the body?\n"
    "LLM's Response: The structure that collects urine in the body is the **urinary bladder**.\n"
    "Answer 1: Bladder\n"
    "Answer 2: Kidney")
    
    print(f"user_prompt: {user_prompt}")

    result = call_gpt4(system_prompt, user_prompt)

    if result:
        print(f"Raw Response: {result}\n")

        # Extract and parse JSON from the response - use the last match
        json_matches = re.findall(r'\{[^{}]*"response_1"[^{}]*"response_2"[^{}]*\}', result, re.DOTALL)

        if json_matches:
            json_str = json_matches[-1]  # Use the last match
            try:
                parsed_json = json.loads(json_str)
                print(f"Parsed JSON:")
                print(f"  response_1: {parsed_json.get('response_1')}")
                print(f"  response_2: {parsed_json.get('response_2')}")
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
        else:
            print("No JSON object found in response")
