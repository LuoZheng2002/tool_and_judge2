import os
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
    system_prompt = ("You are a helpful assistant. The user is going to provide you a question, an LLM's response to the question, and two extra answers. "
    "Your task is to merge the style of the LLM's response with the two given answers, and produce two responses that have the same meaning as the given answers but in the style of the LLM's response. "
    "The two synthesized responses must be IDENTICAL except for the very essence of the answers. "
    "You should only output two lines, each containing one synthesized response. Do not include any additional text or explanations. The essence of the answers in the two responses should be enclosed in <answer> and </answer> tags. "
    "You may slightly modify the wording of the two answers and the LLM's response to ensure coherence in the synthesized responses. "
    "If the LLM's response contains any content that is related to the decision of the answer, you should discard it in the synthesized responses. "
    "\n\n"
    "Here is an example:\n"
    "Question: Judge the following statements: 1+1=3. All integers are either even or odd.\n"
    "LLM's Response: The first statement is incorrect. 1+1=2. The second statement is correct. All integers are either even or odd.\n"
    "Answer 1: True, True\n"
    "Answer 2: False, False\n"
    "Your output:\n\n"
    "The first statement is <answer>true</answer>. The second statement is <answer>true</answer>.\n"
    "The first statement is <answer>false</answer>. The second statement is <answer>false</answer>.\n\n"
    "Now, please process the user's input accordingly.")
    user_prompt = ("Question: Which of the following best describes the structure that collects urine in the body?\n"
    "LLM's Response: The structure that collects urine in the body is the **urinary bladder**.\n"
    "Answer 1: Bladder\n"
    "Answer 2: Kidney")

    result = call_gpt4(system_prompt, user_prompt)

    if result:
        print(f"Response: {result}")
