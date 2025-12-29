



import re
import json
from typing import Any


async def generate_tool_call_async(
    model_name: str, 
    client: any, 
    question: str, 
    tools: list, 
    prompt_passing_in_english: bool
) -> str:
    developer_message = {
        "role": "developer",
        "content": (
            "You are an expert in composing functions. "
            "You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
            "If none of the functions can be used, point it out. "
            "If the given question lacks the parameters required by the function, also point it out.\n\n"
            "You should ONLY return function calls in your response. "
            "You MUST NOT include any other text, explanations, or direct answers. "
            "If you decide to invoke any function(s), you MUST use the provided tools. "
            "Do NOT attempt to answer the question directly without using the available functions. "
            ("IMPORTANT: Pass all parameter values in English" if prompt_passing_in_english else "")
        )
    }
    input_messages = [
        developer_message,
        {"role": "user", "content": question}
    ]
    response = await client.responses.create(
        model = model_name,
        input = input_messages,
        tools = tools,
        parallel_tool_calls = False,
    )
    response_dicts = [response_item.model_dump(exclude_none=True) for response_item in response.output]
    import json
    response_json_str = json.dumps(response_dicts)
    return response_json_str

async def translate_tool_question_async(
    model_name: str, 
    client: any, 
    question: str
) -> str:
    messages = [
            {
                "role": "developer",
                "content": "You are a professional translator. Translate the given text to English accurately. If the given text is already in English or is language agnostic, return it unchanged."
            },
            {
                "role": "user",
                "content": f"Translate the following question to English. Only output the translated question, nothing else:\n\n{question}"
            }
        ]
    response = await client.responses.create(
        model = model_name,
        input = messages,
    )
    return response.output_text.strip()

async def translate_tool_parameter_async(model_name: str, client: any, parameter_value: str) -> str:
    messages = [
            {
                "role": "developer",
                "content": "You are a professional translator. Translate the given text to English accurately. If the given text is already in English or is language agnostic, return it unchanged."
            },
            {
                "role": "user",
                "content": f"Translate the following text to English. Only output the translated text, nothing else:\n\n{parameter_value}"
            }
        ]
    response = await client.responses.create(
        model = model_name,
        input = messages,
    )
    return response.output_text.strip()

# The mismatch names in the prompt must match ToolErrorCategory names in src/tool/error_analysis.rs
async def categorize_parameter_value_async(
    model_name: str,
    client: Any,
    actual_value: str,
    expected_values: str,
) -> str:
    system_prompt = """You are a parameter value categorization system. Given a parameter with its actual value and expected values, determine which category the mismatch belongs to.

Here are the 6 available categories for parameter value mismatches:
1. WRONG_VALUE: The output value is COMPLETELY incorrect (wrong calculation, wrong fact, unrelated content). If some words or meanings overlap with expected values, choose relevant_but_incorrect instead.
2. RELEVANT_BUT_INCORRECT: The value is in English, relevant to the expected values, but not exactly the same in meaning.
3. EXACTLY_SAME_MEANING: The value is in English and conveys the exact same meaning as one of the expected values, though not verbatim.
4. LANGUAGE_MISMATCH_WRONG_VALUE: The value contains non-English text AND is completely incorrect.
5. LANGUAGE_MISMATCH_RELEVANT_BUT_INCORRECT: The value contains non-English text AND is relevant but not exactly correct.
6. LANGUAGE_MISMATCH_EXACTLY_SAME_MEANING: The value contains non-English text AND conveys the same meaning as expected.

CRITICAL: You must put your final decision inside \\boxed{} like this: \\boxed{category_name}
where category_name is exactly one of: WRONG_VALUE, RELEVANT_BUT_INCORRECT, EXACTLY_SAME_MEANING, LANGUAGE_MISMATCH_WRONG_VALUE, LANGUAGE_MISMATCH_RELEVANT_BUT_INCORRECT, or LANGUAGE_MISMATCH_EXACTLY_SAME_MEANING."""
    user_prompt = f"""Actual value: {actual_value}
Expected values: {expected_values}

Which category does this parameter value mismatch belong to?
Put your final answer in \\boxed{{category_name}}."""

    # Make API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        # Extract and validate response content
        if not response.choices or len(response.choices) == 0:
            return "LLM returns no choices"

        content = response.choices[0].message.content
        if content is None or not content.strip():
            return "LLM returns empty content"
        
        # Extract category from \boxed{category_name}
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, content)

        if not match:
            return "LLM returns no boxed category"
        raw_category = match.group(1).strip().upper()

        return raw_category

    except Exception as e:
        # return "LLM returns exception"
        print(f"Exception during categorize_parameter_value_async: {e}")
        exit(1)



async def generate_styled_answers_async(
    model_name: str,
    client: Any,
    index: int,
    lang: str,
    question: str,
    response: str,
    answer_correct: str,
    answer_incorrect: str,
) -> dict:
    '''
    returns a dict with keys: styled_response_correct, styled_response_incorrect
    '''
    system_prompt = (
        "You are a helpful assistant. The user is going to provide you a question, an LLM's response to the question, and two extra answers. "
        "Your task is to merge the style of the LLM's response with the two given answers, and produce two responses that have the same meaning as the given answers but in the style of the LLM's response. "
        "The two synthesized responses must be IDENTICAL except for the very essence of the answers. "
        "The essence of the answers in the two responses should be enclosed in <answer> and </answer> tags. "
        # "You may slightly modify the wording of the two answers and the LLM's response to ensure coherence in the synthesized responses. "
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
        "5. Did you make up styles or sentence framing that do not exist in the LLM's response? If the LLM's response contains only the direct answer without any framing, then wrap the answer with <answer> tags directly without adding any other styles or sentence framing.\n"
        'Finally, output the final version of the two responses in JSON format with keys "response_1" and "response_2".'
    )

    user_prompt = (
        f"Question: {question}\n"
        f"LLM's Response: {response}\n"
        f"Answer 1: {answer_correct}\n"
        f"Answer 2: {answer_incorrect}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        max_trials = 3
        num_trials = 0
        while True:
            num_trials += 1
            if num_trials > max_trials:
                # return {"error": "Exceeded maximum trials for generating styled answers"}
                # print(f"When generating for question {question}, exceeded maximum trials for generating styled answers")
                # exit(1)
                # the fallback is to wrap the original answers with <answer> directly
                print(f"When generating for lang {lang}, question index {index}, question: {question[:50]}, exceeded maximum trials for generating styled answers, use fallback")
                styled_response_1 = f"<answer>{answer_correct}</answer>."
                styled_response_2 = f"<answer>{answer_incorrect}</answer>."
                return {
                    "styled_response_correct": styled_response_1,
                    "styled_response_incorrect": styled_response_2
                }
            assert model_name == 'gpt-4.1'
            api_response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            if not api_response.choices or len(api_response.choices) == 0:
                continue  # Retry

            content = api_response.choices[0].message.content
            if content is None or not content.strip():
                continue  # Retry

            # Extract JSON from the response
            # Look for JSON object pattern in the content - use the last match
            json_matches = re.findall(r'\{[^{}]*"response_1"[^{}]*"response_2"[^{}]*\}', content, re.DOTALL)
            if not json_matches:
                continue  # Retry

            json_str = json_matches[-1]  # Use the last match

            try:
                parsed_json = json.loads(json_str)
            except json.JSONDecodeError:
                continue  # Retry

            # Validate the parsed JSON has required keys
            if "response_1" not in parsed_json or "response_2" not in parsed_json:
                continue  # Retry

            styled_response_1 = parsed_json["response_1"]
            styled_response_2 = parsed_json["response_2"]

            # Validate that responses contain answer tags
            if '<answer>' not in styled_response_1 or '</answer>' not in styled_response_1:
                continue  # Retry
            if '<answer>' not in styled_response_2 or '</answer>' not in styled_response_2:
                continue  # Retry

            return {
                "styled_response_correct": styled_response_1,
                "styled_response_incorrect": styled_response_2
            }

    except Exception as e:
        print(f"Exception during generate_styled_answers_async: {e}")
        return {"error": f"Exception: {str(e)}"}
