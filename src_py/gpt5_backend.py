



async def generate_tool_call_async(model_name: str, client: any, question: str, tools: list, prompt_passing_in_english: bool) -> str:
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
            "Do NOT attempt to answer the question directly without using the available functions."
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
    )
    response_dicts = [response_item.model_dump(exclude_none=True) for response_item in response.output]
    import json
    response_json_str = json.dumps(response_dicts)
    return response_json_str

async def translate_tool_question_async(model_name: str, client: any, question: str) -> str:
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

async def translate_tool_answer_async(model_name: str, client: any, parameter_value: str) -> str:
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