"""
Unified Granite 4 interface for both tool and judge projects.

This interface supports:
- Tool project: Function calling with <tool_call> format
- Judge project: Perplexity calculation and preference comparison

Granite 4 models:
- Uses special tokens: <|start_of_role|>, <|end_of_role|>, <|end_of_text|>
- Tool calling: <tool_call>{...}</tool_call> format
- Follows OpenAI's function definition schema
"""

import json
import re
from typing import List, Dict, Any, Union, Optional, TYPE_CHECKING, Tuple
from .base import (
    JudgeModelInterface,
    ToolModelInterface,
    ModelBackend,
    ComparisonResult,
    DirectComparisonResult,
    CoTComparisonResult,
    ForwardResult,
)

from models.name_mapping import FunctionNameMapper
from config import EvaluationError


class Granite4Interface(JudgeModelInterface, ToolModelInterface):
    """
    Unified interface for Granite 4 models supporting both tool and judge use cases.

    This interface inherits from both JudgeModelInterface and ToolModelInterface,
    providing functionality for:
    - Function calling (tool project)
    - Perplexity calculation (judge project)
    - Preference comparison (judge project)
    """

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-4-8b-instruct",
        direct_temperature: float = 0.0,
        reason_temperature: float = 0.0
    ):
        """
        Initialize the Granite 4 interface.

        Args:
            model_name: Model identifier (e.g., "ibm-granite/granite-4-8b-instruct",
                       "ibm-granite/granite-4-20b-instruct")
            direct_temperature: Temperature for direct comparison (without reasoning)
            reason_temperature: Temperature for reasoning-based comparison (with CoT)
        """
        self.model_name = model_name
        self.direct_temperature = direct_temperature
        self.reason_temperature = reason_temperature

    # =========================================================================
    # ModelInterface Methods
    # =========================================================================

    def get_model_name(self) -> str:
        """Get the model name/identifier."""
        return self.model_name

    # =========================================================================
    # ToolModelInterface Methods
    # =========================================================================

    async def generate_tool_call_async(
        self,
        backend: ModelBackend,
        raw_functions: List[Dict[str, Any]],
        user_query: str,
        name_mapper: FunctionNameMapper,
        prompt_passing_in_english: bool,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate tool/function calls from a user query.

        This method:
        1. Gets tokenizer from backend
        2. Formats tools in Granite's expected format
        3. Uses apply_chat_template to generate the prompt
        4. Calls backend to generate
        5. Returns raw output (with <tool_call> tags)

        Args:
            backend: The backend to use for inference
            raw_functions: List of available function definitions
            user_query: User query as a string
            name_mapper: Function name mapper (unused for Granite)
            prompt_passing_in_english: Whether to request English parameter passing
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Raw model output as a string
        """
        # Get tokenizer from backend
        tokenizer = backend.get_tokenizer()

        # Build system prompt
        system_prompt = self._generate_system_prompt(
            functions=raw_functions,
            prompt_passing_in_english=prompt_passing_in_english
        )

        # Convert raw_functions to Granite's tools format
        # Granite follows OpenAI's function definition schema
        tools = []
        for func in raw_functions:
            if "type" in func and func["type"] == "function":
                # Already in wrapped format
                tools.append(func)
            else:
                # Need to wrap
                func_copy = func.copy()

                # Convert "type": "dict" to "type": "object" in parameters
                if "parameters" in func_copy and isinstance(func_copy["parameters"], dict):
                    params = func_copy["parameters"]
                    if params.get("type") == "dict":
                        params["type"] = "object"

                # Wrap in the expected format
                tools.append({
                    "type": "function",
                    "function": func_copy
                })

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        # Use apply_chat_template to generate the prompt
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Call backend
        result = await backend.generate_async(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return result.generated_text

    def preprocess_functions(
        self,
        functions: List[Dict[str, Any]],
        name_mapper: Optional['FunctionNameMapper']
    ) -> List[Dict[str, Any]]:
        """
        Preprocess function definitions for Granite 4.

        Granite 4 doesn't require name sanitization, so this returns functions unchanged.

        Args:
            functions: List of function definitions
            name_mapper: External name mapper (unused for Granite 4)

        Returns:
            Preprocessed function definitions (unchanged for Granite 4)
        """
        return functions

    def postprocess_tool_calls(
        self,
        raw_output: str,
        name_mapper: Optional['FunctionNameMapper'] = None
    ) -> Union[List[Dict[str, Dict[str, Any]]], Tuple[EvaluationError, Dict[str, Any]]]:
        """
        Postprocess raw output from Granite 4 model to extract function calls.

        Granite 4 outputs in JSON format within <tool_call> tags:
        <tool_call>
        {"name": "function_name", "arguments": {"param1": value1, ...}}
        </tool_call>

        Note: Granite 4 doesn't require name sanitization, so name_mapper is unused.

        Args:
            raw_output: Raw string output from the model
            name_mapper: Unused for Granite 4 (no name sanitization needed)

        Returns:
            On success: List of function calls
            On error: Tuple of (EvaluationError, metadata dict with error details)
        """
        # Parse Granite 4 model's output format: <tool_call>{...}</tool_call>
        model_result_raw = raw_output.strip()

        # Extract content from <tool_call> tags if present
        if "<tool_call>" in model_result_raw:
            # Handle potential multiple tool calls
            tool_calls_list = []
            remaining = model_result_raw

            while "<tool_call>" in remaining:
                start_idx = remaining.find("<tool_call>")
                end_idx = remaining.find("</tool_call>")

                if start_idx != -1 and end_idx != -1:
                    tool_call_content = remaining[start_idx + len("<tool_call>"):end_idx].strip()
                    if tool_call_content:
                        tool_calls_list.append(tool_call_content)
                    remaining = remaining[end_idx + len("</tool_call>"):]
                else:
                    break

            # If we found tool calls, join them as a JSON array
            if tool_calls_list:
                # Each tool call should be a JSON object
                model_result_raw = "[" + ",".join(tool_calls_list) + "]"
            else:
                # Fallback: extract first tool call content
                start_idx = model_result_raw.find("<tool_call>")
                end_idx = model_result_raw.find("</tool_call>")
                if start_idx != -1 and end_idx != -1:
                    model_result_raw = model_result_raw[start_idx + len("<tool_call>"):end_idx]
                    model_result_raw = model_result_raw.strip()

        # Strip backticks and whitespace
        model_result_raw = model_result_raw.strip("`\n ")

        # Add brackets if missing (for single objects or arrays)
        if not model_result_raw.startswith("["):
            # Try to parse as single JSON object first
            if model_result_raw.startswith("{"):
                model_result_raw = "[" + model_result_raw + "]"
            else:
                model_result_raw = "[" + model_result_raw

        if not model_result_raw.endswith("]"):
            model_result_raw = model_result_raw + "]"

        try:
            # Parse the JSON array
            tool_calls = json.loads(model_result_raw)
        except json.JSONDecodeError as e:
            return (EvaluationError.JSON_DECODE_ERROR, {
                "error_message": str(e),
                "raw_output": raw_output
            })

        # Convert Granite 4 format to desired format
        extracted = []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                    func_name = tool_call["name"]
                    func_args = tool_call["arguments"]
                    extracted.append({func_name: func_args})
                else:
                    return (EvaluationError.PARSING_ERROR, {
                        "error_message": "Invalid tool call structure",
                        "raw_output": raw_output
                    })
        else:
            return (EvaluationError.PARSING_ERROR, {
                "error_message": "Expected a list of tool calls",
                "raw_output": raw_output
            })

        if extracted:
            return extracted
        else:
            return (EvaluationError.NO_FUNCTION_CALLS_FOUND, {
                "raw_output": raw_output
            })

    async def translate_tool_question_async(
        self,
        backend: ModelBackend,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Translate a user question to English using Granite 4.

        Args:
            backend: The backend to use for inference
            question: The question text to translate
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Translated question as a string
        """
        # Build translation prompt
        prompt_text = (
            "You are a professional translator. Translate the given text to English accurately. "
            "If the given text is already in English or is language agnostic, return it unchanged.\n\n"
            f"Translate the following question to English. Only output the translated question, nothing else:\n\n{question}"
        )

        # Format with Granite 4 chat template
        formatted_prompt = (
            f"<|start_of_role|>system<|end_of_role|>You are a professional translator.<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{prompt_text}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>"
        )

        # Call backend
        result = await backend.generate_async(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return result.generated_text.strip()

    async def translate_tool_answer_async(
        self,
        backend: ModelBackend,
        parameter_value: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """
        Translate a single function parameter value to English using Granite 4.

        Args:
            backend: The backend to use for inference
            parameter_value: The parameter value to translate
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Translated parameter value as a string
        """
        # Build translation prompt
        prompt_text = (
            "You are a professional translator. Translate the given text to English accurately. "
            "If the given text is already in English or is language agnostic, return it unchanged.\n\n"
            f"Translate the following text to English. Only output the translated text, nothing else:\n\n{parameter_value}"
        )

        # Format with Granite 4 chat template
        formatted_prompt = (
            f"<|start_of_role|>system<|end_of_role|>You are a professional translator.<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{prompt_text}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>"
        )

        # Call backend
        result = await backend.generate_async(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return result.generated_text.strip()

    # =========================================================================
    # JudgeModelInterface Methods
    # =========================================================================

    async def compare_directly_async(
        self,
        backend: ModelBackend,
        question: str,
        answer1: str,
        answer2: str,
        **kwargs
    ) -> DirectComparisonResult:
        """
        Compare two answers directly without reasoning.

        This method:
        1. Formats the comparison prompt
        2. Calls backend to generate with logprobs
        3. Parses output to extract preference (1 or 2)
        4. Extracts probability logits for choices "1" and "2"

        Args:
            backend: The backend to use for inference
            question: The question being answered
            answer1: First answer to compare
            answer2: Second answer to compare
            **kwargs: Additional model-specific parameters

        Returns:
            DirectComparisonResult with preference, probability logits, and optional error
        """
        # Build comparison prompt
        prompt_text = f"""Given the following question and two answers, which answer is better?

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Provide your judgment IMMEDIATELY without reasoning or explanation. Provide your final decision in the following format:
\\boxed{{X}} where X is either 1 or 2."""

        # Format with Granite 4 chat template
        formatted_prompt = (
            f"<|start_of_role|>system<|end_of_role|>You are an expert judge. Provide only the final decision in the requested format.<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{prompt_text}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>"
        )

        # Call backend with logprobs enabled
        result = await backend.generate_async(
            prompt=formatted_prompt,
            max_new_tokens=100,
            temperature=self.direct_temperature,
            return_logprobs=True,  # Request logprobs for probability extraction
            **kwargs
        )

        # Parse preference from output
        raw_output = result.generated_text

        # Try to parse preference
        preference = None
        error = None
        try:
            preference = self._parse_preference(raw_output)
        except ValueError as e:
            error = str(e)

        # Extract probability logits for "1" and "2" (MANDATORY)
        logit_1, logit_2 = self._extract_choice_probabilities(backend, result)

        return DirectComparisonResult(
            preference=preference,
            raw_output=raw_output,
            logit_1=logit_1,
            logit_2=logit_2,
            error=error
        )

    async def compare_thinking_async(
        self,
        backend: ModelBackend,
        question: str,
        answer1: str,
        answer2: str,
        **kwargs
    ) -> CoTComparisonResult:
        """
        Compare two answers with chain-of-thought reasoning.

        This method:
        1. Formats the comparison prompt (encouraging reasoning)
        2. Calls backend with reasoning enabled
        3. Parses output to extract reasoning and preference
        4. Returns merged raw output (reasoning already included)

        Args:
            backend: The backend to use for inference
            question: The question being answered
            answer1: First answer to compare
            answer2: Second answer to compare
            **kwargs: Additional model-specific parameters

        Returns:
            CoTComparisonResult with preference, merged raw output, and optional error
        """
        # Build comparison prompt with CoT instruction
        prompt_text = f"""Given the following question and two answers, which answer is better?

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Please briefly explain your reasoning, and then provide your final decision in the following format:
\\boxed{{X}} where X is either 1 or 2."""

        # Format with Granite 4 chat template
        formatted_prompt = (
            f"<|start_of_role|>system<|end_of_role|>You are an expert judge. Think through your reasoning before providing the final decision.<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{prompt_text}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>"
        )

        # Call backend
        result = await backend.generate_async(
            prompt=formatted_prompt,
            max_new_tokens=512,
            temperature=self.reason_temperature,
            return_logprobs=False,  # No logprobs needed for reasoning
            **kwargs
        )

        # Parse preference and extract reasoning
        raw_output = result.generated_text

        # Try to parse preference
        preference = None
        error = None
        try:
            preference = self._parse_preference(raw_output)
        except ValueError as e:
            error = str(e)

        # raw_output already contains both reasoning and final answer merged
        return CoTComparisonResult(
            preference=preference,
            raw_output=raw_output,
            error=error
        )

    async def forward_for_logits_async(
        self,
        backend: ModelBackend,
        question: str,
        answer: str,
        language: str = "English",
        **kwargs
    ) -> ForwardResult:
        """
        Run forward pass to get logits for perplexity calculation.

        This method:
        1. Gets tokenizer from backend
        2. Formats the prompt with question and answer
        3. Applies chat template using tokenizer
        4. Calls backend.forward_async to get logits

        Args:
            backend: The backend to use for inference
            question: The question
            answer: The answer to calculate perplexity for
            language: Language name (e.g., "English", "Chinese")
            **kwargs: Additional model-specific parameters

        Returns:
            ForwardResult containing logits and input_ids
        """
        # Get tokenizer from backend
        tokenizer = backend.get_tokenizer()

        # Build language-specific instructions
        if language.lower() == "english":
            instruction = "Please answer the question in English with a concise phrase instead of a complete sentence. Start with an uncapitalized first word."
        else:
            instruction = f"Please answer the question in {language} with a concise phrase instead of a complete sentence."

        # Combine question with instruction
        user_content = f"{question}\n\n{instruction}"

        # Build messages
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]

        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Call backend for forward pass
        result = await backend.forward_async(
            prompt=formatted_prompt,
            max_length=2048,
            **kwargs
        )

        return result

    def get_assistant_prefix(self) -> str:
        """Get the Granite 4 assistant prefix used by the model."""
        return "<|start_of_role|>assistant<|end_of_role|>"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_choice_probabilities(
        self,
        backend: ModelBackend,
        result
    ) -> tuple[float, float]:
        """
        Extract probability logits for choices "1" and "2" from generation result.

        This method extracts the probabilities for tokens "1" and "2" from the
        unified logprobs format returned by the backend. This is MANDATORY - if
        logits cannot be extracted, the method raises a RuntimeError.

        Args:
            backend: The backend used for generation (to access tokenizer)
            result: GenerationResult from backend.generate_async with unified logits format

        Returns:
            Tuple of (logit_1, logit_2) containing probabilities for choices "1" and "2"

        Raises:
            RuntimeError: If logits cannot be extracted from the backend response
        """
        import math

        if result.logits is None:
            raise RuntimeError(
                "Backend failed to provide logits. "
                "Direct comparison requires logprobs to be enabled in the backend. "
                "Please ensure return_logprobs=True was passed to generate_async."
            )

        try:
            # Get tokenizer to find token IDs for "1" and "2"
            tokenizer = backend.get_tokenizer()
            token_1_id = tokenizer.encode("1", add_special_tokens=False)[0]
            token_2_id = tokenizer.encode("2", add_special_tokens=False)[0]

            # Extract logprobs from the unified format: List[Dict[int, float]]
            logprobs_data = result.logits

            if not isinstance(logprobs_data, list):
                raise RuntimeError(
                    f"Unexpected logprobs format. Expected List[Dict[int, float]], "
                    f"got {type(logprobs_data)}"
                )

            if len(logprobs_data) == 0:
                raise RuntimeError(
                    "Backend returned empty logprobs list. Expected at least one token."
                )

            # Get the first token's logprobs (the decision token)
            first_token_logprobs = logprobs_data[0]

            if not isinstance(first_token_logprobs, dict):
                raise RuntimeError(
                    f"Unexpected format for first token logprobs. "
                    f"Expected Dict[int, float], got {type(first_token_logprobs)}"
                )

            # Get logprobs for "1" and "2"
            if token_1_id not in first_token_logprobs:
                raise RuntimeError(
                    f"Token '1' (ID: {token_1_id}) not found in logprobs. "
                    f"Available tokens (first 10): {list(first_token_logprobs.keys())[:10]}..."
                )
            if token_2_id not in first_token_logprobs:
                raise RuntimeError(
                    f"Token '2' (ID: {token_2_id}) not found in logprobs. "
                    f"Available tokens (first 10): {list(first_token_logprobs.keys())[:10]}..."
                )

            # Get log probabilities
            logprob_1 = first_token_logprobs[token_1_id]
            logprob_2 = first_token_logprobs[token_2_id]

            # Convert log probabilities to probabilities
            prob_1 = math.exp(logprob_1)
            prob_2 = math.exp(logprob_2)

            return (prob_1, prob_2)

        except RuntimeError:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in RuntimeError
            raise RuntimeError(f"Failed to extract choice probabilities: {e}") from e

    def _parse_preference(self, raw_output: str) -> int:
        """
        Parse preference from model output.

        Looks for \\boxed{1} or \\boxed{2} in the output.

        Args:
            raw_output: Raw model output

        Returns:
            1 or 2 indicating preference

        Raises:
            ValueError: If preference cannot be parsed
        """
        # Look for \\boxed{1} or \\boxed{2}
        match = re.search(r'\\boxed\{(\d+)\}', raw_output)
        if match:
            preference = int(match.group(1))
            if preference in [1, 2]:
                return preference

        # Fallback: look for just "1" or "2" at end of output
        if raw_output.strip().endswith("1"):
            return 1
        elif raw_output.strip().endswith("2"):
            return 2

        raise ValueError(f"Could not parse preference from output: {raw_output}")

    def _extract_reasoning(self, raw_output: str) -> Optional[str]:
        """
        Extract reasoning text from model output.

        Gets the text before the final \\boxed{} decision.

        Args:
            raw_output: Raw model output

        Returns:
            Reasoning text, or None if not found
        """
        # Find the \\boxed{} part
        match = re.search(r'\\boxed\{(\d+)\}', raw_output)
        if match:
            # Get text before the boxed part
            reasoning = raw_output[:match.start()].strip()
            if reasoning:
                return reasoning

        return None

    def _generate_system_prompt(
        self,
        functions: List[Dict[str, Any]],
        prompt_passing_in_english: bool = True
    ) -> str:
        """
        Generate system prompt for Granite 4 based on available functions.

        Args:
            functions: List of available function definitions
            prompt_passing_in_english: Whether to request English parameter passing

        Returns:
            System prompt as a string
        """
        passing_in_english_prompt = (
            " IMPORTANT: Pass in all parameters in function calls in English."
            if prompt_passing_in_english
            else ""
        )

        return f'''You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should ONLY return function calls in your response. You MUST NOT include any other text, explanations, or direct answers. If you decide to invoke any function(s), you MUST use the provided tools. Do NOT attempt to answer the question directly without using the available functions.{passing_in_english_prompt}

You should only return the function calls in your response, in JSON format as a list where each element has the format {{"name": "function_name", "arguments": {{param1: value1, param2: value2, ...}}}}.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.'''
