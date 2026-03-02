# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)


class Qwen3CoderToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Qwen3 XML Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

        logger.info(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        # Store accumulated parameters for type conversion
        self.accumulated_params = {}
        self.streaming_request = None
        # Incremental string streaming state (ported from GLM4 parser)
        self._streaming_string_value = False
        self._value_buffer = ""

    @staticmethod
    def _json_escape_string_content(s: str) -> str:
        """JSON-escape string content for incremental streaming.
        """
        if not s:
            return ""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    def _is_string_type(self, param_name: str) -> bool:
        """Check if a parameter is string type based on tool schema."""
        if not self.streaming_request or not self.streaming_request.tools:
            return True  # Default to string for unknown params
        param_config = self._get_arguments_config(
            self.current_function_name or "",
            self.streaming_request.tools,
        )
        if param_name not in param_config:
            return True  # Default to string for unknown params
        param_type = (
            str(param_config[param_name].get("type", "string")).strip().lower()
            if isinstance(param_config.get(param_name), dict)
            else "string"
        )
        return param_type in [
            "string", "str", "text", "varchar", "char", "enum",
        ]

    def _get_arguments_config(
        self, func_name: str, tools: list[ChatCompletionToolsParam] | None
    ) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            if not hasattr(config, "type") or not (
                hasattr(config, "function") and hasattr(config.function, "name")
            ):
                continue
            if config.type == "function" and config.function.name == func_name:
                if not hasattr(config.function, "parameters"):
                    return {}
                params = config.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.debug("Tool '%s' is not defined in the tools list.", func_name)
        return {}

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.debug(
                    "Parsed parameter '%s' is not defined in the tool "
                    "parameters for tool '%s', directly returning the "
                    "string value.",
                    param_name,
                    func_name,
                )
            return param_value

        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not an "
                    "integer in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                return (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a float "
                    "in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a boolean "
                    "(`true` or `false`) in tool '%s', degenerating to "
                    "false.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.debug(
                        "Parsed value '%s' of parameter '%s' cannot be "
                        "parsed with json.loads in tool '%s', will try "
                        "other methods to parse it.",
                        param_value,
                        param_name,
                        func_name,
                    )
            try:
                param_value = ast.literal_eval(param_value)  # safer
            except (ValueError, SyntaxError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' cannot be "
                    "converted via Python `ast.literal_eval()` in tool "
                    "'%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[ChatCompletionToolsParam] | None
    ) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)
            ),
        )

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [
            match[0] if match[0] else match[1] for match in raw_function_calls
        ]
        return function_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Quick check to avoid unnecessary processing
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = [
                self._parse_xml_function_call(function_call_str, request.tools)
                for function_call_str in function_calls
            ]

            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    )

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            content = model_output[:content_index]  # .rstrip()

            return ExtractedToolCallInformation(
                tools_called=(len(tool_calls) > 0),
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Store request for type conversion
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, return None unless it's an EOS token after tools
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            # Check for tool calls in text even if is_tool_call_started
            # is False (might have been reset after processing all tools)
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                # If we have completed tool calls and populated
                # prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(
                        self.tool_call_start_token
                    ) - current_text.count(self.tool_call_end_token)
                    if open_calls == 0:
                        # Return empty delta for finish_reason processing
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        # Update accumulated text
        self.accumulated_text = current_text

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # Check if this tool call has ended
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                # This tool has ended, advance to next
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}

                # Check if there are more tool calls
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    # No more tool calls
                    self.is_tool_call_started = False
                # Continue processing next tool
                return None

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            # Check if tool call is starting
            if (
                self.tool_call_start_token_id in delta_token_ids
                or self.tool_call_start_token in delta_text
            ):
                self.is_tool_call_started = True
                # Return any content before the tool call
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[
                        : delta_text.index(self.tool_call_start_token)
                    ]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                # Check if we're between tool calls - skip whitespace
                if (
                    current_text.rstrip().endswith(self.tool_call_end_token)
                    and delta_text.strip() == ""
                ):
                    # We just ended a tool call, skip whitespace
                    return None
                # Normal content, no tool call
                return DeltaMessage(content=delta_text)

        # Check if we're between tool calls (waiting for next one)
        # Count tool calls we've seen vs processed
        tool_starts_count = current_text.count(self.tool_call_start_token)
        if self.current_tool_index >= tool_starts_count:
            # We're past all tool calls, shouldn't be here
            return None

        # We're in a tool call, find the current tool call portion
        # Need to find the correct tool call based on current_tool_index
        tool_start_positions: list[int] = []
        idx = 0
        while True:
            idx = current_text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            tool_start_positions.append(idx)
            idx += len(self.tool_call_start_token)

        if self.current_tool_index >= len(tool_start_positions):
            # No more tool calls to process yet
            return None

        tool_start_idx = tool_start_positions[self.current_tool_index]
        # Find where this tool call ends (or current position if not ended yet)
        tool_end_idx = current_text.find(self.tool_call_end_token, tool_start_idx)
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx : tool_end_idx + len(self.tool_call_end_token)
            ]

        # Looking for function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    # Found complete function name
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    # IMPORTANT: Add to prev_tool_call_arr immediately when
                    # we detect a tool call. This ensures
                    # finish_reason="tool_calls" even if parsing isn't complete
                    already_added = any(
                        tool.get("name") == self.current_function_name
                        for tool in self.prev_tool_call_arr
                    )
                    if not already_added:
                        self.prev_tool_call_arr.append(
                            {
                                "name": self.current_function_name,
                                "arguments": "{}",  # Placeholder, will be updated later
                            }
                        )

                    # Send header with function info
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                id=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    name=self.current_function_name, arguments=""
                                ),
                                type="function",
                            )
                        ]
                    )
            return None

        # We've sent header, now handle function body
        if self.in_function:
            # Send opening brace if not sent yet
            if not self.json_started and self.parameter_prefix not in delta_text:
                self.json_started = True
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="{"),
                        )
                    ]
                )

            # Make sure json_started is set if we're processing parameters
            if not self.json_started:
                self.json_started = True

            # Check for function end in accumulated text
            if not self.json_closed and self.function_end_token in tool_text:
                # Close JSON
                self.json_closed = True

                # Extract complete tool call to update
                # prev_tool_call_arr with final arguments
                # Find the function content
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = tool_text.find(self.function_end_token, func_start)
                if func_content_end != -1:
                    func_content = tool_text[func_start:func_content_end]
                    # Parse to get the complete arguments
                    try:
                        parsed_tool = self._parse_xml_function_call(
                            func_content,
                            self.streaming_request.tools
                            if self.streaming_request
                            else None,
                        )
                        if parsed_tool:
                            # Update existing entry in
                            # prev_tool_call_arr with complete args
                            for i, tool in enumerate(self.prev_tool_call_arr):
                                if tool.get("name") == parsed_tool.function.name:
                                    args = parsed_tool.function.arguments
                                    self.prev_tool_call_arr[i]["arguments"] = args
                                    break
                    except Exception:
                        pass  # Ignore parsing errors during streaming

                result = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="}"),
                        )
                    ]
                )

                # Reset state for next tool
                self.in_function = False
                self.json_closed = True
                self.accumulated_params = {}

                return result

            # -------------------------------------------------------
            # Handle incremental string value streaming
            # -------------------------------------------------------
            if self._streaming_string_value:
                # We're in the middle of streaming a string parameter
                # value incrementally. Extract the current value from
                # the accumulated tool_text.
                param_tag = f"{self.parameter_prefix}{self.current_param_name}>"
                param_tag_pos = tool_text.find(param_tag)
                if param_tag_pos == -1:
                    return None

                value_start_pos = param_tag_pos + len(param_tag)
                value_text = tool_text[value_start_pos:]
                # Strip leading newline (Qwen3 format puts \n after >)
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                # Check if the parameter value is complete
                val_end = value_text.find(self.parameter_end_token)
                if val_end != -1:
                    # Parameter complete - emit remaining content and
                    # close the JSON string quote
                    remaining_value = value_text[:val_end]
                    if remaining_value.endswith("\n"):
                        remaining_value = remaining_value[:-1]
                    new_content = remaining_value[len(self._value_buffer):]
                    self._value_buffer = ""
                    self._streaming_string_value = False
                    self.param_count += 1

                    escaped = self._json_escape_string_content(new_content)
                    frag = escaped + '"'
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                function=DeltaFunctionCall(arguments=frag),
                            )
                        ]
                    )
                else:
                    # Parameter still streaming - emit safe content
                    # Check for partial </parameter> at end of buffer
                    current_value = value_text
                    if current_value.endswith("\n"):
                        current_value = current_value[:-1]

                    # Find content safe to emit (not part of closing tag)
                    safe_len = len(current_value)
                    for i in range(1, len(self.parameter_end_token)):
                        if current_value.endswith(
                            self.parameter_end_token[:i]
                        ):
                            safe_len = len(current_value) - i
                            break

                    safe_value = current_value[:safe_len]
                    new_content = safe_value[len(self._value_buffer):]
                    if new_content:
                        self._value_buffer = safe_value
                        escaped = self._json_escape_string_content(
                            new_content
                        )
                        if escaped:
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_index,
                                        function=DeltaFunctionCall(
                                            arguments=escaped
                                        ),
                                    )
                                ]
                            )
                    return None

            # -------------------------------------------------------
            # Look for parameters
            # -------------------------------------------------------
            param_starts = []
            idx = 0
            while True:
                idx = tool_text.find(self.parameter_prefix, idx)
                if idx == -1:
                    break
                param_starts.append(idx)
                idx += len(self.parameter_prefix)

            # Check if we should start a new parameter
            if (
                not self.in_param
                and len(param_starts) > self.param_count
            ):
                # Process the next parameter
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" in remaining:
                    # We have the complete parameter name
                    name_end = remaining.find(">")
                    self.current_param_name = remaining[:name_end]

                    # Check if this is a string type parameter
                    is_string = self._is_string_type(self.current_param_name)

                    # Find the parameter value start
                    value_start = param_start + name_end + 1
                    value_text = tool_text[value_start:]
                    if value_text.startswith("\n"):
                        value_text = value_text[1:]

                    if is_string:
                        # -----------------------------------------
                        # String type: use incremental streaming
                        # -----------------------------------------
                        # Emit the key and opening quote immediately
                        if self.param_count == 0:
                            key_frag = f'"{self.current_param_name}": "'
                        else:
                            key_frag = f', "{self.current_param_name}": "'

                        self._streaming_string_value = True
                        self._value_buffer = ""

                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_index,
                                    function=DeltaFunctionCall(
                                        arguments=key_frag
                                    ),
                                )
                            ]
                        )
                    else:
                        # -----------------------------------------
                        # Non-string type: wait for complete value
                        # -----------------------------------------
                        param_end_idx = value_text.find(
                            self.parameter_end_token
                        )
                        if param_end_idx == -1:
                            # No closing tag yet, look for boundaries
                            next_param_idx = value_text.find(
                                self.parameter_prefix
                            )
                            func_end_idx = value_text.find(
                                self.function_end_token
                            )

                            if next_param_idx != -1 and (
                                func_end_idx == -1
                                or next_param_idx < func_end_idx
                            ):
                                param_end_idx = next_param_idx
                            elif func_end_idx != -1:
                                param_end_idx = func_end_idx
                            else:
                                if self.tool_call_end_token in tool_text:
                                    param_end_idx = len(value_text)
                                else:
                                    return None

                        if param_end_idx != -1:
                            param_value = value_text[:param_end_idx]
                            if param_value.endswith("\n"):
                                param_value = param_value[:-1]

                            self.accumulated_params[
                                self.current_param_name
                            ] = param_value

                            param_config = self._get_arguments_config(
                                self.current_function_name or "",
                                self.streaming_request.tools
                                if self.streaming_request
                                else None,
                            )

                            converted_value = self._convert_param_value(
                                param_value,
                                self.current_param_name,
                                param_config,
                                self.current_function_name or "",
                            )

                            serialized_value = json.dumps(
                                converted_value, ensure_ascii=False
                            )

                            if self.param_count == 0:
                                json_fragment = (
                                    f'"{self.current_param_name}":'
                                    f" {serialized_value}"
                                )
                            else:
                                json_fragment = (
                                    f', "{self.current_param_name}":'
                                    f" {serialized_value}"
                                )

                            self.param_count += 1

                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_index,
                                        function=DeltaFunctionCall(
                                            arguments=json_fragment
                                        ),
                                    )
                                ]
                            )

        return None
