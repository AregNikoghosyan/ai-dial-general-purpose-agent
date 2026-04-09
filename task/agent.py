import asyncio
import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat.legacy.chat_completion import CustomContent, ToolCall
from aidial_sdk.chat_completion import Message, Role, Choice, Request, Response

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor


class GeneralPurposeAgent:

    def __init__(
            self,
            endpoint: str,
            system_prompt: str,
            tools: list[BaseTool],
    ):
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools
        self._tools_dict: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.state: dict[str, Any] = {TOOL_CALL_HISTORY_KEY: []}

    async def handle_request(self, deployment_name: str, choice: Choice, request: Request, response: Response) -> Message:
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version=request.api_version,
        )

        create_params: dict[str, Any] = dict(
            messages=self._prepare_messages(request.messages),
            deployment_name=deployment_name,
            stream=True,
        )
        if self.tools:
            create_params["tools"] = [tool.schema for tool in self.tools]

        chunks = await client.chat.completions.create(**create_params)

        tool_call_index_map: dict[int, Any] = {}
        content = ""

        async for chunk in chunks:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta:
                    if delta.content:
                        choice.append_content(delta.content)
                        content += delta.content
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            if tc_delta.id:
                                # First chunk of a tool call — register it
                                tool_call_index_map[tc_delta.index] = tc_delta
                            else:
                                # Subsequent chunks — accumulate arguments
                                existing = tool_call_index_map[tc_delta.index]
                                if tc_delta.function:
                                    arg_chunk = tc_delta.function.arguments or ""
                                    existing.function.arguments += arg_chunk

        # Build ToolCall objects from accumulated deltas
        tool_calls: list[ToolCall] = []
        for tc in tool_call_index_map.values():
            tc_dict = tc.model_dump() if hasattr(tc, 'model_dump') else tc.dict()
            tool_calls.append(ToolCall.validate(tc_dict))

        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content or None,
            tool_calls=tool_calls if tool_calls else None,
        )

        if assistant_message.tool_calls:
            conversation_id = (request.headers or {}).get("x-conversation-id", "")
            tasks = [
                self._process_tool_call(
                    tool_call=tc,
                    choice=choice,
                    api_key=request.api_key,
                    conversation_id=conversation_id,
                )
                for tc in assistant_message.tool_calls
            ]
            tool_messages = await asyncio.gather(*tasks)

            self.state[TOOL_CALL_HISTORY_KEY].append(
                assistant_message.dict(exclude_none=True)
            )
            self.state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)

            # Recursive call to continue the conversation after tool execution
            return await self.handle_request(deployment_name, choice, request, response)

        # No tool calls — final answer
        choice.set_state(self.state)
        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        unpacked = unpack_messages(messages, self.state[TOOL_CALL_HISTORY_KEY])
        unpacked.insert(0, {"role": "system", "content": self.system_prompt})
        for msg in unpacked:
            print(json.dumps(msg, default=str))
        return unpacked

    async def _process_tool_call(self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str) -> dict[str, Any]:
        tool_name = tool_call.function.name
        stage = StageProcessor.open_stage(choice, name=tool_name)
        tool = self._tools_dict.get(tool_name)

        if tool is None:
            stage.append_content(f"Error: Unknown tool '{tool_name}'")
            StageProcessor.close_stage_safely(stage)
            return {
                "role": Role.TOOL.value,
                "name": tool_name,
                "tool_call_id": tool_call.id,
                "content": f"Error: Tool '{tool_name}' not found.",
            }

        if tool.show_in_stage:
            stage.append_content("## Request arguments: \n")
            stage.append_content(
                f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r"
            )
            stage.append_content("## Response: \n")

        tool_call_params = ToolCallParams(
            tool_call=tool_call,
            stage=stage,
            choice=choice,
            api_key=api_key,
            conversation_id=conversation_id,
        )
        result_message = await tool.execute(tool_call_params)
        StageProcessor.close_stage_safely(stage)
        return result_message.dict(exclude_none=True)
