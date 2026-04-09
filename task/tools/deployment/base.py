import json
from abc import ABC, abstractmethod
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return {}

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        args = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = args.get("prompt")
        del args["prompt"]

        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version="2025-01-01-preview",
        )

        messages = [{"role": "user", "content": prompt}]

        # Pass remaining args (size, quality, style, etc.) as extra_body
        extra_body: dict[str, Any] = dict(args) if args else {}

        stage = tool_call_params.stage

        print(f"[DeploymentTool] Calling deployment={self.deployment_name}, prompt={prompt!r}, extra_body={extra_body}")
        response = await client.chat.completions.create(
            messages=messages,
            stream=False,
            deployment_name=self.deployment_name,
            extra_body=extra_body,
            **self.tool_parameters,
        )
        print(f"[DeploymentTool] Raw response: {response}")

        content = ""
        attachments = []

        if response.choices:
            msg = response.choices[0].message
            print(f"[DeploymentTool] message.content={msg.content!r}")
            print(f"[DeploymentTool] message.custom_content={msg.custom_content}")
            if msg.content:
                content = msg.content
                stage.append_content(content)
            if msg.custom_content and msg.custom_content.attachments:
                for att in msg.custom_content.attachments:
                    print(f"[DeploymentTool] attachment: type={att.type}, url={att.url}, title={att.title}")
                    attachments.append(att)
                    if att.url:
                        stage.add_attachment(
                            url=att.url,
                            type=att.type,
                            title=att.title,
                        )

        custom_content = CustomContent(attachments=attachments) if attachments else None

        return Message(
            role=Role.TOOL,
            content=StrictStr(content) if content else None,
            custom_content=custom_content,
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
        )
