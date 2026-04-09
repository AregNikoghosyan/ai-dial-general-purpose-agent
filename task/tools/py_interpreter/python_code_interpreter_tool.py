import base64
import json
from typing import Any, Optional

from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment
from pydantic import StrictStr, AnyUrl

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    ⚠️ Pay attention that this tool will wrap all the work with PyInterpreter MCP Server.
    """

    def __init__(
            self,
            mcp_client: MCPClient,
            mcp_tool_models: list[MCPToolModel],
            tool_name: str,
            dial_endpoint: str,
    ):
        """
        :param tool_name: it must be actual name of tool that executes code. It is 'execute_code'.
            https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L303
        """
        self.dial_endpoint = dial_endpoint
        self.mcp_client = mcp_client
        self._code_execute_tool: Optional[MCPToolModel] = None
        for model in mcp_tool_models:
            if model.name == tool_name:
                self._code_execute_tool = model
                break
        if self._code_execute_tool is None:
            raise ValueError(
                f"Tool '{tool_name}' not found in MCP server. "
                f"Available tools: {[m.name for m in mcp_tool_models]}"
            )

    @classmethod
    async def create(
            cls,
            mcp_url: str,
            tool_name: str,
            dial_endpoint: str,
    ) -> 'PythonCodeInterpreterTool':
        """Async factory method to create PythonCodeInterpreterTool"""
        client = await MCPClient.create(mcp_url)
        tools = await client.get_tools()
        return cls(
            mcp_client=client,
            mcp_tool_models=tools,
            tool_name=tool_name,
            dial_endpoint=dial_endpoint,
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._code_execute_tool.name

    @property
    def description(self) -> str:
        return self._code_execute_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._code_execute_tool.parameters

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        args = json.loads(tool_call_params.tool_call.function.arguments)
        code: str = args.get("code")
        session_id = args.get("session_id")
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        stage.append_content(f"```python\n\r{code}\n\r```\n\r")

        if session_id and session_id != 0:
            stage.append_content(f"**session_id**: {session_id}\n\r")
        else:
            stage.append_content("New session will be created\n\r")

        # Execute code via MCP
        result_str = await self.mcp_client.call_tool(self._code_execute_tool.name, args)

        # Parse the result
        result_json = json.loads(result_str)
        execution_result = _ExecutionResult.model_validate(result_json)

        # Handle generated files — pull from PyInterpreter and upload to DIAL bucket
        if execution_result.files:
            dial_client = Dial(base_url=self.dial_endpoint, api_key=tool_call_params.api_key)
            files_home = dial_client.my_appdata_home()

            uploaded_files_info = []
            for file_ref in execution_result.files:
                file_name = file_ref.name
                mime_type = file_ref.mime_type

                # Fetch the resource from the MCP server
                resource = await self.mcp_client.get_resource(AnyUrl(file_ref.uri))

                # Decode: text types stay as UTF-8 bytes; binary types are base64-decoded
                if mime_type.startswith('text/') or mime_type in ('application/json', 'application/xml'):
                    if isinstance(resource, str):
                        file_bytes = resource.encode('utf-8')
                    else:
                        file_bytes = resource
                else:
                    if isinstance(resource, str):
                        file_bytes = base64.b64decode(resource)
                    else:
                        file_bytes = resource

                # Build upload URL and upload
                upload_url = f"files/{(files_home / file_name).as_posix()}"
                dial_client.files.upload(upload_url, file_bytes, mime_type=mime_type)

                # Add to stage and choice as an attachment
                stage.add_attachment(url=upload_url, type=mime_type, title=file_name)
                tool_call_params.choice.add_attachment(url=upload_url, type=mime_type, title=file_name)

                uploaded_files_info.append({"name": file_name, "url": upload_url, "mime_type": mime_type})

            # Enrich execution result with DIAL URLs
            result_dict = execution_result.model_dump()
            result_dict["uploaded_files"] = uploaded_files_info
            execution_result = _ExecutionResult.model_validate(result_dict)

        # Truncate individual output entries to avoid context overload
        if execution_result.output:
            execution_result.output = [o[:1000] for o in execution_result.output]

        stage.append_content(f"```json\n\r{execution_result.model_dump_json(indent=2)}\n\r```\n\r")
        return execution_result.model_dump_json()
