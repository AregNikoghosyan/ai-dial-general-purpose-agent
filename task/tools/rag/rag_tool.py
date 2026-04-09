import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

_SYSTEM_PROMPT = """
You are a precise document assistant. Your task is to answer questions based strictly on the provided context.
- Answer only from the given context; do not add external knowledge.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and accurate.
"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.document_cache = document_cache
        self.model = SentenceTransformer(
            model_name_or_path='all-MiniLM-L6-v2',
            device='cpu',
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Performs semantic (RAG) search on a document to find and answer a specific question. "
            "Use this tool for large documents instead of reading every page with file_content_extraction. "
            "Prefer this tool when the file has more than one page or when looking for a specific topic. "
            "Supports: PDF, TXT, CSV, HTML. "
            "The tool indexes the document on first use (cached per conversation) then retrieves the most "
            "relevant chunks to answer the query."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document.",
                },
                "file_url": {
                    "type": "string",
                    "description": "The URL of the file to search in.",
                },
            },
            "required": ["request", "file_url"],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        args = json.loads(tool_call_params.tool_call.function.arguments)
        request: str = args.get("request")
        file_url: str = args.get("file_url")
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**File URL**: {file_url}\n\r")

        cache_document_key = f"{tool_call_params.conversation_id}:{file_url}"
        cached_data = self.document_cache.get(cache_document_key)

        if cached_data:
            index, chunks = cached_data
        else:
            extractor = DialFileContentExtractor(self.endpoint, tool_call_params.api_key)
            text_content = extractor.extract_text(file_url)

            if not text_content:
                stage.append_content("Error: File content not found.\n\r")
                return "Error: File content not found."

            chunks = self.text_splitter.split_text(text_content)
            embeddings = self.model.encode(chunks)
            index = faiss.IndexFlatL2(384)
            index.add(np.array(embeddings, dtype='float32'))
            self.document_cache.set(cache_document_key, (index, chunks))

        query_embedding = self.model.encode([request]).astype('float32')
        distances, indices = index.search(query_embedding, k=3)
        retrieved_chunks = [chunks[idx] for idx in indices[0] if 0 <= idx < len(chunks)]

        augmented_prompt = self.__augmentation(request, retrieved_chunks)

        stage.append_content("## RAG Request: \n")
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        stage.append_content("## Response: \n")

        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version='2025-01-01-preview',
        )

        stream = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt},
            ],
            stream=True,
            deployment_name=self.deployment_name,
        )

        content = ""
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    stage.append_content(delta.content)
                    content += delta.content

        return content

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        context = "\n\n".join(chunks)
        return (
            f"Based on the following context from the document, please answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {request}"
        )
