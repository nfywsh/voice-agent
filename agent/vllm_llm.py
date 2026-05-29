# agent/vllm_llm.py
"""VLLM LLM 适配器 - 解决 VLLM chat_template_kwargs 必须为顶层参数的问题

VLLM 的 chat_template_kwargs 必须是请求体的顶层字段，例如：
    {"model": "...", "messages": [...], "chat_template_kwargs": {"enable_thinking": false}}

如果将其放在 extra_body 中（OpenAI SDK 标准做法），VLLM 会挂起请求。
本模块直接使用 httpx 调用 VLLM API，将 chat_template_kwargs 作为顶层参数注入。

环境变量：
  OPENAI_LLM_BASE_URL: VLLM API 地址 (如 http://nginx_gateway:80/api/v1)
  OPENAI_LLM_API_KEY: API 密钥
  OPENAI_LLM_MODEL: 模型名 (如 Qwen3.6-35B-A3B)
  VLLM_CHAT_TEMPLATE_KWARGS: JSON，{"enable_thinking": false} 等
"""

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any, AsyncIterable

import httpx
from livekit.agents import llm as base_llm
from livekit.agents import _exceptions as agents_exc

if TYPE_CHECKING:
    from monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class VLLMLLM(base_llm.LLM):
    """直接使用 httpx 调用的 LLM，专用于 VLLM 的 chat_template_kwargs 顶层注入。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        chat_template_kwargs: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ):
        super().__init__()
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._chat_template_kwargs = chat_template_kwargs or {}
        self._timeout = httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=5.0)
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "vllm"

    async def aclose(self) -> None:
        await self._client.aclose()

    def chat(
        self,
        *,
        chat_ctx: base_llm.ChatContext,
        tools: list[base_llm.Tool] | None = None,
        conn_options: Any = None,
        parallel_tool_calls: bool = False,
        tool_choice: Any = None,
        response_format: Any = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> "VLLMChatStream":
        from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS

        # 从 extra_kwargs 合并动态的 chat_template_kwargs（如思考模式切换）
        # agent.py llm_node 通过 extra_kwargs={"chat_template_kwargs": {"enable_thinking": True}}
        # 也支持 extra_body.chat_template_kwargs 格式
        merged_chat_tpl = dict(self._chat_template_kwargs)
        if extra_kwargs:
            # 优先取顶层 chat_template_kwargs（VLLM 直接识别）
            if "chat_template_kwargs" in extra_kwargs:
                merged_chat_tpl.update(extra_kwargs["chat_template_kwargs"])
            # 也兼容 extra_body.chat_template_kwargs 格式
            eb = extra_kwargs.get("extra_body", {})
            if isinstance(eb, dict) and "chat_template_kwargs" in eb:
                merged_chat_tpl.update(eb["chat_template_kwargs"])

        return VLLMChatStream(
            llm=self,
            model=self._model,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            chat_template_kwargs=merged_chat_tpl,
            client=self._client,
            api_key=self._api_key,
            base_url=self._base_url,
        )


class VLLMChatStream(base_llm.LLMStream):
    """直接使用 httpx 流式调用 VLLM，chat_template_kwargs 作为顶层参数。"""

    def __init__(
        self,
        llm: VLLMLLM,
        model: str,
        chat_ctx: base_llm.ChatContext,
        tools: list[base_llm.Tool],
        conn_options: Any,
        chat_template_kwargs: dict[str, Any],
        client: httpx.AsyncClient,
        api_key: str,
        base_url: str,
    ):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._chat_template_kwargs = chat_template_kwargs
        self._client = client
        self._api_key = api_key
        self._base_url = base_url
        self._tool_ctx = base_llm.ToolContext(tools)

    async def _run(self) -> None:
        self._oai_stream: Any = None
        retryable = True

        try:
            # Convert chat context to provider format
            chat_ctx, _ = self._chat_ctx.to_provider_format(format="openai")

            # Build tool schemas
            tool_schemas = self._tool_ctx.parse_function_tools("openai", strict=True)

            # Build the request body - chat_template_kwargs at TOP LEVEL (critical for VLLM)
            # VLLM expects: {"chat_template_kwargs": {"enable_thinking": false}}
            # NOT: {"enable_thinking": false} directly spread
            body: dict[str, Any] = {
                "model": self._model,
                "messages": chat_ctx,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": dict(self._chat_template_kwargs),
            }

            if tool_schemas:
                body["tools"] = tool_schemas

            url = f"{self._base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }

            async with self._client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    body_text = await resp.aread()
                    raise agents_exc.APIStatusError(
                        message=f"VLLM API error {resp.status_code}: {body_text.decode()}",
                        status_code=resp.status_code,
                        request_id=None,
                        body=body_text,
                        retryable=False,
                    )

                async for line in resp.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                    else:
                        data_str = line.strip()

                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Parse SSE chunk into ChatChunk
                    chat_chunk = self._parse_sse_chunk(chunk_data)
                    if chat_chunk is not None:
                        retryable = False
                        self._event_ch.send_nowait(chat_chunk)

        except httpx.TimeoutException:
            raise agents_exc.APITimeoutError(retryable=retryable) from None
        except httpx.HTTPStatusError as e:
            raise agents_exc.APIStatusError(
                message=str(e),
                status_code=e.response.status_code,
                request_id=None,
                body=e.response.content,
                retryable=False,
            ) from None
        except Exception as e:
            raise agents_exc.APIConnectionError(retryable=retryable) from e

    def _parse_sse_chunk(self, chunk_data: dict) -> base_llm.ChatChunk | None:
        """Parse VLLM SSE chunk into ChatChunk."""
        try:
            choices = chunk_data.get("choices", [])
            if not choices:
                return None

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Handle content delta
            content = delta.get("content", "") or ""

            # Handle tool calls
            tool_calls = delta.get("tool_calls", [])
            if tool_calls:
                # Process tool calls
                fn_tool_calls = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    fn_tool_calls.append(base_llm.FunctionToolCall(
                        name=func.get("name", ""),
                        arguments=func.get("arguments", ""),
                        call_id=tc.get("id", ""),
                    ))
                if fn_tool_calls:
                    return base_llm.ChatChunk(
                        id=chunk_data.get("id", ""),
                        delta=base_llm.ChoiceDelta(
                            role="assistant",
                            content=content,
                            tool_calls=fn_tool_calls,
                        ),
                    )

            if not content and not tool_calls:
                return None

            return base_llm.ChatChunk(
                id=chunk_data.get("id", ""),
                delta=base_llm.ChoiceDelta(
                    content=content,
                    role="assistant",
                ),
            )
        except Exception:
            return None


def create_llm() -> base_llm.LLM:
    """工厂函数：从环境变量创建 VLLM LLM 实例。

    环境变量：
      OPENAI_LLM_BASE_URL: VLLM API 地址
      OPENAI_LLM_API_KEY: API 密钥
      OPENAI_LLM_MODEL: 模型名
      VLLM_CHAT_TEMPLATE_KWARGS: JSON，关闭思考模式等
    """
    api_key = os.environ.get("OPENAI_LLM_API_KEY", "") or os.environ.get("LLM_API_KEY", "placeholder")
    base_url = os.environ.get("OPENAI_LLM_BASE_URL", "") or os.environ.get("LLM_BASE_URL", "")
    model = os.environ.get("OPENAI_LLM_MODEL", "") or os.environ.get("LLM_MODEL", "Qwen3.6-35B-A3B")
    timeout = float(os.environ.get("LLM_TIMEOUT", "60"))

    chat_template_kwargs: dict[str, Any] = {}
    chat_tpl_raw = os.environ.get("VLLM_CHAT_TEMPLATE_KWARGS", "")
    if chat_tpl_raw:
        try:
            chat_template_kwargs = json.loads(chat_tpl_raw)
        except Exception as e:
            logger.warning(f"[vllm_llm] Failed to parse VLLM_CHAT_TEMPLATE_KWARGS: {e}")

    logger.info(f"[vllm_llm] Creating VLLM LLM: base_url={base_url}, model={model}, chat_template_kwargs={chat_template_kwargs}")

    return VLLMLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        chat_template_kwargs=chat_template_kwargs,
        timeout=timeout,
    )