"""Conversation support for DeepSeek."""

import dataclasses
import json
from collections.abc import Callable
from typing import Any, Literal

import openai
import probatio
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.json import json_dumps

from . import DeepSeekConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_THINKING,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_THINKING,
    RECOMMENDED_TOP_P,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

# `thinking_content` is only available on newer Home Assistant versions.
_ASSISTANT_CONTENT_FIELDS = frozenset(
    field.name for field in dataclasses.fields(conversation.AssistantContent)
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: DeepSeekConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = DeepSeekConversationEntity(config_entry)
    async_add_entities([agent])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format a Home Assistant tool for the DeepSeek API."""
    schema = probatio.to_openapi(tool.parameters, custom_serializer=custom_serializer)
    tool_spec = FunctionDefinition(name=tool.name, parameters=schema)
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


def _convert_content_to_param(
    content: conversation.Content,
) -> ChatCompletionMessageParam:
    """Convert a chat log content item to a DeepSeek (OpenAI) message."""
    if isinstance(content, conversation.SystemContent):
        return ChatCompletionSystemMessageParam(role="system", content=content.content)
    if isinstance(content, conversation.UserContent):
        return ChatCompletionUserMessageParam(role="user", content=content.content)
    if isinstance(content, conversation.ToolResultContent):
        # Home Assistant renamed `tool_result` (raw dict) to `result`
        # (a llm.ToolResult) in newer versions. Support both.
        result = getattr(content, "result", None)
        data = result.data if result is not None else content.tool_result
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json_dumps(data),
        )
    if isinstance(content, conversation.AssistantContent):
        param = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=content.content,
        )
        # DeepSeek thinking mode requires the reasoning_content of previous
        # turns to be passed back whenever tools are used, otherwise the API
        # returns a 400 error.
        thinking_content = getattr(content, "thinking_content", None)
        if thinking_content:
            param["reasoning_content"] = thinking_content  # type: ignore[typeddict-unknown-key]
        if content.tool_calls:
            param["tool_calls"] = [
                ChatCompletionMessageToolCallParam(
                    id=tool_call.id,
                    type="function",
                    function=Function(
                        name=tool_call.tool_name,
                        arguments=json.dumps(tool_call.tool_args),
                    ),
                )
                for tool_call in content.tool_calls
            ]
        return param
    raise ValueError(f"Unexpected content type: {type(content)}")


class DeepSeekConversationEntity(conversation.ConversationEntity):
    """DeepSeek conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: DeepSeekConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="DeepSeek",
            model=entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.entry.options

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_handle_chat_log(
        self, chat_log: conversation.ChatLog
    ) -> None:
        """Generate a response for the chat log."""
        options = self.entry.options
        client = self.entry.runtime_data
        thinking = options.get(CONF_THINKING, RECOMMENDED_THINKING)

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            messages = [
                _convert_content_to_param(content) for content in chat_log.content
            ]

            tools: list[ChatCompletionToolParam] | None = None
            if chat_log.llm_api:
                tools = [
                    _format_tool(tool, chat_log.llm_api.custom_serializer)
                    for tool in chat_log.llm_api.tools
                ]

            try:
                result = await client.chat.completions.create(
                    model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                    messages=messages,
                    tools=tools or NOT_GIVEN,
                    max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                    temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                    user=chat_log.conversation_id,
                    stream=False,
                    extra_body={
                        "thinking": {
                            "type": "enabled" if thinking else "disabled"
                        }
                    },
                )
            except openai.OpenAIError as err:
                LOGGER.error("DeepSeek API调用错误: %s", err)
                raise HomeAssistantError("DeepSeek服务调用失败") from err

            response = result.choices[0].message

            tool_calls: list[llm.ToolInput] | None = None
            if response.tool_calls:
                tool_calls = [
                    llm.ToolInput(
                        id=tool_call.id,
                        tool_name=tool_call.function.name,
                        tool_args=json.loads(tool_call.function.arguments or "{}"),
                    )
                    for tool_call in response.tool_calls
                ]

            content_kwargs: dict[str, Any] = {
                "agent_id": self.entity_id,
                "content": response.content,
                "tool_calls": tool_calls,
            }
            if "thinking_content" in _ASSISTANT_CONTENT_FIELDS:
                content_kwargs["thinking_content"] = getattr(
                    response, "reasoning_content", None
                )
            assistant_content = conversation.AssistantContent(**content_kwargs)

            # Adds the assistant content to the chat log and executes any tool
            # calls, yielding the tool results as they complete.
            async for _tool_result in chat_log.async_add_assistant_content(
                assistant_content
            ):
                pass

            if not assistant_content.tool_calls:
                break

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)
