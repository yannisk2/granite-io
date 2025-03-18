# SPDX-License-Identifier: Apache-2.0

"""
Common shared types
"""

# Standard
from typing import Any, List, Optional, Union

# Third Party
from typing_extensions import Literal, TypeAlias
import httpx
import pydantic


class FunctionCall(pydantic.BaseModel):
    id: str
    name: str

    # This field should adhere to the argument schema from the  associated
    # FunctionDefinition in the generation request that produced it.
    arguments: dict[str, Any] | None


class Hallucination(pydantic.BaseModel):
    """Hallucination data as returned by the model output parser"""

    hallucination_id: str
    risk: str
    response_text: str
    response_begin: int
    response_end: int


class Citation(pydantic.BaseModel):
    """Citation data as returned by the model output parser"""

    citation_id: str
    doc_id: str
    context_text: str
    context_begin: int
    context_end: int
    response_text: str
    response_begin: int
    response_end: int


class Document(pydantic.BaseModel):
    """Document data as returned by the model output parser"""

    doc_id: str
    text: str


class _ChatMessageBase(pydantic.BaseModel):
    """Base class for all message types.

    Due to the vaguaries of Pydantic's JSON parser, we use this class only for common
    functionality, and NOT for defining a common dataclass base type. Use the
    :class:`ChatMessage` type alias to annotate a field or argument as accepting all
    subclasses of this one."""

    content: str
    """Every message has raw string content, even if it also contains parsed structured
    content such as a JSON record."""

    def to_openai_json(self):
        result = {"role": self.role, "content": self.content}
        return result


class UserMessage(_ChatMessageBase):
    role: Literal["user"] = "user"


class AssistantMessage(_ChatMessageBase):
    role: Literal["assistant"] = "assistant"
    tool_calls: list[FunctionCall] = []
    reasoning_content: str | None = None
    # Raw response content without any parsing for re-serialization
    _raw: str | None = None
    citations: list[Citation] | None = None
    documents: list[Document] | None = None
    hallucinations: list[Hallucination] | None = None
    stop_reason: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._raw = kwargs.pop("raw", None)

    @property
    def raw(self) -> str:
        """Get the raw content of the response"""
        return self._raw if self._raw is not None else self.content


class ToolResultMessage(_ChatMessageBase):
    role: Literal["tool"] = "tool"
    tool_call_id: str


class SystemMessage(_ChatMessageBase):
    role: Literal["system"] = "system"


ChatMessage: TypeAlias = (
    UserMessage | AssistantMessage | ToolResultMessage | SystemMessage
)
"""Type alias for all message types. We use this Union instead of the actual base class
:class:`_ChatMessageBase` so that Pydantic can parse the message list from JSON."""


class FunctionDefinition(pydantic.BaseModel):
    name: str
    description: str | None = None

    # This field holds a JSON schema for a record, but the `jsonschema` package doesn't
    # define an object type for such a schema, instead using a dictionary.
    parameters: dict[str, Any] | None = None

    def to_openai_json(self) -> dict:
        """
        :returns: JSON representation of this function, as the Python equivalent of the
            standard JSON used in the OpenAI chat completions API for the `tools`
            argument.
        """
        raise NotImplementedError("TODO: Implement this")


class GenerateInputs(pydantic.BaseModel):
    """Common inputs for backends"""

    prompt: Optional[Union[str, List[Union[str, List[Union[str, List[int]]]]]]] = None
    model: Optional[str] = None
    best_of: Optional[int] = None
    timeout: Optional[Union[float, str, httpx.Timeout]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stream_options: Optional[dict] = None
    stop: Union[Optional[str], List[str], None] = None
    # max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    # modalities: Optional[List[ChatCompletionModality]] = None
    # prediction: Optional[ChatCompletionPredictionContentParam] = None
    # audio: Optional[ChatCompletionAudioParam] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    user: Optional[str] = None
    # reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    # response_format: Optional[Union[dict, Type[BaseModel]]] = None
    seed: Optional[int] = None
    # tools: Optional[List] = None
    # tool_choice: Optional[Union[str, dict]] = None
    # ??? bool or int?  logprobs: Optional[bool] = None
    # top_logprobs: Optional[int] = None
    # parallel_tool_calls: Optional[bool] = None
    # deployment_id=None
    extra_headers: Optional[dict] = None
    # soon to be deprecated params by OpenAI
    # functions: Optional[List] = None
    # function_call: Optional[str] = None
    # # set api_base, api_version, api_key
    # base_url: Optional[str] = None
    # api_version: Optional[str] = None
    # api_key: Optional[str] = None
    # model_list: Optional[list] = None  # pass in a list of api_base,keys, etc.
    # thinking: Optional[AnthropicThinkingParam] = None
    # **kwargs

    model_config = pydantic.ConfigDict(
        # Pass through arbitrary additional keyword arguments for handling by model- or
        # specific I/O processors.
        arbitrary_types_allowed=True,
        extra="allow",
    )


class ChatCompletionInputs(pydantic.BaseModel):
    """
    Class that represents the lowest-common-denominator inputs to a chat completion
    call.  Individual input/output processors can extend this schema with their own
    additional proprietary fields.
    """

    messages: list[ChatMessage]
    tools: list[FunctionDefinition] = []
    generate_inputs: Optional[GenerateInputs] = None

    num_return_sequences: int = 1

    model_config = pydantic.ConfigDict(
        # Pass through arbitrary additional keyword arguments for handling by model- or
        # specific I/O processors.
        extra="allow"
    )

    def __getattr__(self, name: str) -> any:
        """Allow attribute access for unknown attributes"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return None

    def with_next_message(self, next_message: ChatMessage) -> "ChatCompletionInputs":
        """Create a version of this object with one additional message in the messages
        list. Does not modify the original object."""
        new_messages = self.messages.copy()
        new_messages.append(next_message)
        return self.model_copy(update={"messages": new_messages})


class ChatCompletionResult(pydantic.BaseModel):
    """
    Class that represents the lowest-common-denominator outputs of a chat completion
    call.  Individual input/output processors can extend this schema with their own
    additional proprietary fields.
    """

    next_message: ChatMessage

    def __getattr__(self, name: str) -> any:
        """Allow attribute access for unknown attributes"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return None


class ChatCompletionResults(pydantic.BaseModel):
    results: list[ChatCompletionResult]


class GenerateResult(pydantic.BaseModel):
    """
    All the things that our internal :func:`generate()` methods return,
    rolled into a dataclass for ease of maintenance.
    """

    # Not including input characters
    completion_string: str

    # Not including input tokens
    completion_tokens: list[int]

    stop_reason: str


class GenerateResults(pydantic.BaseModel):
    results: list[GenerateResult]
