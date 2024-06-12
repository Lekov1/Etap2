from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.messages import AIMessage
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor
from langchain_community.adapters.openai import convert_message_to_dict
import together
from transformers import AutoTokenizer

from .prompt_generators import *


CUSTOM_TEMPLATES = {
    "mistralai/Mistral-7B-Instruct-v0.1": build_mistral_prompt_template,
}


class ChatTogether(BaseChatModel):
    model_name: str = Field(default="mistralai/Mistral-7B-Instruct-v0.1", alias="model")
    tokenizer: Optional[Any] = None
    max_retries: int = 2
    temperature: float = 0.1
    top_p: float = 0.7
    top_k: int = 50
    logprobs: Optional[int] = None
    repetition_penalty: float = 1
    max_tokens: int = 500
    n: int = 1
    stop: list = None
    streaming: bool = False
    together_api_key: Optional[str] = Field(default=None, alias="api_key")

    class Config:
        extra = Extra.forbid

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"together_api_key": "TOGETHER_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain", "chat_models", "together"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _default_params(self) -> Dict[str, Any]:
        return dict(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            logprobs=self.logprobs,
            repetition_penalty=self.repetition_penalty,
            stop=self.stop,
        )

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined

    def _get_custom_prompt_generator(self):
        template_fn = CUSTOM_TEMPLATES.get(self.model_name)
        if template_fn is None:
            return None

        return template_fn

    def _create_message_dicts(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts

    def _create_prompt_string(self, messages, tokenize=False):
        prompt_generator = self._get_custom_prompt_generator()
        if prompt_generator is not None:
            prompt_string = prompt_generator(messages)
            if not tokenize:
                return prompt_string
            return self._tokenizer.encode(prompt_string)

        message_dicts = self._create_message_dicts(messages)
        return self.tokenizer.apply_chat_template(message_dicts, tokenize=tokenize)

    def _create_chat_result(self, response: dict) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = AIMessage(content=res["text"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        prompt_string = self._create_prompt_string(messages)
        response = together.Complete.create(prompt_string, **self._default_params)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        prompt_string = self._create_prompt_string(messages)
        response = await together.AsyncComplete.create(prompt_string, **self._default_params)
        return self._create_chat_result(response['output'])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt_string = self._create_prompt_string(messages)
        output = together.Complete.create_streaming(prompt_string, **self._default_params)
        for token in output:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        result = await run_in_executor(
            None,
            self._stream,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        for chunk in result:
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "together-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return dict(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop=self.stop or [],
        )

    def get_token_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        tokens = self._create_prompt_string(messages, tokenize=True)
        return len(tokens)
