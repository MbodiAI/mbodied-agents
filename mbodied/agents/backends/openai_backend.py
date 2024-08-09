# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
from typing import Any, List

import backoff
import httpx
from anthropic import RateLimitError as AnthropicRateLimitError
from openai._exceptions import RateLimitError as OpenAIRateLimitError

from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image
from mbodied.agents.backends.backend import Backend
ERRORS = (
    OpenAIRateLimitError,
    AnthropicRateLimitError,
    httpx.HTTPError,
    ConnectionError,
)


class OpenAISerializer(Serializer):
    """Serializer for OpenAI-specific data formats."""

    @classmethod
    def serialize_image(cls, image: Image) -> dict[str, Any]:
        """Serializes an image to the OpenAI format.

        Args:
            image: The image to be serialized.

        Returns:
            A dictionary representing the serialized image.
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": image.url,
            },
        }

    @classmethod
    def serialize_text(cls, text: str) -> dict[str, Any]:
        """Serializes a text string to the OpenAI format.

        Args:
            text: The text to be serialized.

        Returns:
            A dictionary representing the serialized text.
        """
        return {"type": "text", "text": text}


class OpenAIBackendMixin(Backend):
    """Backend for interacting with OpenAI's API.

    Attributes:
        api_key: The API key for the OpenAI service.
        client: The client for the OpenAI service.
        serialized: The serializer for the OpenAI backend.
        response_format: The format for the response.
    """

    INITIAL_CONTEXT = [
        Message(role="system", content="You are a robot with advanced spatial reasoning."),
    ]
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str | None = None, client: Any | None = None, response_format: str = None, **kwargs):
        """Initializes the OpenAIBackend with the given API key and client.

        Args:
            api_key: The API key for the OpenAI service.
            client: An optional client for the OpenAI service.
            response_format: The format for the response.
            **kwargs: Additional keyword arguments.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("MBODI_API_KEY")
        self.client = client
        if self.client is None:
            from openai import OpenAI

            kwargs.pop("model_src", None)
            self.client = OpenAI(api_key=self.api_key or "any_key", **kwargs)
        self.serialized = OpenAISerializer
        self.response_format = response_format


    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_tries=3,
        on_backoff=lambda details: print(f"Backing off {details['wait']:.1f} seconds after {details['tries']} tries."),  # noqa
    )
    def predict(
        self, message: Message, context: List[Message] | None = None, model: Any | None = None, **kwargs
    ) -> str:
        """Create a completion based on the given message and context.

        Args:
            message (Message): The message to process.
            context (Optional[List[Message]]): The context of messages.
            model (Optional[Any]): The model used for processing the messages.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the completion.
        """
        context = context or self.INITIAL_CONTEXT
        model = kwargs.pop("model", model) or self.DEFAULT_MODEL
        serialized_messages = [self.serialized(msg).serialize() for msg in context + [message]]

        completion = self.client.chat.completions.create(
            model=model,
            messages=serialized_messages,
            temperature=0,
            max_tokens=1000,
            **kwargs,
        )
        return completion.choices[0].message.content

    
    def stream(
        self, message: Message, context: List[Message] =  None, model: str = "gpt-4o", **kwargs
    ) -> str:
        """Streams a completion for the given messages using the OpenAI API standard.

        Args:
            messages: A list of messages to be sent to the completion API.
            model: The model to be used for the completion.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The content of the completion response.
        """
        model = kwargs.pop("model", model) or self.DEFAULT_MODEL
        context = context or self.INITIAL_CONTEXT
        serialized_messages = [self.serialized(msg).serialize() for msg in context + [message]]
        with self.client.with_streaming_response.chat.completions.create(
            messages=serialized_messages,
            model=model,
            temperature=0,
            stream=True,
            **kwargs,
        ) as stream:
            for chunk in stream.iter_text():
                chunk = chunk.lstrip("data:")
                if "DONE" in chunk:
                    return
                chunk = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                yield chunk

def main():
    backend = OpenAIBackendMixin()
    message = Message(role="user", content="What is the capital of France?")
    resp = ""
    for chunk in backend.stream(message, model="astro-world"):
        resp += chunk
        print(resp)

if __name__ == "__main__":
    main()

