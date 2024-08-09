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

"""Run a LanguageAgent with memory, optional remote acting, and optional automatic dataset creation capabilities.

While it is always recommended to explicitly define your observation and action spaces,
which can be set with a gym.Space object or any python object using the Sample class
(see examples/using_sample.py for a tutorial), you can have the recorder infer the spaces
by setting recorder="default" for automatic dataset recording.

For example:
    >>> agent = LanguageAgent(context=SYSTEM_PROMPT, model_src=backend, recorder="default")
    >>> agent.act_and_record("pick up the fork", image)

Alternatively, you can define the recorder separately to record the space you want.
For example, to record the dataset with the image and instruction observation and AnswerAndActionsList as action:
    >>> observation_space = spaces.Dict({"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)})
    >>> action_space = AnswerAndActionsList(actions=[HandControl()] * 6).space()
    >>> recorder = Recorder(
    ...     'example_recorder',
    ...     out_dir='saved_datasets',
    ...     observation_space=observation_space,
    ...     action_space=action_space

To record:
    >>> recorder.record(
    ...     observation={
    ...         "image": image,
    ...         "instruction": instruction,
    ...     },
    ...     action=answer_actions,
    ... )
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import AsyncGenerator, Generator, List, Literal, TypeAlias

from art import text2art
from pydantic import AnyUrl, DirectoryPath, FilePath, NewPath

from mbodied.agents import Agent
from mbodied.agents.backends import OpenAIBackend
from mbodied.types.message import Message
from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image

SupportsOpenAI: TypeAlias = OpenAIBackend


@dataclass
class Reminder:
    """A reminder to show the agent a prompt every n messages."""

    prompt: str | Image | Message
    n: int

    def __iter__(self):
        yield self.prompt
        yield self.n

    def __getitem__(self, key):
        if key == 0:
            return self.prompt
        elif key == 1:
            return self.n
        else:
            raise IndexError("Invalid index")



def make_context_list(context: list[str | Image | Message] | Image | str | Message | None) -> List[Message]:
    """Convert the context to a list of messages."""
    if isinstance(context, list):
        return [Message(content=c) if not isinstance(c, Message) else c for c in context]
    if isinstance(context, Message):
        return [context]
    if isinstance(context, str | Image):
        return [Message(role="user", content=[context]), Message(role="assistant", content="Understood.")]
    return []


class LanguageAgent(Agent):
    """An agent that can interact with users using natural language.

    This class extends the functionality of a base Agent to handle natural language interactions. It manages memory, dataset-recording, and asynchronous remote inference, supporting multiple platforms including OpenAI, Anthropic, and Gradio.

    Attributes:
        reminders (List[Reminder]): A list of reminders that prompt the agent every n messages.
        context (List[Message]): The current context of the conversation.
        Inherits all attributes from the parent class `Agent`.

    Examples:
        Basic usage with OpenAI:
            >>> cognitive_agent = LanguageAgent(api_key="...", model_src="openai", recorder="default")
            >>> cognitive_agent.act("your instruction", image)

        Automatically act and record to dataset:
            >>> cognitive_agent.act_and_record("your instruction", image)
    """

    _art_printed = False

    def __init__(
        self,
        model_src: Literal["openai", "anthropic", "gradio", "ollama", "http"]
        | AnyUrl
        | FilePath
        | DirectoryPath
        | NewPath = "openai",
        context: list | Image | str | Message = None,
        api_key: str | None = os.getenv("OPENAI_API_KEY"),
        model_kwargs: dict = None,
        recorder: Literal["default", "omit"] | str = "omit",
        recorder_kwargs: dict = None,
    ) -> None:
        """Agent with memory,  asynchronous remote acting, and automatic dataset recording.

         Additionally supports asynchronous remote inference,
            supporting multiple platforms including OpenAI, Anthropic, vLLM, Gradio, and Ollama.

        Args:
            model_src: The source of the model to use for inference. It can be one of the following:
                - "openai": Use the OpenAI backend (or vLLM).
                - "anthropic": Use the Anthropic backend.
                - "gradio": Use the Gradio backend.
                - "ollama": Use the Ollama backend.
                - "http": Use a custom HTTP API backend.
                - AnyUrl: A URL pointing to the model source.
                - FilePath: A local path to the model's weights.
                - DirectoryPath: A local directory containing the model's weights.
                - NewPath: A new path object representing the model source.
            context (Union[list, Image, str, Message], optional): The starting context to use for the conversation.
                    It can be a list of messages, an image, a string, or a message.
                    If a string is provided, it will be interpreted as a user message. Defaults to None.
            api_key (str, optional): The API key to use for the remote actor (if applicable).
                 Defaults to the value of the OPENAI_API_KEY environment variable.
            model_kwargs (dict, optional): Additional keyword arguments to pass to the model source.
                See the documentation of the specific backend for more details. Defaults to None.
            recorder (Union[str, Literal["default", "omit"]], optional):
                The recorder configuration or name or action. Defaults to "omit".
            recorder_kwargs (dict, optional): Additional keyword arguments to pass to the recorder. Defaults to None.
        """
        if not LanguageAgent._art_printed:
            print("Welcome to")  # noqa: T201
            print(text2art("mbodi"))  # noqa: T201
            print("A platform for intelligent embodied agents.\n\n")  # noqa: T201
            LanguageAgent._art_printed = True
        self.reminders: List[Reminder] = []
        print(f"Initializing language agent for robot using : {model_src}")  # noqa: T201

        super().__init__(
            recorder=recorder,
            recorder_kwargs=recorder_kwargs,
            model_src=model_src,
            model_kwargs=model_kwargs,
            api_key=api_key,
        )

        self.context = make_context_list(context)

    def forget_last(self) -> Message:
        """Forget the last message in the context."""
        try:
            return self.context.pop(-1)
        except IndexError:
            logging.warning("No message to forget in the context")

    def forget_after(self, first_n: int) -> None:
        """Forget after the first n messages in the context.

        Args:
            first_n: The number of messages to keep.
        """
        self.context = self.context[:first_n]

    def forget(self, everything=False, last_n: int = -1) -> List[Message]:
        """Forget the last n messages in the context.

        Args:
            everything: Whether to forget everything.
            last_n: The number of messages to forget.
        """
        if everything:
            context = self.context
            self.context = []
            return context
        forgotten = []
        for _ in range(last_n):
            last = self.forget_last()
            if last:
                forgotten.append(last)
        return forgotten

    def history(self) -> List[Message]:
        """Return the conversation history."""
        return self.context

    def remind_every(self, prompt: str | Image | Message, n: int) -> None:
        """Remind the agent of the prompt every n messages.

        Args:
            prompt: The prompt to remind the agent of.
            n: The frequency of the reminder.
        """
        message = Message([prompt]) if not isinstance(prompt, Message) else prompt
        self.reminders.append(Reminder(message, n))

    def _check_for_reminders(self) -> None:
        """Check if there are any reminders to show."""
        for reminder, n in self.reminders:
            if len(self.context) % n == 0:
                self.context.append(reminder)

    def act_and_parse(
        self,
        instruction: str,
        image: Image = None,
        parse_target: type[Sample] = Sample,
        context: list | str | Image | Message = None,
        model=None,
        max_retries: int = 1,
        record=False,
        **kwargs,
    ) -> Sample:
        """Responds to the given instruction, image, and context and parses the response into a Sample object.

        Args:
            instruction: The instruction to be processed.
            image: The image to be processed.
            parse_target: The target type to parse the response into.
            context: Additonal context to include in the response. If context is a list of messages, it will be interpreted
                as new memory.
            model: The model to use for the response.
            max_retries: The maximum number of retries to parse the response.
            record: Whether to record the interaction for training.
            **kwargs: Additional keyword arguments.
        """
        original_instruction = instruction
        for attempt in range(max_retries + 1):
            if record:
                response = self.act_and_record(instruction, image, context, model, **kwargs)
            else:
                response = self.act(instruction, image, context, model, **kwargs)
            response = response[response.find("{") : response.rfind("}") + 1]
            try:
                return parse_target.model_validate_json(response)
            except Exception as e:
                if attempt == max_retries:
                    raise ValueError(f"Failed to parse response after {max_retries + 1} attempts") from e
                error = f"Error parsing response: {e}"
                instruction = original_instruction + f". Avoid the following error: {error}"
                self.forget(last_n=2)
                logging.warning(f"\nReceived response: {response}.\n Retrying with error message: {instruction}")
        raise ValueError(f"Failed to parse response after {max_retries + 1} attempts")

    async def async_act_and_parse(
        self,
        instruction: str,
        image: Image = None,
        parse_target: Sample = Sample,
        context: list | str | Image | Message = None,
        model=None,
        max_retries: int = 1,
        **kwargs,
    ) -> Sample:
        """Responds to the given instruction, image, and context asynchronously and parses the response into a Sample object."""
        return await asyncio.to_thread(
            self.act_and_parse,
            instruction,
            image,
            parse_target,
            context,
            model=model,
            max_retries=max_retries,
            **kwargs,
        )

    def prepare_inputs(
        self, instruction: str, image: Image = None, context: list | str | Image | Message = None
    ) -> tuple[Message, list[Message]]:
        """Helper method to prepare the inputs for the agent.

        Args:
            instruction: The instruction to be processed.
            image: The image to be processed.
            context: Additonal context to include in the response. If context is a list of messages, it will be interpreted
                as new memory.
        """
        self._check_for_reminders()
        memory = self.context
        if context and all(isinstance(c, Message) for c in context):
            memory += context
            context = []

        # Prepare the inputs
        inputs = [instruction]
        if image is not None:
            inputs.append(image)
        if context:
            inputs.extend(context if isinstance(context, list) else [context])
        message = Message(role="user", content=inputs)

        return message, memory

    def postprocess_response(self, response: str, message: Message, memory: list[Message], **kwargs) -> str:
        """Postprocess the response."""
        self.context.append(message)
        self.context.append(Message(role="assistant", content=response))
        return response

    def act(
        self,
        instruction: str,
        image: Image = None,
        context: list | str | Image | Message = None,
        model=None,
        **kwargs,
    ) -> str:
        """Responds to the given instruction, image, and context.

        Uses the given instruction and image to perform an action.

        Args:
            instruction: The instruction to be processed.
            image: The image to be processed.
            context: Additonal context to include in the response. If context is a list of messages, it will be interpreted
                as new memory.
            model: The model to use for the response.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The response to the instruction.

        Example:
            >>> agent.act("Hello, world!", Image("scene.jpeg"))
            "Hello! What can I do for you today?"
            >>> agent.act("Return a plan to pickup the object as a python list.", Image("scene.jpeg"))
            "['Move left arm to the object', 'Move right arm to the object']"
        """
        message, memory = self.prepare_inputs(instruction, image, context)
        model = model or kwargs.pop("model", None)
        response = self.actor.predict(message, memory, model=model, **kwargs)
        return self.postprocess_response(response, message, memory, **kwargs)

    def act_and_stream(
        self, instruction: str, image: Image = None, context: list | str | Image | Message = None, model=None, **kwargs
    ) -> Generator[str, None, str]:
        """Responds to the given instruction, image, and context and streams the response."""
        message, memory = self.prepare_inputs(instruction, image, context)
        response = ""
        model = model or kwargs.pop("model", None)
        for chunk in self.actor.stream(memory + [message], model=model, **kwargs):
            response += chunk
            yield chunk
        return self.postprocess_response(response, message, memory, **kwargs)

    async def async_act_and_stream(
        self, instruction: str, image: Image = None, context: list | str | Image | Message = None, model=None, **kwargs
    ) -> AsyncGenerator[str, None]:
        # TODO(sebastian): fix this. Response is None maybe due to three nested async yields.
        raise NotImplementedError("Async streaming is not supported for this agent.")
        # message, memory, model = self.prepare_inputs(instruction, image, context, model)
        # model = model or kwargs.pop("model", None)
        # response = ""
        # async for chunk in self.actor.astream(memory + [message], model=model, **kwargs):
        #     response += chunk
        #     yield chunk
        # self.postprocess_response(response, message, memory, **kwargs)
