<div align="center">
 <img src="assets/logo_full.png" width=200;/>
  <div>&nbsp;</div>
  <div align="center">
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5"></font></b>
    <sup>
      <a href="https://api.mbodi.ai">
        <i><font size="4">Benchmark, Explore, and Send API Requests Now</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mbodied)](https://pypi.org/project/mbodied/)
[![PyPI](https://img.shields.io/pypi/v/mbodied)](https://pypi.org/project/mbodied)
[![MacOS](https://github.com/mbodiai/opensource/actions/workflows/macos.yml/badge.svg?branch=main)](https://github.com/mbodiai/opensource/actions/workflows/macos.yml)
[![Ubuntu](https://github.com/mbodiai/opensource/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/mbodiai/opensource/actions/workflows/ubuntu.yml)
[![Documentation Status](https://readthedocs.com/projects/mbodi-ai-mbodied/badge/?version=latest)](https://readthedocs.com/projects/mbodi-ai-mbodied-agents/badge/?version=latest)

📖 **Docs**: [readthedocs](https://mbodi-ai-mbodied-agents.readthedocs-hosted.com/en/latest/)

🚀 **Simple Robot Agent Example:** [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1KN0JohcjHX42wABBHe-CxXP-NWJjbZts?usp=sharing) </br>
💻 **Simulation Example with [SimplerEnv](https://github.com/simpler-env/SimplerEnv):** [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/18oiuw1yTxO5x-eT7Z8qNyWtjyYd8cECI?usp=sharing) </br>
🤖 **Motor Agent using OpenVLA:** [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1flnMrqyepGOO8J9rE6rehzaLdZPsw6lX?usp=sharing)</br>
⏺️ **Record Dataset on a Robot**[<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/15UuFbMUJGEjqJ_7I_b5EvKvLCKnAc8bB?usp=sharing)</br>

🫡 **Support, Discussion, and How-To's** **:** </br>
[![](https://dcbadge.limes.pink/api/server/BPQ7FEGxNb?theme=discord&?logoColor=pink)](https://discord.gg/BPQ7FEGxNb)

</div>

**Updates:**

**Aug 6 2024, embodied-agents v1.1**

- Added the features to record dataset on [robot](mbodied/robots/robot.py) natively.
- Add multiple new Sensory Agents, i.e. [depth estimation](mbodied/agents/sense/depth_estimation_agent.py), [object detection](mbodied/agents/sense/object_detection_agent.py), [image segmentation](mbodied/agents/sense/segmentation_agent.py) with public [API endpoints](https://api.mbodi.ai/sense/) hosted.

**June 30 2024, embodied-agents v1.0**:

- Added Motor Agent supporting OpenVLA with free [API endpoint](https://api.mbodi.ai/community-models) hosted.
- Added Sensory Agent supporting i.e. 3D object pose detection.
- Improved automatic dataset recording.
- Agent now can make remote act calls to API servers i.e. Gradio, vLLM.
- Bug fixes and performance improvements have been made.
- PyPI project is renamed to `mbodied`.

# embodied agents

**embodied agents** is a toolkit for integrating large multi-modal models into existing robot stacks with just a few lines of code. It provides consistency, reliability, scalability and is configurable to any observation and action space.

<img src="assets/new_demo.gif" alt="Demo GIF" style="width: 550px;">

- [embodied agents](#embodied-agents)
  - [Overview](#overview)
    - [Motivation](#motivation)
    - [Goals](#goals)
    - [Limitations](#limitations)
    - [Scope](#scope)
    - [Features](#features)
    - [Endpoints](#endpoints)
    - [Support Matrix](#support-matrix)
    - [To Do](#to-do)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
    - [Customize a Motion to fit a robot's action space.](#customize-a-motion-to-fit-a-robots-action-space)
    - [Run a robotics transformer model on a robot.](#run-a-robotics-transformer-model-on-a-robot)
    - [Notebooks](#notebooks)
  - [The Sample Class](#the-sample-class)
  - [Building Blocks](#building-blocks)
  - [Directory Structure](#directory-structure)
  - [Contributing](#contributing)

## Overview

This repository is broken down into 3 main components: **Agents**, **Data**, and **Hardware**. Inspired by the efficiency of the central nervous system, each component is broken down into 3 meta-modalities: **Language**, **Motion**, and **Sense**. Each agent has an `act` method that can be overridden and satisfies:

- **Language Agents** always return a string.
- **Motor Agents** always return a `Motion`.
- **Sensory Agents** always return a `SensorReading`.

A call to `act` or `async_act` can perform local or remote inference synchronously or asynchronously. Remote execution can be performed with [Gradio](https://www.gradio.app/docs/python-client/introduction), [httpx](https://www.python-httpx.org/), or different LLM clients. Validation is performed with [Pydantic](https://docs.pydantic.dev/latest/).

<img src="assets/architecture.jpg" alt="Architecture Diagram" style="width: 700px;">

- Language Agents natively support OpenAI, Anthropic, Ollama, vLLM, Gradio, etc
- Motor Agents natively support OpenVLA, RT1(upcoming)
- Sensory Agents support Depth Anything, YOLO, Segment Anything 2

Jump to [getting started](#getting-started) to get up and running on [real hardware](https://colab.research.google.com/drive/1KN0JohcjHX42wABBHe-CxXP-NWJjbZts?usp=sharing) or [simulation](https://colab.research.google.com/drive/1gJlfEvsODZWGn3rK8Nx4A0kLnLzJtJG_?usp=sharing). Be sure to join our [Discord](https://discord.gg/BPQ7FEGxNb) for 🥇-winning discussions :)

**⭐ Give us a star on GitHub if you like us!**

### Motivation

<details>
<summary>There is a signifcant barrier to entry for running SOTA models in robotics</summary>

It is currently unrealistic to run state-of-the-art AI models on edge devices for responsive, real-time applications. Furthermore,
the complexity of integrating multiple models across different modalities is a significant barrier to entry for many researchers,
hobbyists, and developers. This library aims to address these challenges by providing a simple, extensible, and efficient way to
integrate large models into existing robot stacks.

</details>

### Goals

<details>
<summary>Facillitate data-collection and sharing among roboticists.</summary>

This requires reducing much of the complexities involved with setting up inference endpoints, converting between different model formats, and collecting and storing new datasets for future availibility.

We aim to achieve this by:

1. Providing simple, Python-first abstrations that are modular, extensible and applicable to a wide range of tasks.
2. Providing endpoints, weights, and interactive Gradio playgrounds for easy access to state-of-the-art models.
3. Ensuring that this library is observation and action-space agnostic, allowing it to be used with any robot stack.

Beyond just improved robustness and consistency, this architecture makes asynchronous and remote agent execution exceedingly simple. In particular we demonstrate how responsive natural language interactions can be achieved in under 30 lines of Python code.

</details>

### Limitations

_Embodied Agents are not yet capable of learning from in-context experience_:

- Frameworks for advanced RAG techniques are clumsy at best for OOD embodied applications however that may improve.
- Amount of data required for fine-tuning is still prohibitively large and expensive to collect.
- Online RL is still in its infancy and not yet practical for most applications.

### Scope

- This library is intended to be used for research and prototyping.
- This library is still experimental and under active development. Breaking changes may occur although they will be avoided as much as possible. Feel free to report issues!

### Features

- Extensible, user-friendly python SDK with explicit typing and modularity
- Asynchronous and remote thread-safe agent execution for maximal responsiveness and scalability.
- Full-compatiblity with HuggingFace Spaces, Datasets, Gymnasium Spaces, Ollama, and any OpenAI-compatible api.
- Automatic dataset-recording and optionally uploads dataset to huggingface hub.

### Endpoints

- [OpenVLA](https://api.mbodi.ai/community-models/)
- [Embodied AI Playground](https://api.mbodi.ai/benchmark/)
- [3D Object Pose Detection](https://api.mbodi.ai/3d-object-pose-detection/)

### Support Matrix

- Closed: OpenAI, Anthropic
- Open Weights: OpenVLA, Idefics2, Llava-1.6-Mistral, Phi-3-vision-128k-instruct
- All gradio endpoints hosted on HuggingFace spaces.

### Roadmap

- [ ] More Motor Agents
- [ ] Yolo, SAM2, DepthAnything Sensory Agents
- [ ] FineTuning Scripts

## Installation

```shell
pip install mbodied

# For audio support
pip install mbodied[audio]
```

Or install from source:

```shell
pip install git+https://github.com/mbodiai/embodied-agents.git
```

## Getting Started

### Customize a Motion to fit a robot's action space.

```python
from mbodied.types.motion.control import HandControl, FullJointControl
from mbodied.types.motion import AbsoluteMotionField, RelativeMotionField

class FineGrainedHandControl(HandControl):
    comment: str = Field(None, description="A comment to voice aloud.")
    index: FullJointControl = AbsoluteMotionField([0,0,0],bounds=[-3.14, 3.14], shape=(3,))
    thumb: FullJointControl = RelativeMotionField([0,0,0],bounds=[-3.14, 3.14], shape=(3,))
```

### Run a robotics transformer model on a robot.

```python
import os
from mbodied.agents import LanguageAgent
from mbodied.agents.motion import OpenVlaAgent
from mbodied.agents.sense.audio import AudioAgent
from mbodied.robots import SimRobot

cognition = LanguageAgent(
  context="You are an embodied planner that responds with a python list of strings and nothing else.",
  api_key=os.getenv("OPENAI_API_KEY"),
  model_src="openai",
  recorder="auto",
)
audio = AudioAgent(use_pyaudio=False, api_key=os.getenv("OPENAI_API_KEY")) # pyaudio is buggy on mac
motion = OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/")

# Subclass and override do() and capture() methods.
robot = SimRobot()

instruction = audio.listen()
plan = cognition.act(instruction, robot.capture())

for step in plan.strip('[]').strip().split(','):
  print("\nMotor agent is executing step: ", step, "\n")
  for _ in range(10):
    hand_control = motion.act(step, robot.capture())
    robot.do(hand_control)
```

Example Scripts:

- [1_simple_robot_agent.py](examples/1_simple_robot_agent.py): A very simple language based cognitive agent taking instruction from user and output voice and actions.
- [2_openvla_motor_agent_example.py](examples/2_openvla_motor_agent_example.py): Run robotic transformers, i.e. OpenVLA, in several lines on the robot.
- [3_reason_plan_act_robot.py](examples/3_reason_plan_act_robot.py): Full example of language based cognitive agent and OpenVLA motor agent executing task.
- [4_language_reason_plan_act_robot.py](examples/4_language_reason_plan_act_robot.py): Full example of all languaged based cognitive and motor agent executing task.
- [5_teach_robot_record_dataset.py](examples/5_teach_robot_record_dataset.py): Example of collecting dataset on robot's action at a specific frequency by just yelling at the robot!

### Notebooks

Real Robot Hardware: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KN0JohcjHX42wABBHe-CxXP-NWJjbZts?usp=sharing)

Simulation with: [SimplerEnv](https://github.com/simpler-env/SimplerEnv.git) : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18oiuw1yTxO5x-eT7Z8qNyWtjyYd8cECI?usp=sharing)

Run OpenVLA with embodied-agents in simulation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1flnMrqyepGOO8J9rE6rehzaLdZPsw6lX?usp=sharing)

Record dataset on a robot: [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/15UuFbMUJGEjqJ_7I_b5EvKvLCKnAc8bB?usp=sharing)

## The [Sample](mbodied/base/sample.py) Class

The Sample class is a base model for serializing, recording, and manipulating arbitrary data. It is designed to be extendable, flexible, and strongly typed. By wrapping your observation or action objects in the [Sample](mbodied/base/sample.py) class, you'll be able to convert to and from the following with ease:

- A Gym space for creating a new Gym environment.
- A flattened list, array, or tensor for plugging into an ML model.
- A HuggingFace dataset with semantic search capabilities.
- A Pydantic BaseModel for reliable and quick json serialization/deserialization.

To learn more about all of the possibilities with embodied agents, check out the [documentation](https://mbodi-ai-mbodied-agents.readthedocs-hosted.com/en/latest/)

### 💡 Did you know

- You can `pack` a list of `Sample`s or Dicts into a single `Sample` or `Dict` and `unpack` accordingly?
- You can `unflatten` any python structure into a `Sample` class so long you provide it with a valid json schema?

<details> <summary><h2 style="display: inline-block;">Deep Dive</h2></summary>

## Building Blocks

#### Creating a Sample

Creating a Sample requires just wrapping a python dictionary with the `Sample` class. Additionally, they can be made from kwargs, Gym Spaces, and Tensors to name a few.

```python
from mbodied.types.sample import Sample
# Creating a Sample instance
sample = Sample(observation=[1,2,3], action=[4,5,6])

# Flattening the Sample instance
flat_list = sample.flatten()
print(flat_list) # Output: [1, 2, 3, 4, 5, 6]

# Generating a simplified JSON schema
>>> schema = sample.schema()
{'type': 'object', 'properties': {'observation': {'type': 'array', 'items': {'type': 'integer'}}, 'action': {'type': 'array', 'items': {'type': 'integer'}}}}

# Unflattening a list into a Sample instance
Sample.unflatten(flat_list, schema)
>>> Sample(observation=[1, 2, 3], action=[4, 5, 6])
```

#### Serialization and Deserialization with Pydantic

The Sample class leverages Pydantic's powerful features for serialization and deserialization, allowing you to easily convert between Sample instances and JSON.

```python
# Serialize the Sample instance to JSON
sample = Sample(observation=[1,2,3], action=[4,5,6])
json_data = sample.model_dump_json()
print(json_data) # Output: '{"observation": [1, 2, 3], "action": [4, 5, 6]}'

# Deserialize the JSON data back into a Sample instance
json_data = '{"observation": [1, 2, 3], "action": [4, 5, 6]}'
sample = Sample.model_validate(from_json(json_data))
print(sample) # Output: Sample(observation=[1, 2, 3], action=[4, 5, 6])
```

#### Converting to Different Containers

```python
# Converting to a dictionary
sample_dict = sample.to("dict")
print(sample_dict) # Output: {'observation': [1, 2, 3], 'action': [4, 5, 6]}

# Converting to a NumPy array
sample_np = sample.to("np")
print(sample_np) # Output: array([1, 2, 3, 4, 5, 6])

# Converting to a PyTorch tensor
sample_pt = sample.to("pt")
print(sample_pt) # Output: tensor([1, 2, 3, 4, 5, 6])
```

#### Gym Space Integration

```python
gym_space = sample.space()
print(gym_space)
# Output: Dict('action': Box(-inf, inf, (3,), float64), 'observation': Box(-inf, inf, (3,), float64))
```

See [sample.py](mbodied/base/sample.py) for more details.

### Message

The [Message](mbodied/types/message.py) class represents a single completion sample space. It can be text, image, a list of text/images, Sample, or other modality. The Message class is designed to handle various types of content and supports different roles such as user, assistant, or system.

You can create a `Message` in versatile ways. They can all be understood by mbodi's backend.

```python
from mbodied.types.message import Message

Message(role="user", content="example text")
Message(role="user", content=["example text", Image("example.jpg"), Image("example2.jpg")])
Message(role="user", content=[Sample("Hello")])
```

### Backend

The [Backend](mbodied/base/backend.py) class is an abstract base class for Backend implementations. It provides the basic structure and methods required for interacting with different backend services, such as API calls for generating completions based on given messages. See [backend directory](mbodied/agents/backends) on how various backends are implemented.

### Agent

[Agent](mbodied/base/agent.py) is the base class for various agents listed below. It provides a template for creating agents that can talk to a remote backend/server and optionally record their actions and observations.

### Language Agent

The [Language Agent](mbodied/agents/language/language_agent.py) can connect to different backends or transformers of your choice. It includes methods for recording conversations, managing context, looking up messages, forgetting messages, storing context, and acting based on an instruction and an image.

Natively supports API services: OpenAI, Anthropic, vLLM, Ollama, HTTPX, or any gradio endpoints. More upcoming!

To use OpenAI for your robot backend:

```python
from mbodied.agents.language import LanguageAgent

agent = LanguageAgent(context="You are a robot agent.", model_src="openai")
```

To execute an instruction:

```python
instruction = "pick up the fork"
response = robot_agent.act(instruction, image)
```

Language Agent can connect to vLLM as well. For example, suppose you are running a vLLM server Mistral-7B on 1.2.3.4:1234. All you need to do is:

```python
agent = LanguageAgent(
    context=context,
    model_src="openai",
    model_kwargs={"api_key": "EMPTY", "base_url": "http://1.2.3.4:1234/v1"},
)
response = agent.act("Hello, how are you?", model="mistralai/Mistral-7B-Instruct-v0.3")
```

### Motor Agent

[Motor Agent](mbodied/agents/motion/motor_agent.py) is similar to Language Agent but instead of returning a string, it always returns a `Motion`. Motor Agent is generally powered by robotic transformer models, i.e. OpenVLA, RT1, Octo, etc.
Some small model, like RT1, can run on edge devices. However, some, like OpenVLA, may be challenging to run without quantization. See [OpenVLA Agent](mbodied/agents/motion/openvla_agent.py) and an [example OpenVLA server](examples/servers/gradio_example_openvla.py)

### Sensory Agent

These agents interact with the environment to collect sensor data. They always return a `SensorReading`, which can be various forms of processed sensory input such as images, depth data, or audio signals.

Currently, we have:

- [depth estimation](mbodied/agents/sense/depth_estimation_agent.py)
- [object detection](mbodied/agents/sense/object_detection_agent.py)
- [image segmentation](mbodied/agents/sense/segmentation_agent.py)
- [3D object pose estimator](mbodied/agents/sense/object_pose_estimator_3d.py)

agents that process robot's sensor information.

### Motions

The [motion_controls](mbodied/types/motion_controls.py) module defines various motions to control a robot as Pydantic models. They are also subclassed from `Sample`, thus possessing all the capability of `Sample` as mentioned above. These controls cover a range of actions, from simple joint movements to complex poses and full robot control.

### Robot

You can integrate your custom robot hardware by subclassing [Robot](mbodied/robot/robot.py) quite easily. You only need to implement `do()` function to perform actions (and some additional methods if you want to record dataset on the robot). In our examples, we use a [mock robot](mbodied/robot/sim_robot.py). We also have an [XArm robot](mbodied/robot/xarm_robot.py) as an example.

#### Recording a Dataset

Recording a dataset on a robot is very easy! All you need to do is implement the `get_observation()`, `get_state()`, and `prepare_action()` methods for your robot. After that, you can record a dataset on your robot anytime you want. See [examples/5_teach_robot_record_dataset.py](examples/5_teach_robot_record_dataset.py) and this colab: [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/15UuFbMUJGEjqJ_7I_b5EvKvLCKnAc8bB?usp=sharing) for more details.

```python
from mbodied.robots import SimRobot
from mbodied.types.motion.control import HandControl, Pose

robot = SimRobot()
robot.init_recorder(frequency_hz=5)
with robot.record("pick up the fork"):
  motion = HandControl(pose=Pose(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3))
  robot.do(motion)
```

### Recorder

Dataset [Recorder](mbodied/data/recording.py) is a lower level recorder to record your conversation and the robot's actions to a dataset as you interact with/teach the robot. You can define any observation space and action space for the Recorder. See [gymnasium](https://github.com/Farama-Foundation/Gymnasium) for more details about spaces.

```python
from mbodied.data.recording import Recorder
from mbodied.types.motion.control import HandControl
from mbodied.types.sense.vision import Image
from gymnasium import spaces

observation_space = spaces.Dict({
    'image': Image(size=(224, 224)).space(),
    'instruction': spaces.Text(1000)
})
action_space = HandControl().space()
recorder = Recorder('example_recorder', out_dir='saved_datasets', observation_space=observation_space, action_space=action_space)

# Every time robot makes a conversation or performs an action:
recorder.record(observation={'image': image, 'instruction': instruction,}, action=hand_control)
```

The dataset is saved to `./saved_datasets`.

### Replayer

The [Replayer](mbodied/data/replaying.py) class is designed to process and manage data stored in HDF5 files generated by `Recorder`. It provides a variety of functionalities, including reading samples, generating statistics, extracting unique items, and converting datasets for use with HuggingFace. The Replayer also supports saving specific images during processing and offers a command-line interface for various operations.

Example for iterating through a dataset from Recorder with Replayer:

```python
from mbodied.data.replaying import Replayer

replayer = Replayer(path=str("path/to/dataset.h5"))
for observation, action in replayer:
   ...
```

</details>

## Directory Structure

```shell
├─ assets/ ............. Images, icons, and other static assets
├─ examples/ ........... Example scripts and usage demonstrations
├─ resources/ .......... Additional resources for examples
├─ src/
│  └─ mbodied/
│     ├─ agents/ ....... Modules for robot agents
│     │  ├─ backends/ .. Backend implementations for different services for agents
│     │  ├─ language/ .. Language based agents modules
│     │  ├─ motion/ .... Motion based agents modules
│     │  └─ sense/ ..... Sensory, e.g. audio, processing modules
│     ├─ data/ ......... Data handling and processing
│     ├─ hardware/ ..... Hardware modules, i.e. camera
│     ├─ robot/ ........ Robot interface and interaction
│     └─ types/ ........ Common types and definitions
└─ tests/ .............. Unit tests
```

## Contributing

We welcome issues, questions and PRs. See the [contributing guide](CONTRIBUTING.md) for more information.
