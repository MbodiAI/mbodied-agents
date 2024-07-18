import logging

from mbodied.agents.language import LanguageAgent
from mbodied.agents.sense.audio.audio_agent import AudioAgent
from mbodied.hardware.sim_interface import SimInterface
from mbodied.hardware.xarm_interface import XarmInterface
from mbodied.hri.hri import HRI, AgentTask
from mbodied.types.motion.control import HandControl

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    control_type = HandControl
    agents = {
        "audio_agent": AudioAgent(),
        "language_agent": LanguageAgent(context=f"you are a robot. output in json schema {control_type.model_json_schema()}. Your movement should always be smaller than 0.3", model_src="openai"),
        "robot": XarmInterface(),
    }

    tasks = [
        AgentTask(name="capture_image", agent="robot", inputs=[], outputs=["observation"], is_queue=False, func=lambda robot: robot.capture()),
        AgentTask(name="audio_task", agent="audio_agent", inputs=[], outputs=["instruction"], is_queue=False, func=lambda agent: agent.act()),
        AgentTask(name="language_task", agent="language_agent", inputs=["instruction", "observation"], outputs=["motion"], is_queue=False, func=lambda agent, instr, obs: agent.act_and_parse(instr, obs, parse_target=control_type)),
        AgentTask(name="robot_task", agent="robot", inputs=["motion"], outputs=[], is_queue=False, func=lambda robot, motion: robot.do(motion)),
    ]

    hri = HRI(agents, tasks)
    hri.run()
