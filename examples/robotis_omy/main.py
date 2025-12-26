# examples/robotis_omy/main.py
import dataclasses
import logging

import tyro
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent

from examples.robotis_omy import env as _env


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    action_horizon: int = 25
    max_hz: float = 20.0

    num_episodes: int = 1
    max_episode_steps: int = 0


def main(args: Args) -> None:
    ws = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = ws.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")

    runtime = _runtime.Runtime(
        environment=_env.OmyRealEnvironment(reset_position=metadata.get("reset_pose")),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(policy=ws, action_horizon=args.action_horizon)
        ),
        subscribers=[],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )
    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
