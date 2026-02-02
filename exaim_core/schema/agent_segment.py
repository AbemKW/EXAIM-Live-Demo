from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSegment:
    agent_id: str
    segment: str
