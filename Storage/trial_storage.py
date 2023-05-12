from dataclasses import dataclass
import requests


@dataclass
class Trial:
    games: list[str]
    win_rate: float
    agent_1: str
    agent_2: str
    time: str
    avg_turns: float


class TrialStorage:
    def upload_trial(self, trial: Trial):
        res = requests.post("http://localhost:8000/trials", json=trial.__dict__)
        print(res.status_code)
        print(res.text)
