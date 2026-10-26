import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScoredExplanation:
    explanation: str
    score: float | None


@dataclass
class FeatureExplanations:
    explanations: List[ScoredExplanation] = field(default_factory=list)

    @classmethod
    def load_from_jsonl(cls, file_path: str) -> "FeatureExplanations":
        items = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                items.append(ScoredExplanation(**data))
        return cls(explanations=items)

    def save_to_jsonl(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            for explanation in self.explanations:
                json_line = json.dumps(
                    {
                        "explanation": explanation.explanation,
                        "score": explanation.score,
                    },
                    ensure_ascii=False,
                )
                f.write(json_line + "\n")

    def get_best_explanation(self) -> Optional[ScoredExplanation]:
        if not self.explanations:
            return None
        if all(exp.score is None for exp in self.explanations):
            return self.explanations[0]

        return max(self.explanations, key=lambda x: x.score)
