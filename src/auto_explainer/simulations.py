from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SequenceSimulation:
    tokens: List[str]
    expected_activations: List[float]


@dataclass
class ScoredSequenceSimulation:
    simulation: SequenceSimulation
    true_activations: List[float]

    # Correlation coefficient between expected values(nornmalized) and the true activations(unnormalized)
    ev_correlation_score: float

    rsquared_score: Optional[float] = None

    # Score based on absolute difference between real and simulated activations.
    # absolute_dev_score = 1 - mean(abs(real-predicted))/mean(abs(real))
    absolute_dev_explained_score: Optional[float] = None


@dataclass
class ScoredSimulation:
    """Result of scoring a feature simulation on multiple sequences."""

    scored_sequence_simulations: List[ScoredSequenceSimulation]
    ev_correlation_score: Optional[float] = None
    rsquared_score: Optional[float] = None
    absolute_dev_explained_score: Optional[float] = None

    def get_preferred_score(self) -> Optional[float]:
        return self.ev_correlation_score


@dataclass
class ScoredExplanation:
    """Simulator parameters and the results of scoring it on multiple sequences."""

    explanation: str  # The explanation string used for simulation. (e.g., "This feature activates on medical terms.")
    scored_simulation: ScoredSimulation  # The scored simulation results. (multiple sequences)

    def get_preferred_score(self) -> Optional[float]:
        return self.scored_simulation.get_preferred_score()
