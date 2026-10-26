from src.auto_explainer.explanations import FeatureExplanations
from src.global_settings import settings

EXPLANATION_DIR = settings.feature_explanation_dir


def load_feature_explanation(
    checkpoint: str,
    layer: int,
    feature_index: int,
    postprocess: bool = True,
) -> tuple[str, str]:
    """Load explanation and description for a feature."""
    explanation_path = (
        EXPLANATION_DIR / checkpoint / str(layer) / f"{feature_index}.jsonl"
    )
    try:
        feature_explanations = FeatureExplanations.load_from_jsonl(explanation_path)
        explanation = (
            feature_explanations.get_best_explanation().explanation
            if feature_explanations.get_best_explanation()
            else "N/A"
        )
        description = (
            feature_explanations.explanations[1].explanation
            if len(feature_explanations.explanations) > 1
            else ""
        )
    except FileNotFoundError:
        explanation = "N/A"
        description = ""

    if postprocess:
        explanation = explanation.replace("**", "").strip()
        description = description.replace("**", "").strip()

    return explanation, description
