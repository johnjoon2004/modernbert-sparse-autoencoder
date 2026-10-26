from enum import Enum
from typing import Dict, List, Optional, TypedDict

import numpy as np


class TermLevel(str, Enum):
    SENTENCE = "sentence (a short descriptive sentence)"
    PHRASE = "phrase (a short descriptive label)"


SYSTEM_MESSAGE = """We are studying the FEATURES that emerge in a deep learning classification model.  
A feature can be observed through multiple document samples and the active tokens included in them.  
Each feature tends to capture or detect a particular pattern within short documents.

You are assisting us in analyzing these document samples and their token activations to infer the semantic meaning of each feature.  
You will be given dozens of independent document samples.  
Each document sample includes:
- The full text of the document
- A list of tokens
- The activation value (a non-negative real number) for each token

Carefully examine the parts of the documents where the feature shows high activation values and summarize, in a single concise {term_level}, what the feature is looking for.  

Features may vary in their level of abstraction and complexity.  
For example, a feature may represent:
- A specific keyword or phrase  
- A lexical, syntactic, or stylistic pattern  
- A conceptual theme shared by multiple keywords  
- Or a more complex combination of expressions or ideas  

Think carefully and reason in sufficient depth to describe the most accurate concept the feature represents.  
Do not simply list examples of words or tokens.

Remember:
- Higher activation values indicate a stronger match between the feature and the corresponding text span.  
- Your output should consist only of the final short phrase (label) describing what the feature represents.  
- Output in either English or Korean.  
"""
# System prompt token length: 289


USER_MESSAGE_TEMPLATE = """<Feature #{feature_index}>

Below are text samples where this feature shows high activation values. Each sample shows:
1. The full text of the document
2. The tokens with highest activation values for this feature

Please analyze these samples and describe in ONE CONCISE {term_level}, what pattern or concept this feature detects.
{formatted_sample_records}

Your explanation:"""
# User prompt template token: 71


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class HarmonyMessage(TypedDict):
    role: str
    content: str


class PromptFormat(str, Enum):
    COMPLETION = "completion"
    CHAT = "chat"
    INSTRUCTION_FOLLOWING = "instruction_following"

    @classmethod
    def from_string(cls, s: str) -> "PromptFormat":
        for prompt_format in cls:
            if prompt_format.value == s:
                return prompt_format
        raise ValueError(f"{s} is not a valid PromptFormat")


class PromptBuilder:
    def __init__(self):
        self._messages: List[HarmonyMessage] = []

    def add_message(self, role: Role, content: str) -> None:
        self._messages.append(HarmonyMessage(role=role.value, content=content))

    def build(self, prompt_format: PromptFormat) -> str | List[HarmonyMessage]:
        messages = [dict(msg) for msg in self._messages]

        # Validates the role order
        # NOTE: Assume that must starts with system message
        expected_next_role = Role.SYSTEM
        for message in messages:
            role = message["role"]
            if role != expected_next_role:
                raise ValueError(
                    f"Invalid message role sequence: expected message from {expected_next_role} but got message from {role}"
                )
            if role == Role.SYSTEM:
                expected_next_role = Role.USER
            elif role == Role.USER:
                expected_next_role = Role.ASSISTANT
            elif role == Role.ASSISTANT:
                expected_next_role = Role.USER

        if prompt_format == PromptFormat.INSTRUCTION_FOLLOWING:
            last_user_message = None
            for message in messages:
                if message["role"] == Role.USER:
                    last_user_message = message
            if last_user_message is None:
                raise ValueError("No user message found for instruction-following format.")

            last_user_message["content"] += "<|endofprompt|>"

        if prompt_format == PromptFormat.CHAT:
            return messages

        elif prompt_format in {PromptFormat.COMPLETION, PromptFormat.INSTRUCTION_FOLLOWING}:
            prompt_parts = []
            for message in messages:
                prompt_parts.append(message["content"])
            return "\n".join(prompt_parts)

        else:
            raise ValueError(f"Unknown prompt format: {prompt_format}")


def format_sample_records(
    samples: List[Dict],
    num_samples: Optional[int] = None,
    topk_tokens: Optional[int] = None,
    order_tokens: bool = True,
    omit_zeros: bool = True,
) -> str:
    """
    Format samples (dict) into readable activation records.

    Args:
        samples: Dict containing 'tokens' and 'activations' lists
        num_samples: Number of samples to include in the output
        topk_tokens: Number of top tokens by activation to include per sample
        omit_zeros: Whether to omit tokens with zero activation

    Returns:
        (str) Formatted string representation of activation records
    """

    key_tokens = "active_tokens"
    # assert key_tokens in samples[0], f"'{key_tokens}' key not found in samples"

    if num_samples is not None:
        samples_to_show = min(num_samples, len(samples))
        samples = samples[:samples_to_show]

    formatted_parts = []
    for i, sample in enumerate(samples):
        text = sample["text"]

        if len(text) < 10:  # skip too-short
            continue

        formatted_parts.append(f"\n\n--- Sample {i + 1} ---")
        formatted_parts.append(f"Text: {text}")

        tokens = sample[key_tokens]

        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        if order_tokens:
            tokens = sorted(tokens, key=lambda x: x["token_score"], reverse=True)
        if topk_tokens is not None:
            tokens = sorted(tokens, key=lambda x: x["token_score"], reverse=True)[:topk_tokens]

        token_parts = []
        for token in tokens:
            token_idx = token["token_idx"]
            token_str = token["token_str"]
            token_score = token["token_score"]  # activation value

            if omit_zeros and token_score < 1e-8:
                continue

            token_parts.append(f"  • [idx={token_idx}] '{token_str}' : {token_score:.3f}")

        if token_parts:
            formatted_parts.append("\nActive Tokens:")
            formatted_parts.extend(token_parts)
        else:
            formatted_parts.append("\nActive Tokens: (none)")

    formatted_sample_records = "\n".join(formatted_parts)
    return formatted_sample_records


def format_user_message(
    feature_index: int,
    samples: List[Dict],
    term_level: TermLevel = TermLevel.PHRASE,
    num_samples: Optional[int] = None,
    topk_tokens: Optional[int] = None,
) -> str:
    formatted_sample_records = format_sample_records(
        samples=samples,
        num_samples=num_samples,
        topk_tokens=topk_tokens,
        omit_zeros=True,
    )
    user_message = USER_MESSAGE_TEMPLATE.format(
        feature_index=feature_index,
        term_level=term_level.value,
        formatted_sample_records=formatted_sample_records,
    )
    return user_message


# @dataclass
# class Example:
#     records: List[SampleRecord]
#     explanation: str
#     first_revealed_activation_indices: List[int]
#     """For each activation record, the index of the first token for which the activation value in the
#     prompt should be an actual number rather than "unknown".
#     Examples all start with the activations rendered as "unknown", then transition to revealing
#     specific normalized activation values. The goal is to lead the model to predict that activation
#     sequences will eventually transition to predicting specific activation values instead of just
#     "unknown". This lets us cheat and get predictions of activation values for every token in a
#     single round of inference by having the activations in the sequence we're predicting always be
#     "unknown" in the prompt: the model will always think that maybe the next token will be a real activation."""

# FEW_SHOT_EXAMPLES = [
#     Example(
#         records=[
#             SampleRecord(tokens=["a", "b", "c"], activations=[1.0, 0.0, 0.0]),
#             SampleRecord(tokens=["d", "e", "f"], activations=[0.0, 1.0, 0.0]),
#         ],
#         explanation="vowel letters",
#         first_revealed_activation_indices=[0, 1],
#     ),
# ]
