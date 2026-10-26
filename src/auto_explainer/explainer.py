import json
import random
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from src.auto_explainer.explanations import FeatureExplanations, ScoredExplanation
from src.auto_explainer.prompts import (
    SYSTEM_MESSAGE,
    PromptBuilder,
    PromptFormat,
    Role,
    TermLevel,
    format_user_message,
)
from src.features import load_sample_records
from src.global_settings import settings
from src.utils.vllm.client import VLLMClient
from src.utils.vllm.config import GenConfig
from transformers import AutoTokenizer


class AutoExplainer(ABC):
    """
    Reference:
        [openai/automated-interpretability](https://github.com/openai/automated-interpretability/blob/main/neuron-explainer/neuron_explainer/explanations/explainer.py)
    """

    def __init__(self, llm_client, **kwargs):
        self.llm_client = llm_client

        if kwargs and "checkpoint" in kwargs and "layer" in kwargs and "feature_index" in kwargs:
            self.init_feature(**kwargs)

    def init_feature(self, checkpoint: str, layer: int | str, feature_index: int | str) -> None:
        self.checkpoint = checkpoint
        self.layer = int(layer)
        self.feature_index = int(feature_index)

    @abstractmethod
    def make_explanation_prompt(self, prompt_template: PromptFormat) -> str:
        return

    @abstractmethod
    def generate_explanations(
        self,
        dataset: List[Dict[str, Any]],
        save_path: Path,
        prompt_template: str,
        system_prompt: str | None = None,
        gen_config: GenConfig = GenConfig(),
        max_workers: int = 4,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        prompts = [prompt_template.format(**data) for data in dataset]

        responses = self.llm_client.generate_batch_parallel(
            prompts=prompts,
            config=gen_config,
            system_prompt=system_prompt,
            extract_reasoning=False,
            max_workers=max_workers,
        )

        results = []
        for data, response in zip(dataset, responses):
            results.append(
                {
                    **data,
                    "annotation": response,
                }
            )

        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        return results

    @abstractmethod
    def postprocess_explanations(
        self,
    ) -> list[Any]:
        pass

    @abstractmethod
    def make_explanation_prompt(self) -> str | list[Any]:
        pass


class ZeroShotAutoExplainer(AutoExplainer):
    def __init__(self, llm_client: VLLMClient | None = None):
        self.llm_client = llm_client or VLLMClient()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
        self.seed = 42
        self._sample_cache = {}

    def init_messages(self, system_message: str | None):
        if system_message is None:
            system_message = SYSTEM_MESSAGE

        self.prompt_builder = PromptBuilder()
        self.prompt_builder.add_message(role=Role.SYSTEM, content=system_message)

    def init_feature(self, checkpoint, layer, feature_index):
        self.checkpoint = checkpoint
        self.layer = int(layer)
        self.feature_index = int(feature_index)

    def get_samples(self, num_samples: int = 20, randomize: bool = True) -> List[Dict]:
        cache_key = (self.checkpoint, self.layer, self.feature_index)

        if cache_key not in self._sample_cache:
            samples_records = load_sample_records(
                checkpoint=self.checkpoint,
                layer=self.layer,
                feature_index=self.feature_index,
                sample_type="positive",
                as_frame=False,
            )
            self._sample_cache[cache_key] = samples_records
        else:
            samples_records = self._sample_cache[cache_key]

        if randomize:
            random.seed(self.seed)
            samples = random.sample(samples_records, min(num_samples, len(samples_records)))
        else:
            samples = samples_records[:num_samples]

        samples = [asdict(sample) for sample in samples]
        return samples

    def make_explanation_prompt(self, **kwargs) -> str:
        term_level = kwargs.get("term_level", TermLevel.PHRASE)
        num_samples = kwargs.get("num_samples", 20)
        num_tokens = kwargs.get("num_tokens", 10)
        randomize = kwargs.get("randomize", True)

        samples = self.get_samples(num_samples=num_samples, randomize=randomize)
        feature_explain_prompt = format_user_message(
            feature_index=self.feature_index,
            samples=samples,
            term_level=term_level,
            num_samples=num_samples,
            topk_tokens=num_tokens,
        )
        return feature_explain_prompt

    def generate_explanation(self, gen_config: GenConfig, **kwargs) -> str:
        system_message = kwargs.get("system_prompt", None)
        self.init_messages(system_message=system_message)
        explanation_prompt = self.make_explanation_prompt(**kwargs)
        self.prompt_builder.add_message(role=Role.USER, content=explanation_prompt)

        # prompt = self.prompt_builder.build(PromptFormat.CHAT)
        # TODO: integrate `VLLMClient.genearte` with `prompt_builder`
        response = self.llm_client.generate(
            prompt=explanation_prompt, config=gen_config, system_prompt=system_message, extract_reasoning=False
        )
        explanation = response

        return explanation

    async def generate_explanation_async(self, gen_config: GenConfig, **kwargs) -> str:
        system_message = kwargs.get("system_prompt", None)
        self.init_messages(system_message=system_message)
        explanation_prompt = self.make_explanation_prompt(**kwargs)
        self.prompt_builder.add_message(role=Role.USER, content=explanation_prompt)

        response = await self.llm_client.generate_async(
            prompt=explanation_prompt,
            config=gen_config,
            system_prompt=system_message,
            extract_reasoning=False,
        )
        return response

    async def generate_explanations_async(
        self, gen_config: GenConfig, term_levels: List[TermLevel], **kwargs
    ) -> List[str]:
        system_message = kwargs.get("system_prompt", None)
        prompts = []
        for term_level in term_levels:
            self.init_messages(system_message=system_message)
            explanation_prompt = self.make_explanation_prompt(
                term_level=term_level, **{k: v for k, v in kwargs.items() if k != "term_level"}
            )
            prompts.append(explanation_prompt)

        responses = await self.llm_client.generate_batch_async(
            prompts=prompts,
            config=gen_config,
            system_prompt=system_message,
            extract_reasoning=False,
            max_concurrent=len(prompts),
        )
        return responses

    @staticmethod
    async def batch_explain_features(
        explainer_configs: List[Dict],
        llm_client: VLLMClient,
        gen_config: GenConfig,
        max_concurrent: int = 20,
    ) -> List[List[str]]:
        """여러 feature의 explanation을 큰 배치로 처리

        Args:
            explainer_configs: [{
                'checkpoint': str,
                'layer': int,
                'feature_index': int,
                'term_levels': [TermLevel.PHRASE, TermLevel.SENTENCE]
            }, ...]
        """
        all_prompts = []
        prompt_metadata = []  # 각 프롬프트가 어느 feature의 것인지 추적

        for config in explainer_configs:
            explainer = ZeroShotAutoExplainer(llm_client=llm_client)
            explainer.init_feature(
                checkpoint=config["checkpoint"], layer=config["layer"], feature_index=config["feature_index"]
            )

            system_message = config.get("system_prompt", None)
            term_levels = config.get("term_levels", [TermLevel.PHRASE, TermLevel.SENTENCE])

            for term_level in term_levels:
                explainer.init_messages(system_message=system_message)
                prompt = explainer.make_explanation_prompt(term_level=term_level)
                all_prompts.append(prompt)
                prompt_metadata.append(
                    {
                        "checkpoint": config["checkpoint"],
                        "layer": config["layer"],
                        "feature_index": config["feature_index"],
                        "term_level": term_level,
                    }
                )

        # 모든 프롬프트를 한 번에 배치 처리
        all_responses = await llm_client.generate_batch_async(
            prompts=all_prompts,
            config=gen_config,
            system_prompt=system_message,
            extract_reasoning=False,
            max_concurrent=max_concurrent,
        )

        # 결과를 feature별로 재구성
        results = {}
        for metadata, response in zip(prompt_metadata, all_responses):
            key = (metadata["checkpoint"], metadata["layer"], metadata["feature_index"])
            if key not in results:
                results[key] = []
            results[key].append(response)

        return results

    def load_exist_explanations(self) -> List[ScoredExplanation]:
        explanation_path = (
            Path(settings.feature_explanation_dir) / self.checkpoint / str(self.layer) / f"{self.feature_index}.jsonl"
        )
        if not explanation_path.exists():
            return []

        explanations = FeatureExplanations.load_from_jsonl(explanation_path)
        return explanations.explanations

    def save_unscored_explanations(self, explanations: List[str]) -> None:
        explanation_path = (
            Path(settings.feature_explanation_dir) / self.checkpoint / str(self.layer) / f"{self.feature_index}.jsonl"
        )
        explanation_path.parent.mkdir(parents=True, exist_ok=True)

        feature_explanations = [ScoredExplanation(explanation=explanation, score=None) for explanation in explanations]
        feature_explanations = FeatureExplanations(explanations=feature_explanations)
        feature_explanations.save_to_jsonl(explanation_path)

    def generate_explanations(self):
        raise NotImplementedError

    def postprocess_explanations(self):
        raise NotImplementedError

    # def simulate_activation(self, explanation: str, samples: List[Dict], num_tokens: int = 10) -> List[Dict]:
    #     simulate_system_prompt = "..."
    #     simulate_prompt = "..."
    #     simulate_gen_confg = GenConfig()

    #     # NOTE: requires structured output
    #     simulation_result = self.llm_client.generate(
    #         simulate_prompt, system_prompt=simulate_system_prompt, config=simulate_gen_confg, extract_reasoning=False
    #     )
    #     simulation_data = json.loads(simulation_result)
    #     simulation_dict: List[TokenRecord] = [
    #         {
    #             "token_idx": record["token_idx"],
    #             "token_str": record["token_str"],
    #             "token_score": record["token_score"],
    #         }
    #         for record in simulation_data
    #     ]
    #     return simulation_dict
