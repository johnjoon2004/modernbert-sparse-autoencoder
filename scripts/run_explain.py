import asyncio
from asyncio import Semaphore
from pathlib import Path
from typing import List

import pandas as pd
from src.auto_explainer.explainer import ZeroShotAutoExplainer
from src.auto_explainer.prompts import TermLevel
from src.features import load_score_data
from src.global_settings import settings
from src.utils.vllm.client import VLLMClient
from src.utils.vllm.config import GenConfig
from tqdm.asyncio import tqdm_asyncio as tqdm

META_DIR = Path(settings.feature_meta_dir)
RECORD_DIR = Path(settings.feature_record_dir)
EXPLANATION_DIR = Path(settings.feature_explanation_dir)


def is_explanation_exist(checkpoint: str, layer: int, feature_index: int) -> bool:
    explanation_path = Path(settings.feature_explanation_dir) / checkpoint / str(layer) / f"{feature_index}.jsonl"
    return explanation_path.exists()


async def explain_feature_async(
    llm_client: VLLMClient,
    gen_config: GenConfig,
    checkpoint: str,
    layer: int,
    feature_index: int,
    semaphore: Semaphore,
    skip_existing: bool = True,
):
    async with semaphore:
        try:
            if skip_existing and is_explanation_exist(checkpoint, layer, feature_index):
                return

            explainer = ZeroShotAutoExplainer(llm_client=llm_client)
            explainer.init_feature(checkpoint=checkpoint, layer=layer, feature_index=feature_index)

            # explanation_term = await explainer.generate_explanation_async(gen_config, term_level=TermLevel.PHRASE)
            # explanation_sent = await explainer.generate_explanation_async(gen_config, term_level=TermLevel.SENTENCE)

            explanation_term, explanation_sent = await asyncio.gather(
                explainer.generate_explanation_async(gen_config, term_level=TermLevel.PHRASE),
                explainer.generate_explanation_async(gen_config, term_level=TermLevel.SENTENCE),
            )
            explainer.save_unscored_explanations(explanations=[explanation_term, explanation_sent])

        except Exception as e:
            print(f"❌ Error explaining feature {feature_index} at layer {layer} of {checkpoint}: {e}")


def filter_live_features(checkpoint: str, layer: int, min_active_sample_count: int = 200) -> List[int]:
    scores = load_score_data(checkpoint, layer, metric_name="total_post")
    scores = pd.DataFrame(scores)
    active_sample_counts = (scores > 0).sum(axis=0)
    live_feature_indices = active_sample_counts[active_sample_counts >= min_active_sample_count].index.tolist()
    return live_feature_indices


async def main_async():
    llm_client = VLLMClient(async_mode=True)
    gen_config = GenConfig(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
    )

    checkpoints = ["f960_k8", "f3072_k256", "f6144_k256", "f6144_k1024"]
    # checkpoints = ["f3072_k8", "f6144_k32"]
    checkpoints = [f"{ckpt}_alpha" for ckpt in checkpoints]
    layers = [21, 11, 1, 6, 15]

    live_feature_index_map = {}
    total = 0
    for checkpoint in checkpoints:
        for layer in layers:
            live_feature_indices = filter_live_features(
                checkpoint=checkpoint, layer=layer, min_active_sample_count=200
            )
            live_feature_index_map[(checkpoint, layer)] = live_feature_indices
            total += len(live_feature_indices)

    skip_existing = True

    max_concurrent_features = 16
    semaphore = Semaphore(max_concurrent_features)

    tasks = []
    for (checkpoint, layer), live_feature_indices in live_feature_index_map.items():
        for feature_index in live_feature_indices:
            task = explain_feature_async(
                llm_client=llm_client,
                gen_config=gen_config,
                checkpoint=checkpoint,
                layer=layer,
                feature_index=feature_index,
                semaphore=semaphore,
                skip_existing=skip_existing,
            )
            tasks.append(task)

    await tqdm.gather(*tasks, desc="Explaining Features...", total=total)

    # ---------------------------------------------------------------------------------------------------------------
    # # Batch generation version (DEPRECATED)
    # ---------------------------------------------------------------------------------------------------------------
    # explainer_configs = []
    # for (checkpoint, layer), live_feature_indices in live_feature_index_map.items():
    #     for feature_index in live_feature_indices:
    #         explainer_configs.append(
    #             {
    #                 "checkpoint": checkpoint,
    #                 "layer": layer,
    #                 "feature_index": feature_index,
    #                 "term_levels": [TermLevel.PHRASE, TermLevel.SENTENCE],
    #             }
    #         )
    #
    # batch_size = 1
    # for i in tqdm(range(0, len(explainer_configs), batch_size), desc="Processing batches"):
    #     batch = explainer_configs[i : i + batch_size]
    #     results = await ZeroShotAutoExplainer.batch_explain_features(
    #         explainer_configs=batch,
    #         llm_client=llm_client,
    #         gen_config=gen_config,
    #         max_concurrent=4,
    #     )
    #     explainer = ZeroShotAutoExplainer(llm_client=llm_client)
    #     for (checkpoint, layer, feature_index), explanations in results.items():
    #         try:
    #             explainer.init_feature(checkpoint=checkpoint, layer=layer, feature_index=feature_index)
    #             explainer.save_unscored_explanations(explanations=explanations)
    #         except Exception as e:
    #             print(
    #                 f"❌ Error saving explanations for feature {feature_index} at layer {layer} of {checkpoint}: {e}"
    #             )
    # ---------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(main_async())
