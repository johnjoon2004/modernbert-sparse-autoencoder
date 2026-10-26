import os
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI, OpenAI
from src.utils.vllm.config import ConfigManager, GenConfig


class VLLMClient:
    def __init__(
        self,
        config_path: Optional[Path] = None,
        api_key: str | None = None,
        async_mode: bool = True,
    ):
        self.config_manager = ConfigManager(config_path)
        self.vllm_config = self.config_manager.get_vllm_config()
        api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.base_url = f"http://{self.vllm_config.host}:{self.vllm_config.port}/v1"
        self.async_mode = async_mode
        if async_mode:
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)
            self.model_id = None
            self.model_config = None
            self.is_reasoning = False
        else:
            self.client = OpenAI(base_url=self.base_url, api_key=api_key)
            self.model_id = self.client.models.list().data[0].id
            self.model_config = self.config_manager.get_model_config(self.model_id)
            self.is_reasoning = self.model_config.reasoning

    async def initialize_model_async(self):
        if self.async_mode and self.model_id is None:
            models = await self.client.models.list()
            self.model_id = models.data[0].id
            self.model_config = self.config_manager.get_model_config(self.model_id)
            self.is_reasoning = self.model_config.reasoning

    async def generate_async(
        self,
        prompt: str,
        config: Optional[GenConfig] = None,
        system_prompt: Optional[str] = None,
        extract_reasoning: bool = False,
    ) -> str | Dict[str, str]:
        if self.model_id is None:
            await self.initialize_model_async()

        if config is None:
            config = self.config_manager.get_gen_config()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            stop=config.stop,
        )
        content = response.choices[0].message.content
        if self.is_reasoning and extract_reasoning:
            return self._parse_reasoning_response(content)
        return content

    async def generate_batch_async(
        self,
        prompts: List[str],
        config: Optional[GenConfig] = None,
        system_prompt: Optional[str] = None,
        extract_reasoning: bool = False,
        max_concurrent: int = 4,
    ) -> List[str | Dict[str, str]]:
        if self.model_id is None:
            await self.initialize_model_async()

        if config is None:
            config = self.config_manager.get_gen_config()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(prompt):
            async with semaphore:
                try:
                    return await self.generate_async(
                        prompt=prompt,
                        config=config,
                        system_prompt=system_prompt,
                        extract_reasoning=extract_reasoning,
                    )
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
                    return None

        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def generate(
        self,
        prompt: str,
        config: Optional[GenConfig] = None,
        system_prompt: Optional[str] = None,
        extract_reasoning: bool = False,
    ) -> str | Dict[str, str]:
        """
        Returns:
            str: Non-reasoning
            Dict[str, str]: Reasoning model {'reasoning': ..., 'answer': ..., 'full': ...}
        """
        if config is None:
            config = self.config_manager.get_gen_config()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_id,  # matched with `--served-model-name`
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            stop=config.stop,
        )
        content = response.choices[0].message.content

        # reasoning response need parsing
        if self.is_reasoning and extract_reasoning:
            return self._parse_reasoning_response(content)

        return content

    def generate_batch_parallel(
        self,
        prompts: List[str],
        config: Optional[GenConfig] = None,
        system_prompt: Optional[str] = None,
        extract_reasoning: bool = True,
        max_workers: int = 4,
    ) -> List[str | Dict[str, str]]:
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate, prompt, config, system_prompt, extract_reasoning): idx
                for idx, prompt in enumerate(prompts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"❌ Prompt {idx} failed: {e}")
                    results[idx] = None

        return results

    def _parse_reasoning_response(self, content: str) -> Dict[str, str]:
        """
        Parse format:
        <think>
            <|reasoning|>
        </think>
            <|answer|>
        """
        reasoning_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            answer = content.split("</think>", 1)[1].strip() if "</think>" in content else ""
            return {
                "reasoning": reasoning,
                "answer": answer,
                "full": content,
            }
        return {
            "reasoning": "",
            "answer": content,
            "full": content,
        }
