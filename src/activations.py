import glob
import json
import math
import os
import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from src.global_settings import settings
from src.utils.seed import seed_worker
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedModel

DEFAULT_DTYPE = getattr(torch, settings.train_dtype)


class ActivationCache:
    """
    Usage Example:
        model = AutoModelForSequenceClassification.from_pretrained(...)
        tokenizer = AutoTokenizer.from_pretrained(...)

        cache = ActivationCache(model)
        inputs = tokenizer("...", return_tensors="pt")
        target_layers = [0, 5, 10]

        with cache.capture(inputs, layer_indices=target_layers) as activations:
            # 'activations'는 레이어 인덱스를 키로 갖는 dict
            activations_layer_5 = activations[5]
            print(activations_layer_5.shape)
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.model.eval()
        self._captured_activations: Dict[int, torch.Tensor] = {}
        self._hook_handles = []

    def _get_hook(self, layer_idx: int):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor):
            # Transformer layer의 입력 튜플 중 첫 번째 요소가 residual stream
            # 계산 그래프에서 분리하고 CPU로 이동
            self._captured_activations[layer_idx] = inputs[0].detach().cpu()

        return hook

    def _get_target_module(self, layer_idx: int) -> nn.Module:
        # 모델 구조에 따라 적절히 수정 필요
        # 현재는 ModernBert 아키텍처에 맞춤
        return self.model.model.layers[layer_idx]

    @contextmanager
    def capture(self, inputs: Dict[str, torch.Tensor] | BatchEncoding, layer_indices: List[int]):
        """
        지정된 레이어들의 residual stream을 캡처하는 context manager.
        `with` 블록이 시작될 때 hook을 등록하고, 끝날 때 자동으로 제거.

        Args:
            inputs (Dict[str, torch.Tensor] | BatchEncoding): 모델에 입력될 토큰화된 텐서
            layer_indices (List[int]): 캡처할 레이어 인덱스 리스트
        """

        layer_indices = sorted(set(layer_indices))

        try:
            # 이전 실행에서 남은 데이터 초기화
            self._captured_activations.clear()
            self._hook_handles.clear()

            # 지정된 각 레이어에 대해 hook 등록
            for idx in layer_indices:
                module = self._get_target_module(idx)
                handle = module.register_forward_hook(self._get_hook(idx))
                self._hook_handles.append(handle)

            # 모델 forward pass (실행 과정에서 hook이 activation을 자동 캡처)
            with torch.no_grad():
                self.model(**inputs)

            # 추출된 activation(dict)을 with 블록으로 전달
            yield self._captured_activations

        finally:
            # hook 제거
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()
            self._captured_activations.clear()


class ActivationDataset(IterableDataset):
    """
    Usage Example:
        g = torch.Generator()
        g.manual_seed(42)
        activation_dataset = ActivationDataset(data_dir="path/to/activations/layer_k")
        activation_loader = DataLoader(activation_dataset, batch_size=32, num_workers=1, worker_init_fn=seed_worker, generator=g)

    Args:
        data_dir (str): 'batch_*.pt' 파일들이 저장된 디렉토리 경로
        dtype (torch.dtype): 로드할 때 사용할 데이터 타입. float16으로 로드하기 전에 SAE 모델이 half-precision 학습 가능한지 확인해야 함. Defaults to torch.float32.
    """

    def __init__(
        self,
        data_dir: str,
        dtype: torch.dtype = DEFAULT_DTYPE,
        file_paths: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.dtype = dtype

        # Load if 'metadata.json' exists, otherwise create
        self.metadata = ActivationDataset.check_metadata(self.data_dir)

        if file_paths:
            self.file_paths = file_paths
        else:
            # self.file_paths = sorted(glob.glob(os.path.join(data_dir, "batch_*.pt")))
            self.file_paths = sorted([os.path.join(self.data_dir, f) for f in self.metadata.keys()])

        if not self.file_paths:
            raise FileNotFoundError(f"No activation files found in {data_dir}")

        self.total_samples = sum(len(self.metadata[os.path.basename(p)]) for p in self.file_paths)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        """
        Yields tuple as (tensor, attn_mask, index).
        """
        worker_info = get_worker_info()
        if worker_info is None:
            start_idx, end_idx = 0, len(self.file_paths)
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            chunk_size = int(math.ceil(len(self.file_paths) / num_workers))
            start_idx = worker_id * chunk_size
            end_idx = min(start_idx + chunk_size, len(self.file_paths))

        for i in range(start_idx, end_idx):
            filepath = self.file_paths[i]
            try:
                data_dict = torch.load(filepath, map_location="cpu")

                batch_ts = data_dict["activations"].to(dtype=self.dtype)
                batch_attn_mask = data_dict["attention_mask"]
                batch_ids = data_dict["indices"]

                num_samples_in_file = batch_ts.shape[0]
                for j in range(num_samples_in_file):
                    # NOTE: the order of Tuple is (tensor, attention_mask, index)
                    yield batch_ts[j], batch_attn_mask[j], batch_ids[j]

            finally:
                if "data_dict" in locals():
                    del data_dict

    @staticmethod
    def check_metadata(data_dir: str) -> Dict[str, List[int]]:
        """
        Scans the data_dir for 'batch_*.pt' files and creates a metadata.json file, recording each file's name and the indices of samples.
        """

        metadata_path = os.path.join(data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)

        print(f"Metadata file not found. Creating new metadata at {metadata_path}...")

        file_paths = sorted(glob.glob(os.path.join(data_dir, "batch_*.pt")))
        if not file_paths:
            print(f"No 'batch_*.pt' files found in {data_dir}. Cannot create metadata.")
            metadata = {}
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            return metadata

        metadata = {}
        for filepath in tqdm(file_paths, desc="Scanning files to create metadata..."):
            try:
                data_dict = torch.load(filepath, map_location="cpu")
                indices = data_dict["indices"].tolist()
                filename = os.path.basename(filepath)
                metadata[filename] = indices
            except Exception as e:
                print(f"Warning: Could not process file {filepath}. Error: {e}")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return metadata


class ActivationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        split_ratios: Tuple[float, float, float] = (0.9, 0.1, 0.0),
        dtype: str | torch.dtype = "float16",
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.num_workers = num_workers
        self.seed = seed

        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        all_files = sorted(glob.glob(os.path.join(self.data_dir, "batch_*.pt")))
        if not all_files:
            raise FileNotFoundError(f"No activation files found in {self.data_dir}")

        random.Random(self.seed).shuffle(all_files)

        n_files = len(all_files)
        n_train = int(n_files * self.split_ratios[0])
        n_val = int(n_files * self.split_ratios[1])
        n_test = n_files - n_train - n_val

        train_files = all_files[:n_train]
        val_files = all_files[n_train : n_train + n_val]
        test_files = all_files[n_train + n_val :]

        if stage == "fit" or stage is None:
            self.train_dataset = ActivationDataset(
                data_dir=self.data_dir, file_paths=train_files, **self.common_dataset_params()
            )
            self.val_dataset = ActivationDataset(
                data_dir=self.data_dir, file_paths=val_files, **self.common_dataset_params()
            )
        if stage == "test" or stage is None:
            self.test_dataset = ActivationDataset(
                data_dir=self.data_dir, file_paths=test_files, **self.common_dataset_params()
            )

    def common_dataset_params(self):
        return {"dtype": self.dtype}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            worker_init_fn=seed_worker if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            worker_init_fn=seed_worker if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            worker_init_fn=seed_worker if self.num_workers > 0 else None,
        )
