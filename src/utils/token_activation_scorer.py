from typing import Dict, List, Optional, Tuple

import torch
from src.global_settings import settings
from src.sae import SparseAutoencoder
from transformers import AutoTokenizer, PreTrainedModel

LM_CKPT = settings.lm_weights_path
MAX_SEQ_LEN = settings.prev_seq_len


class TokenActivationScorer:
    def __init__(
        self,
        model: SparseAutoencoder,
        tokenizer: Optional[AutoTokenizer] = None,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = (
            AutoTokenizer.from_pretrained(settings.lm_weights_path)
            if tokenizer is None
            else tokenizer
        )

    def compute_token_scores(
        self,
        text: str,
        feature_idx: int,
        layer_idx: int,
        lm_model: PreTrainedModel | None = None,
        max_length: int = MAX_SEQ_LEN,
        exclude_special_tokens: bool = True,
        return_pre_activation: bool = False,
    ) -> Tuple[List[Dict[str, any]]]:
        if lm_model is None:
            lm_model = PreTrainedModel.from_pretrained(LM_CKPT).to(self.device)

        lm_model.eval()

        encoded = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        with torch.no_grad():
            outputs = lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            activation = outputs.hidden_states[layer_idx]

            token_scores, summary = self.compute_token_scores_from_activation(
                activation=activation,
                feature_idx=feature_idx,
                tokens=tokens,
                attention_mask=attention_mask,
                exclude_special_tokens=exclude_special_tokens,
                return_pre_activation=return_pre_activation,
            )

        return token_scores, summary

    def compute_token_scores_from_activation(
        self,
        activation: torch.Tensor,
        feature_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        tokens: Optional[List[str]] = None,
        exclude_special_tokens: bool = True,
        return_pre_activation: bool = False,
    ) -> Tuple[List[Dict[str, any]], Dict[str, float]]:
        """
        LM의 레이어 출력(activation)으로부터 특정 feature의 토큰별 활성화 점수를 계산

        Args:
            activation: LM의 특정 레이어 출력 (shape: [batch_size, seq_len, input_dim] or [seq_len, input_dim])
            feature_idx: 분석할 feature의 인덱스
            attention_mask: attention mask (shape: [batch_size, seq_len] or [seq_len])
            tokens: 토큰 문자열 리스트 (없으면 인덱스만 표시)
            exclude_special_tokens: 특수 토큰 제외 여부
            return_pre_activation: pre-activation 점수도 반환할지 여부

        Returns:
            token_scores: 각 토큰별 점수 정보 리스트
            summary: 요약 통계
                - total_score_pre: pre-activation의 합
                - total_score_post: post-activation의 합
                - avg_score_pre: pre-activation의 평균
                - avg_score_post: post-activation의 평균
                - num_active_tokens: 활성화된 토큰 수 (post-activation > 0)
        """
        # 입력 차원 확인 및 조정
        if activation.dim() == 2:
            activation = activation.unsqueeze(
                0
            )  # [seq_len, input_dim] -> [1, seq_len, input_dim]

        batch_size, seq_len, input_dim = activation.shape

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=activation.device)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # 특수 토큰 제외를 위한 마스크 생성
        attn_mask = attention_mask.clone()
        if exclude_special_tokens:
            # [CLS] 토큰 제외
            attn_mask[:, 0] = 0
            # [SEP] 토큰 제외
            seq_len_actual = attention_mask.sum(dim=1).long()
            attn_mask[torch.arange(attn_mask.size(0)), seq_len_actual - 1] = 0

        activation = activation.to(self.device)
        attn_mask = attn_mask.to(self.device)

        with torch.no_grad():
            # SAE를 통한 feature activation 계산
            processed_input = self.model._preprocess_input(activation)
            pre_activations = self.model.encoder(
                processed_input
            )  # [batch_size, seq_len, feature_dim]
            post_activations = self.model._apply_activation(pre_activations)

            # 특정 feature 추출
            feature_pre = pre_activations[0, :, feature_idx]  # [seq_len]
            feature_post = post_activations[0, :, feature_idx]  # [seq_len]
            mask = attn_mask[0]  # [seq_len]

            # 마스킹 적용
            feature_pre_masked = feature_pre * mask
            feature_post_masked = feature_post * mask

            # 요약 통계 계산
            num_tokens = mask.sum().item()
            num_tokens = max(num_tokens, 1)  # division by zero 방지

            total_score_pre = feature_pre_masked.sum().item()
            total_score_post = feature_post_masked.sum().item()
            avg_score_pre = total_score_pre / num_tokens
            avg_score_post = total_score_post / num_tokens
            num_active_tokens = (feature_post_masked > 0).sum().item()

            # 토큰별 점수 리스트 생성
            token_scores = []
            special_tokens = {r"<s>", r"<\\s>", r"<pad>", "[CLS]", "[SEP]", "[PAD]"}

            for idx in range(seq_len):
                if mask[idx] == 0:
                    continue

                token_str = tokens[idx] if tokens is not None else f"[{idx}]"

                # 특수 토큰 제외 옵션이 켜져있고 특수 토큰이면 스킵
                if exclude_special_tokens and token_str in special_tokens:
                    continue

                score_info = {
                    "token_idx": idx,
                    "token_str": token_str,
                    "token_score": round(feature_post[idx].item(), 4),
                }

                if return_pre_activation:
                    score_info["token_score_pre"] = round(feature_pre[idx].item(), 4)

                token_scores.append(score_info)

            summary = {
                "total_score_pre": round(total_score_pre, 4),
                "total_score_post": round(total_score_post, 4),
                "avg_score_pre": round(avg_score_pre, 4),
                "avg_score_post": round(avg_score_post, 4),
                "num_active_tokens": num_active_tokens,
            }

        return token_scores, summary

    def get_top_activating_tokens(
        self,
        activation: torch.Tensor,
        feature_idx: int,
        top_k: int = 10,
        attention_mask: Optional[torch.Tensor] = None,
        tokens: Optional[List[str]] = None,
        exclude_special_tokens: bool = True,
    ) -> List[Dict[str, any]]:
        """
        특정 feature에 대해 가장 높은 activation 점수를 가진 상위 k개 토큰 반환

        Args:
            activation: LM의 특정 레이어 출력
            feature_idx: 분석할 feature의 인덱스
            top_k: 반환할 상위 토큰 개수
            attention_mask: attention mask
            tokens: 토큰 문자열 리스트
            exclude_special_tokens: 특수 토큰 제외 여부

        Returns:
            상위 k개 토큰의 점수 정보 리스트 (점수 내림차순 정렬)
        """
        token_scores, _ = self.compute_token_scores_from_activation(
            activation=activation,
            feature_idx=feature_idx,
            attention_mask=attention_mask,
            tokens=tokens,
            exclude_special_tokens=exclude_special_tokens,
            return_pre_activation=False,
        )

        # post-activation 점수 기준 내림차순 정렬
        sorted_tokens = sorted(
            token_scores, key=lambda x: x["token_score"], reverse=True
        )

        return sorted_tokens[:top_k]

    def compute_multiple_features(
        self,
        activation: torch.Tensor,
        feature_indices: List[int],
        attention_mask: Optional[torch.Tensor] = None,
        tokens: Optional[List[str]] = None,
        exclude_special_tokens: bool = True,
    ) -> Dict[int, Tuple[List[Dict[str, any]], Dict[str, float]]]:
        """
        여러 feature에 대해 동시에 토큰별 활성화 점수 계산

        Args:
            activation: LM의 특정 레이어 출력
            feature_indices: 분석할 feature 인덱스 리스트
            attention_mask: attention mask
            tokens: 토큰 문자열 리스트
            exclude_special_tokens: 특수 토큰 제외 여부

        Returns:
            Dict[feature_idx, (token_scores, summary)]: feature별 결과
        """
        results = {}

        for feat_idx in feature_indices:
            token_scores, summary = self.compute_token_scores_from_activation(
                activation=activation,
                feature_idx=feat_idx,
                attention_mask=attention_mask,
                tokens=tokens,
                exclude_special_tokens=exclude_special_tokens,
                return_pre_activation=True,
            )
            results[feat_idx] = (token_scores, summary)

        return results
