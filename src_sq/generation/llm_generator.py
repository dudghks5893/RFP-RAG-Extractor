# src/generation/llm_generator.py
#
# Hugging Face CausalLM 기반 RAG 답변 생성 모듈입니다.
#
# 주요 역할:
# - tokenizer/model 로드
# - Qwen Instruct chat template 적용
# - RFP RAG prompt 기반 답변 생성
# - Qwen3 계열의 <think>...</think> reasoning 출력 제거
# - latency, token count 기록
# - GPU 메모리 정리 함수 제공
#
# 사용 예:
#
# from src.generation.llm_generator import LLMGenerator
#
# generator = LLMGenerator(
#     model_name="Qwen/Qwen2.5-1.5B-Instruct",
#     max_new_tokens=512,
#     do_sample=False,
# ).load()
#
# result = generator.generate_from_retrieved_chunks(
#     question="이 사업의 기대효과는 무엇인가요?",
#     retrieved_chunks=retrieved_chunks,
# )

from __future__ import annotations

import gc
import re
import time
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.generation.prompts import build_rfp_rag_messages
from src.utils.progress_utils import log_step


class LLMGenerator:
    """
    Hugging Face CausalLM 기반 답변 생성기입니다.

    현재 베이스라인 모델:
    - Qwen/Qwen2.5-1.5B-Instruct

    Parameters
    ----------
    model_name:
        Hugging Face 모델 이름입니다.

    max_new_tokens:
        생성할 최대 토큰 수입니다.

    temperature:
        sampling 사용 시 temperature 값입니다.

    do_sample:
        True이면 sampling 기반 생성, False이면 deterministic generation입니다.

    trust_remote_code:
        Hugging Face 모델 로드 시 trust_remote_code 사용 여부입니다.

    torch_dtype:
        모델 로드 dtype입니다.
        None이면 CUDA 사용 가능 시 float16, 아니면 float32를 사용합니다.

    device_map:
        transformers의 device_map 옵션입니다.
        기본값 "auto"입니다.

    low_cpu_mem_usage:
        모델 로드 시 CPU 메모리 사용량 절약 옵션입니다.

    prompt_type:
        src/generation/prompts.py의 prompt_type입니다.
        - "default"
        - "strict"

    max_chars_per_chunk:
        prompt에 넣을 각 청크의 최대 문자 수입니다.
        None이면 청크 전체를 넣습니다.

    include_metadata:
        prompt context에 doc_id, chunk_id, section_title 등 메타데이터 포함 여부입니다.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str | Dict[str, Any] | None = "auto",
        low_cpu_mem_usage: bool = True,
        prompt_type: str = "default",
        max_chars_per_chunk: Optional[int] = None,
        include_metadata: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage

        self.prompt_type = prompt_type
        self.max_chars_per_chunk = max_chars_per_chunk
        self.include_metadata = include_metadata

        self.tokenizer = None
        self.model = None

    # ---------------------------------------------------------
    # 모델 로드
    # ---------------------------------------------------------
    def _resolve_torch_dtype(self) -> torch.dtype:
        """
        torch_dtype이 명시되지 않은 경우 실행 환경에 따라 dtype을 결정합니다.

        CUDA 사용 가능:
        - float16

        CPU/MPS:
        - float32

        Returns
        -------
        torch.dtype
        """
        if self.torch_dtype is not None:
            return self.torch_dtype

        if torch.cuda.is_available():
            return torch.float16

        return torch.float32

    def load_tokenizer(self) -> "LLMGenerator":
        """
        tokenizer를 로드합니다.
        """
        with log_step(f"Tokenizer load: {self.model_name}"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )

        return self

    def load_model(self) -> "LLMGenerator":
        """
        causal language model을 로드합니다.
        """
        dtype = self._resolve_torch_dtype()

        with log_step(f"LLM model load: {self.model_name}"):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
            )

        self.model.eval()

        return self

    def load(self) -> "LLMGenerator":
        """
        tokenizer와 model을 모두 로드합니다.

        Returns
        -------
        LLMGenerator
            자기 자신을 반환합니다.
        """
        self.load_tokenizer()
        self.load_model()
        return self

    def _ensure_loaded(self) -> None:
        """
        tokenizer와 model이 로드되어 있는지 확인합니다.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer가 로드되지 않았습니다. "
                "먼저 generator.load_tokenizer() 또는 generator.load()를 호출하세요."
            )

        if self.model is None:
            raise RuntimeError(
                "LLM model이 로드되지 않았습니다. "
                "먼저 generator.load_model() 또는 generator.load()를 호출하세요."
            )

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """
        Qwen3 계열 모델이 출력하는 <think>...</think> 구간과
        태그 없이 출력된 영어 reasoning prefix를 제거합니다.
        """
        text = str(text or "")

        # 1. 정상적으로 닫힌 <think>...</think> 블록 제거
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # 2. 단독으로 남은 <think>, </think> 태그 제거
        text = re.sub(
            r"</?think>",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        # 3. Qwen3가 태그 없이 영어 reasoning으로 시작하는 경우 제거
        reasoning_prefix_patterns = [
            r"^Okay,\s*let'?s.*?(?=\n\s*(?:[-•○●]|\d+[.)]|[가-힣]))",
            r"^Okay,.*?(?=\n\s*(?:[-•○●]|\d+[.)]|[가-힣]))",
            r"^Let'?s.*?(?=\n\s*(?:[-•○●]|\d+[.)]|[가-힣]))",
            r"^I need to.*?(?=\n\s*(?:[-•○●]|\d+[.)]|[가-힣]))",
            r"^The user is asking.*?(?=\n\s*(?:[-•○●]|\d+[.)]|[가-힣]))",
            r"^We need to.*?(?=\n\s*(?:[-•○●]|\d+[.)]|[가-힣]))",
        ]

        for pattern in reasoning_prefix_patterns:
            text = re.sub(
                pattern,
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()

        return text.strip()

    # ---------------------------------------------------------
    # device 관련
    # ---------------------------------------------------------
    @property
    def device(self):
        """
        모델 입력 tensor를 올릴 device를 반환합니다.

        device_map='auto'로 여러 device에 분산된 경우에도,
        일반적으로 첫 번째 parameter의 device에 입력을 올리면 됩니다.
        """
        self._ensure_loaded()
        return next(self.model.parameters()).device

    # ---------------------------------------------------------
    # prompt / tokenization
    # ---------------------------------------------------------
    def build_messages(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        RFP RAG용 chat messages를 생성합니다.
        """
        return build_rfp_rag_messages(
            question=question,
            retrieved_chunks=retrieved_chunks,
            prompt_type=self.prompt_type,
            max_chars_per_chunk=self.max_chars_per_chunk,
            include_metadata=self.include_metadata,
        )

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        tokenizer의 chat template을 적용해 prompt text를 생성합니다.
    
        Qwen3 계열 모델은 enable_thinking=False를 전달하면
        thinking mode를 비활성화할 수 있습니다.
        Qwen2.5 등 해당 인자를 지원하지 않는 모델도 있으므로
        TypeError 발생 시 기존 방식으로 fallback합니다.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer가 로드되지 않았습니다.")
    
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    def tokenize_prompt(self, prompt_text: str) -> Dict[str, Any]:
        """
        prompt text를 모델 입력 tensor로 변환합니다.
        """
        self._ensure_loaded()

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )

        # device_map='auto'인 경우에도 입력은 모델의 첫 device로 보냅니다.
        inputs = inputs.to(self.device)

        return inputs

    # ---------------------------------------------------------
    # 생성
    # ---------------------------------------------------------
    def _build_generation_kwargs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        model.generate에 전달할 generation kwargs를 구성합니다.
        """
        kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # do_sample=False일 때 temperature를 넘기면 일부 transformers 버전에서 warning이 납니다.
        if self.do_sample:
            kwargs["temperature"] = self.temperature

        return kwargs

    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        return_prompt: bool = False,
    ) -> Dict[str, Any]:
        """
        chat messages를 입력받아 답변을 생성합니다.

        Parameters
        ----------
        messages:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            형태의 chat messages입니다.

        return_prompt:
            True이면 prompt_text도 반환합니다.
            디버깅에는 유용하지만 저장 파일이 커질 수 있으므로 기본값은 False입니다.

        Returns
        -------
        Dict[str, Any]
            {
                "response": "...",
                "input_tokens": 1234,
                "output_tokens": 128,
                "total_tokens": 1362,
                "generation_latency_sec": 1.23,
                "prompt_text": "..."  # return_prompt=True일 때만
            }
        """
        self._ensure_loaded()

        prompt_text = self.apply_chat_template(
            messages=messages,
            add_generation_prompt=True,
        )

        inputs = self.tokenize_prompt(prompt_text)

        input_tokens = int(inputs["input_ids"].shape[-1])

        generation_kwargs = self._build_generation_kwargs(inputs)

        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)

        generation_latency_sec = time.perf_counter() - start_time

        output_tokens = int(outputs.shape[-1] - inputs["input_ids"].shape[-1])

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

        response_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        # Qwen3 계열 모델이 출력하는 <think>...</think> reasoning 구간 제거
        response_text = self._strip_think_tags(response_text)

        result = {
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "generation_latency_sec": generation_latency_sec,
        }

        if return_prompt:
            result["prompt_text"] = prompt_text

        return result

    def generate_from_retrieved_chunks(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        return_prompt: bool = False,
    ) -> Dict[str, Any]:
        """
        질문과 검색된 청크를 입력받아 RAG 답변을 생성합니다.

        Parameters
        ----------
        question:
            사용자 질문입니다.

        retrieved_chunks:
            retriever.retrieve() 결과입니다.

        return_prompt:
            True이면 prompt_text도 반환합니다.

        Returns
        -------
        Dict[str, Any]
            generate_from_messages()와 동일한 구조입니다.
        """
        messages = self.build_messages(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        return self.generate_from_messages(
            messages=messages,
            return_prompt=return_prompt,
        )

    # ---------------------------------------------------------
    # 메모리 정리
    # ---------------------------------------------------------
    def unload(self) -> None:
        """
        tokenizer와 model 객체를 해제하고 CUDA cache를 비웁니다.

        JupyterHub/GPU 환경에서 OOM 방지를 위해 사용합니다.
        """
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("LLMGenerator unloaded and CUDA cache cleared.")

    # ---------------------------------------------------------
    # 상태 출력
    # ---------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """
        현재 generator 설정과 로드 상태를 dict로 반환합니다.
        """
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "trust_remote_code": self.trust_remote_code,
            "device_map": self.device_map,
            "prompt_type": self.prompt_type,
            "max_chars_per_chunk": self.max_chars_per_chunk,
            "include_metadata": self.include_metadata,
            "tokenizer_loaded": self.tokenizer is not None,
            "model_loaded": self.model is not None,
            "device": str(self.device) if self.model is not None else None,
        }


def load_llm_generator(
    model_name: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    trust_remote_code: bool = True,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: str | Dict[str, Any] | None = "auto",
    low_cpu_mem_usage: bool = True,
    prompt_type: str = "default",
    max_chars_per_chunk: Optional[int] = None,
    include_metadata: bool = True,
) -> LLMGenerator:
    """
    LLMGenerator를 생성하고 바로 load까지 수행하는 편의 함수입니다.
    """
    generator = LLMGenerator(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        prompt_type=prompt_type,
        max_chars_per_chunk=max_chars_per_chunk,
        include_metadata=include_metadata,
    )

    generator.load()

    return generator