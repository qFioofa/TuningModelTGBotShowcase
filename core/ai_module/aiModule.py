import torch
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from core.store.ai_model_settings import AiLevel

MODEL_PATH_DEFAULT: Path = Path("./fine_tuned_model_conflict_export")
BASE_MODEL_NAME_DEFAULT: str = "unsloth/Phi-3-mini-4k-instruct"
MAX_SEQ_LENGTH_DEFAULT: int = 2048

ERROR_MODEL_LOAD_FAILED: str = "Failed to load the base model."
ERROR_ADAPTER_LOAD_FAILED: str = "Failed to load the LoRA adapter."
ERROR_TOKENIZER_LOAD_FAILED: str = "Failed to load the tokenizer."
ERROR_INFERENCE_FAILED: str = "Model inference failed."

class AiModel:
    def __init__(
        self,
        model_path: Path = MODEL_PATH_DEFAULT,
        base_model_name: str = BASE_MODEL_NAME_DEFAULT,
        max_seq_length: int = MAX_SEQ_LENGTH_DEFAULT
    ) -> None:
        self._model_path: Path = model_path
        self._base_model_name: str = base_model_name
        self._max_seq_length: int = max_seq_length
        self._model: Optional[object] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            if self._tokenizer is None:
                raise RuntimeError(ERROR_TOKENIZER_LOAD_FAILED)
        except Exception as e:
            raise RuntimeError(f"{ERROR_TOKENIZER_LOAD_FAILED} Details: {e}") from e

        try:
            self._model, _ = FastLanguageModel.from_pretrained(
                model_name=self._base_model_name,
                max_seq_length=self._max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
        except Exception as e:
            raise RuntimeError(f"{ERROR_MODEL_LOAD_FAILED} Details: {e}") from e

        try:
            self._model = FastLanguageModel.get_peft_model(
                self._model,
                r=256,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=512,
                lora_dropout=0.1,
                bias="lora_only",
                use_gradient_checkpointing="unsloth",
                random_state=1025,
                use_rslora=True,
                loftq_config=None,
            )
            self._model.load_adapter(self._model_path)
        except Exception as e:
            raise RuntimeError(f"{ERROR_ADAPTER_LOAD_FAILED} Details: {e}") from e

        FastLanguageModel.for_inference(self._model)

    def get_response(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        if not self._model or not self._tokenizer:
            raise RuntimeError(ERROR_INFERENCE_FAILED)

        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = self._model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
        )

        response: str = self._tokenizer.batch_decode(outputs)[0]
        return response

class AiRouter:

    _rout : dict[AiLevel , AiModel]

    def __init__(self) -> None:
        self._rout = {}
        self._init_router()

    def _init_router(self) -> None:
        pass

    def get_response(self, model : AiLevel, user_input : list[dict[str, str]]) -> str:
        _model : AiModel = self._rout.get(model)

        if _model is None:
            raise RuntimeError(f"Model: {model.value} does not exist")

        response : str = _model.get_response(
            user_input
        )

        return response
