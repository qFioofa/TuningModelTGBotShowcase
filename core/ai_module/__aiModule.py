import torch
from enum import Enum
from pathlib import Path
from typing import Optional
from peft import PeftModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH_DEFAULT: Path = Path("./.trained_models")
BASE_MODEL_NAME_DEFAULT: str = "unsloth/Phi-3-mini-4k-instruct"
MAX_SEQ_LENGTH_DEFAULT: int = 2048

ERROR_MODEL_LOAD_FAILED: str = "Failed to load the base model."
ERROR_ADAPTER_LOAD_FAILED: str = "Failed to load the LoRA adapter."
ERROR_TOKENIZER_LOAD_FAILED: str = "Failed to load the tokenizer."
ERROR_INFERENCE_FAILED: str = "Model inference failed."
ERROR_NO_GPU_UNSLOTH: str = "Cannot initialize Unsloth model without a GPU."

class AiChatRoles(Enum):
    USER : str = "user"
    SYSTEM : str = "system"
    ASSISTANT : str = "assistant"

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
        self._use_unsloth: bool = False
        self._text_generator: Optional[object] = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        if torch.cuda.is_available():
            self._load_model_and_tokenizer_with_unsloth()
        else:
            self._load_model_and_tokenizer_cpu_fallback()

    def _load_model_and_tokenizer_with_unsloth(self) -> None:
        self._use_unsloth = True
        try:
            from unsloth import FastLanguageModel
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)

            original_model, _ = FastLanguageModel.from_pretrained(
                model_name=self._model_path if (self._model_path / "config.json").exists() else self._base_model_name,
                max_seq_length=self._max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            self._model = FastLanguageModel.get_peft_model(
                original_model,
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
            FastLanguageModel.for_inference(self._model)

        except Exception as e:
            raise RuntimeError(f"{ERROR_MODEL_LOAD_FAILED} Details: {e}") from e

    def _load_model_and_tokenizer_cpu_fallback(self) -> None:
        print("Предупреждение: GPU не найден. Используется альтернативная реализация без unsloth.")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)

            base_model = AutoModelForCausalLM.from_pretrained(
                self._base_model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
            )

            self._model = PeftModel.from_pretrained(base_model, self._model_path)
            self._model.eval()

            self._text_generator = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device_map="cpu"
            )

        except Exception as e:
            raise RuntimeError(f"{ERROR_MODEL_LOAD_FAILED} Details: {e}") from e


    def get_response(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        if not self._model or not self._tokenizer:
            raise RuntimeError(ERROR_INFERENCE_FAILED)

        if self._use_unsloth:
            return self._get_response_unsloth(messages, max_new_tokens, temperature, top_p)
        else:
            return self._get_response_cpu_fallback(messages, max_new_tokens, temperature, top_p)

    def _get_response_unsloth(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        if not torch.cuda.is_available():
             raise RuntimeError(ERROR_NO_GPU_UNSLOTH)

        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
        )

        response: str = self._tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        return response

    def _get_response_cpu_fallback(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        if not self._text_generator:
            raise RuntimeError("Pipeline для генерации не инициализирован для CPU версии.")

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self._text_generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        generated_text = outputs[0]['generated_text']
        response = generated_text[len(prompt):]
        return response
