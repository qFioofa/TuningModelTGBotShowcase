from pathlib import Path
from typing import Optional, Literal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

MODEL_PATH_DEFAULT: Path = Path("./.trained_models")

"""
Доступные модели (LoRA fine-tuned):
- BroneBonBon/Conflict-Generator-Mistral
- BroneBonBon/Conflict-Generator-Phi

Базовые модели (если нужны):
- mistralai/Mistral-7B-Instruct-v0.2
- microsoft/Phi-3-mini-4k-instruct
"""

# Маппинг fine-tuned моделей на их базовые модели
MODEL_BASE_MAPPING = {
    "BroneBonBon/Conflict-Generator-Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "BroneBonBon/Conflict-Generator-Phi": "microsoft/Phi-3-mini-4k-instruct",
}

BASE_MODEL_NAME_DEFAULT: str = "BroneBonBon/Conflict-Generator-Phi"


class AiModule:
    def __init__(
        self,
        model_path: Path = MODEL_PATH_DEFAULT,
        model_name: str = BASE_MODEL_NAME_DEFAULT,
        use_4bit: bool = True,
        load_mode: Literal["finetuned", "base_with_lora", "base_only"] = "finetuned"
    ) -> None:
        """
        Инициализация модуля AI
        
        Args:
            model_path: Путь для кеширования моделей
            model_name: Имя модели (fine-tuned или базовой)
            use_4bit: Использовать 4-битную квантизацию
            load_mode: 
                - "finetuned": загрузить полностью обученную модель из HF
                - "base_with_lora": загрузить базовую + применить LoRA адаптеры
                - "base_only": только базовая модель без fine-tuning
        """
        self._model_path = model_path
        self._model_name = model_name
        self._use_4bit = use_4bit
        self._load_mode = load_mode
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._base_model_name = MODEL_BASE_MAPPING.get(model_name, model_name)
        self._load_model()

    def _load_model(self) -> None:
        try:
            print(f"🔧 Режим загрузки: {self._load_mode}")
            print(f"📦 Модель: {self._model_name}")
            print(f"💻 Устройство: {self._device}")
            
            # Загрузка токенизатора
            tokenizer_name = self._model_name if self._load_mode == "finetuned" else self._base_model_name
            print(f"📝 Загрузка токенизатора: {tokenizer_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                cache_dir=self._model_path,
                trust_remote_code=True
            )
        
            # Настройка pad_token если отсутствует
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Установка chat template для Phi-3 модели
            if "Phi-3" in self._model_name or "phi" in self._model_name.lower():
                self._tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '<|end|>\n' }}{% else %}{{ '<|system|>\n' + message['content'] + '<|end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
            elif "Mistral" in self._model_name or "mistral" in self._model_name.lower():
                self._tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% endif %}{% endfor %}"
            
            # ... остальная часть вашего кода остается без изменений
            
            # Конфигурация квантизации
            quantization_config = None
            if self._use_4bit and self._device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            
            # Загрузка модели в зависимости от режима
            if self._load_mode == "finetuned":
                # Загрузка полностью обученной модели
                print(f"⏳ Загрузка fine-tuned модели...")
                if quantization_config and self._device == "cuda":
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self._model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        cache_dir=self._model_path,
                        trust_remote_code=True
                    )
                else:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self._model_name,
                        cache_dir=self._model_path,
                        torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                        trust_remote_code=True
                    )
                    if self._device == "cpu":
                        self._model.to(self._device)
                        
            elif self._load_mode == "base_with_lora":
                # Загрузка базовой модели + LoRA адаптеры
                print(f"⏳ Загрузка базовой модели: {self._base_model_name}")
                
                if quantization_config and self._device == "cuda":
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self._base_model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        cache_dir=self._model_path,
                        trust_remote_code=True
                    )
                else:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self._base_model_name,
                        cache_dir=self._model_path,
                        torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                        trust_remote_code=True
                    )
                    if self._device == "cpu":
                        base_model.to(self._device)
                
                print(f"🔗 Применение LoRA адаптеров из: {self._model_name}")
                self._model = PeftModel.from_pretrained(
                    base_model,
                    self._model_name,
                    cache_dir=self._model_path
                )
                
            else:  # base_only
                # Только базовая модель без fine-tuning
                print(f"⏳ Загрузка базовой модели без fine-tuning: {self._base_model_name}")
                if quantization_config and self._device == "cuda":
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self._base_model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        cache_dir=self._model_path,
                        trust_remote_code=True
                    )
                else:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self._base_model_name,
                        cache_dir=self._model_path,
                        torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                        trust_remote_code=True
                    )
                    if self._device == "cpu":
                        self._model.to(self._device)
            
            print("✅ Модель успешно загружена!")
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            print("\n💡 Возможные решения:")
            print("  1. Проверьте подключение к интернету")
            print("  2. Убедитесь, что установлены: transformers, peft, bitsandbytes")
            print("  3. Для CPU попробуйте use_4bit=False")
            print("  4. Проверьте наличие модели на HuggingFace Hub")
            raise

    def get_response(
        self, 
        prompt: str, 
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        use_chat_template: bool = False  # Изменено на False по умолчанию
    ) -> str:
        try:
            # Подготовка входа
            if use_chat_template and hasattr(self._tokenizer, 'chat_template') and self._tokenizer.chat_template:
                # Используем chat template для инструктивных моделей
                messages = [
                    {"role": "user", "content": prompt}
                ]
                inputs = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self._device)
            else:
                # Простая токенизация без шаблона чата
                # Убедимся, что prompt - это строка
                if not isinstance(prompt, str):
                    prompt = str(prompt)
                
                # Для моделей типа Mistral/Phi добавим инструктивный формат
                if "Mistral" in self._base_model_name or "mistral" in self._base_model_name.lower():
                    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                elif "Phi-3" in self._base_model_name or "phi" in self._base_model_name.lower():
                    formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
                else:
                    formatted_prompt = prompt
                
                inputs = self._tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True
                ).input_ids.to(self._device)
            
            # Проверим, что inputs - это тензор PyTorch
            if not isinstance(inputs, torch.Tensor):
                # Если это dict, возьмем input_ids
                if isinstance(inputs, dict) and 'input_ids' in inputs:
                    inputs = inputs['input_ids'].to(self._device)
                else:
                    raise ValueError(f"Неправильный формат inputs: {type(inputs)}")
            
            # Генерация
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Декодирование
            full_response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только ответ модели (убираем промт)
            if use_chat_template:
                response = self._extract_assistant_response(full_response)
            else:
                # Для инструктивных форматов извлекаем ответ
                if "<|assistant|>" in full_response and "<|end|>" in full_response:
                    # Phi-3 формат
                    start = full_response.find("<|assistant|>") + len("<|assistant|>")
                    end = full_response.find("<|end|>", start)
                    response = full_response[start:end].strip() if end != -1 else full_response[start:].strip()
                elif "[/INST]" in full_response:
                    # Mistral формат
                    response = full_response.split("[/INST]")[-1].strip()
                else:
                    response = full_response[len(prompt):].strip() if len(full_response) > len(prompt) else full_response.strip()
            
            return response if response else "[Модель не сгенерировала ответ]"
            
        except Exception as e:
            return f"[Ошибка генерации: {e}]"

    def _extract_assistant_response(self, full_text: str) -> str:
        """Извлекает ответ ассистента из полного текста"""
        # Для Phi-3 и Mistral ответ идет после <|assistant|> или подобных токенов
        markers = ['<|assistant|>', 'assistant\n', 'Assistant:', 'ASSISTANT:']
        
        for marker in markers:
            if marker in full_text:
                parts = full_text.split(marker, 1)
                if len(parts) > 1:
                    response = parts[1].split('<|end|>')[0].strip()
                    return response
        
        # Если маркеры не найдены, возвращаем весь текст
        return full_text.strip()

    def get_model_info(self) -> dict:
        """Возвращает информацию о модели"""
        return {
            "model_name": self._model_name,
            "base_model": self._base_model_name,
            "load_mode": self._load_mode,
            "device": self._device,
            "quantized_4bit": self._use_4bit,
            "vocab_size": len(self._tokenizer) if self._tokenizer else 0,
            "model_type": self._model.config.model_type if self._model else "unknown"
        }


def chat_simulation(ai_module: AiModule) -> None:
    """Интерактивный чат с моделью"""
    print("\n" + "="*60)
    print("🤖 Conflict Generator - Локальный запуск")
    print("="*60)
    
    # Показываем информацию о модели
    info = ai_module.get_model_info()
    print(f"\n📊 Модель: {info['model_name']}")
    print(f"🏗️  Базовая: {info['base_model']}")
    print(f"🔧 Режим: {info['load_mode']}")
    print(f"💻 Устройство: {info['device']}")
    print(f"⚡ Квантизация 4-bit: {info['quantized_4bit']}")
    print(f"📝 Тип модели: {info['model_type']}")
    print(f"\n💡 Команды:")
    print("  - 'exit' - выход")
    print("  - 'help' - справка")
    print("  - 'info' - информация о модели")
    print("  - 'example' - пример запроса\n")
    
    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == "exit":
                print("\n👋 Завершение работы...")
                break
                
            if user_input.lower() == "help":
                print("\n📖 Справка по использованию:")
                print("  Модель обучена генерировать конфликтные ситуации.")
                print("  Примеры запросов:")
                print("    - 'создай конфликт в IT сфере'")
                print("    - 'напиши конфликтную ситуацию между менеджером и разработчиком'")
                print("    - 'сгенерируй конфликт в формате: Ситуация, Тип, Атака, Защита'")
                continue
                
            if user_input.lower() == "info":
                info = ai_module.get_model_info()
                print("\n📊 Информация о модели:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input.lower() == "example":
                example = "напиши конфликт в IT сфере. Выведи его в формате: Ситуация:\\nТип:\\nАтака:\\nЗащита:\\n"
                print(f"\n💡 Пример запроса: {example}")
                user_input = example
            
            # Генерация ответа
            print("\n🤖 Генерация ответа...", end="", flush=True)
            response = ai_module.get_response(user_input)
            print("\r" + " "*30 + "\r", end="")  # Очистка строки
            print(f"🤖 AI:\n{response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


def main() -> None:
    """Основная функция запуска"""
    try:
        print("🚀 Инициализация Conflict Generator...")
        
        # Выбор модели
        # ВАЖНО: Модели содержат только LoRA адаптеры!
        # Используйте режим "base_with_lora"
        
        # 1. Conflict Generator Phi (рекомендуется):
        ai = AiModule(
            model_name="BroneBonBon/Conflict-Generator-Mistral",
            use_4bit=True,
            load_mode="finetuned"  # ✅ ПРАВИЛЬНО — это full model
        )
        
        # 2. Conflict Generator Mistral:
        # ai = AiModule(
        #     model_name="BroneBonBon/Conflict-Generator-Mistral",
        #     use_4bit=True,
        #     load_mode="base_with_lora"  # ← ПРАВИЛЬНЫЙ РЕЖИМ
        # )
        
        # 3. Только базовая модель без fine-tuning (для сравнения):
        # ai = AiModule(
        #     model_name="microsoft/Phi-3-mini-4k-instruct",
        #     use_4bit=True,
        #     load_mode="base_only"
        # )
        
        chat_simulation(ai)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())