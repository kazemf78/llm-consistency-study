# vllm_local.py
from typing import List
from llm_consistency.models.base import LocalLLM
from vllm import LLM, SamplingParams

class VLLMLocalLLM(LocalLLM):
    """vLLM-backed local LLM with the same interface as HFLocalLLM."""

    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(model_id, *args, **kwargs)
        self.llm = None
        self._gen_defaults = {}

    # ---- helpers ----
    def apply_chat(self, p: str) -> str:
        # vLLM does NOT apply chat templates automatically
        # so this stays basically identical
        msgs = [{"role": "user", "content": p}]
        kwargs = {}
        if hasattr(self, "enable_thinking"):
            kwargs["enable_thinking"] = self.enable_thinking # todo: check if this is still a good way to configure this or not?
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                **kwargs
            )
        return p

    # ---- PREPARE ----
    def prepare(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # only keep args SamplingParams actually accepts
        sampling_fields = SamplingParams.__annotations__.keys()
        self._gen_defaults = {
            k: v for k, v in self.extra_kwargs.items()
            if k in sampling_fields
        }

        load_kwargs = {
            k: v for k, v in self.extra_kwargs.items()
            if k not in sampling_fields
        }
        print(load_kwargs)
        self.llm = LLM(model=self.model_id, **load_kwargs)

    # ---- SINGLE PROMPT ----
    def single(self, prompt: str, *args, **kwargs) -> str:
        prompt = self.apply_chat(prompt)

        gen_kwargs = {**self._gen_defaults, **kwargs}
        gen_kwargs.setdefault("max_tokens", 256)
        if "max_new_tokens" in gen_kwargs:
            gen_kwargs["max_tokens"] = gen_kwargs["max_new_tokens"]

        sampling = SamplingParams(**gen_kwargs)
        
        outputs = self.llm.generate([prompt], sampling)

        return outputs[0].outputs[0].text.strip()

    # ---- BATCH PROMPTS ----
    def batch(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts = [self.apply_chat(p) for p in prompts]

        gen_kwargs = {**self._gen_defaults, **kwargs}
        gen_kwargs.setdefault("max_tokens", 256)
        if "max_new_tokens" in gen_kwargs:
            gen_kwargs["max_tokens"] = gen_kwargs["max_new_tokens"]


        sampling = SamplingParams(**gen_kwargs)
        print("*"*100)
        print(sampling)

        outputs = self.llm.generate(prompts, sampling)

        return [o.outputs[0].text.strip() for o in outputs]
