# src/llm_consistency/models/factory.py
"""
Factory for creating LLM instances based on model name.
Auto-detects whether to use API or local models.
"""

from typing import Optional, Union


# Model name patterns for API providers
API_MODEL_PATTERNS = {
    "openai": ["gpt-", "o1-", "o3-"],
    "anthropic": ["claude-"],
    # Add more providers as needed
}

# Known API models (can be extended)
KNOWN_API_MODELS = {
    # OpenAI
    "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo",
    "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo-16k",
    "o1-preview", "o1-mini",
    
    # Anthropic (if you add support)
    "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
    "claude-3-5-sonnet", "claude-3-5-haiku",
}

# Special suffix for enabling thinking mode
THINKING_SUFFIX = "[with_thinking]"


def _parse_model_name(model_name: str) -> tuple[str, bool]:
    """
    Parse model name and extract thinking mode flag.
    
    Args:
        model_name: Model name, optionally with [with_thinking] suffix
        
    Returns:
        (clean_model_name, enable_thinking)
        
    Examples:
        _parse_model_name("gpt-4o") → ("gpt-4o", False)
        _parse_model_name("gpt-4o[with_thinking]") → ("gpt-4o", True)
        _parse_model_name("meta-llama/Llama-3.1-8B[with_thinking]") → ("meta-llama/Llama-3.1-8B", True)
    """
    enable_thinking = model_name.lower().endswith(THINKING_SUFFIX.lower())
    if enable_thinking:
        clean_name = model_name[:-(len(THINKING_SUFFIX))]
    else:
        clean_name = model_name
    return clean_name, enable_thinking


def is_api_model(model_name: str) -> bool:
    """
    Determine if a model should use API (vs local inference).
    Handles [with_thinking] suffix automatically.
    
    Args:
        model_name: Name of the model (may include [with_thinking])
        
    Returns:
        True if model should use API, False if local
        
    Examples:
        is_api_model("gpt-4o") → True
        is_api_model("gpt-4o[with_thinking]") → True
        is_api_model("meta-llama/Llama-3.1-8B-Instruct") → False
        is_api_model("Qwen/Qwen2.5-7B-Instruct[with_thinking]") → False
    """
    # Strip thinking suffix for detection
    clean_name, _ = _parse_model_name(model_name)
    model_lower = clean_name.lower()
    
    # Check exact matches first
    if clean_name in KNOWN_API_MODELS:
        return True
    
    # Check patterns
    for provider, patterns in API_MODEL_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in model_lower:
                if "/" not in clean_name:  # Exclude HuggingFace-style names
                    return True
                
    
    # Check for HuggingFace-style naming (org/model)
    # These are typically local models
    if "/" in model_name:
        # Exception: if it's an API-hosted model with slash
        # (you can add exceptions here if needed)
        return False
    
    # Default: assume local if not recognized
    return False


def get_llm(
    model_name: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.85,
    use_vllm: bool = True,
    **kwargs
) -> Union['APILLM', 'LocalLLM']: # todo: change this to handle diffusion model as well!
    """
    Factory function to create the appropriate LLM instance.
    Auto-detects whether to use API or local based on model name.
    Supports [with_thinking] suffix to enable thinking mode.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4o", "meta-llama/Llama-3.1-8B")
                   Can include [with_thinking] suffix for thinking mode
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        gpu_memory_utilization: GPU memory fraction for local models (vLLM)
        use_vllm: Use vLLM for local models (faster) vs HF transformers
        **kwargs: Additional arguments passed to LLM constructor
        
    Returns:
        Initialized LLM instance (not yet prepared!)
        
    Examples:
        # API model (auto-detected)
        llm = get_llm("gpt-4o")
        
        # Local model (auto-detected)
        llm = get_llm("meta-llama/Llama-3.1-8B-Instruct")
        
        # Enable thinking mode
        llm = get_llm("gpt-4o[with_thinking]")
        llm = get_llm("meta-llama/Llama-3.1-8B[with_thinking]")
        
        # Force specific backend
        llm = get_llm("meta-llama/Llama-3.1-8B-Instruct", use_vllm=False)
    """
    
    # Parse model name and thinking flag
    clean_model_name, enable_thinking = _parse_model_name(model_name)
    
    # Parse model name and thinking flag
    clean_model_name, enable_thinking = _parse_model_name(model_name)
    
    if is_api_model(clean_model_name):
        # Use API
        from llm_consistency.models.openai_api import OpenAIAPILLM
        llm = OpenAIAPILLM(
            model=clean_model_name,
            max_output_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    else:
        # Use local model
        if use_vllm:
            from llm_consistency.models.vllm_local import VLLMLocalLLM
            kwargs = {k: v for k, v in kwargs.items() if k != "api_conc"}
            llm = VLLMLocalLLM(
                model_id=clean_model_name,
                max_tokens=max_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
                temperature=temperature,
                **kwargs
            )
        else:
            from llm_consistency.models.hf_local import HFLocalLLM
            llm = HFLocalLLM(
                model_id=clean_model_name,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                **kwargs
            )
    
    # Set thinking mode if requested
    if enable_thinking:
        llm.enable_thinking = True
    else:
        llm.enable_thinking = False
    
    return llm


def get_llm_for_evaluation(
    model_name: str,
    max_tokens: int = 32,
    **kwargs
) -> Union['APILLM', 'LocalLLM']:
    """
    Convenience function for getting LLM for evaluation/grading.
    Uses appropriate defaults for evaluation tasks.
    
    Args:
        model_name: Name of the judge model
        max_tokens: Max tokens (evaluation responses are short)
        **kwargs: Additional arguments
        
    Returns:
        LLM instance configured for evaluation
    """
    return get_llm(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=kwargs.pop("temperature", 0.0),  # Deterministic is the default for grading unless overridden by caller
        **kwargs
    )


# Convenience function for backward compatibility
def get_llm_from_list(
    model_name: str,
    api_models: list,
    max_local_tokens: int = 256,
    max_api_tokens: int = 256,
    **kwargs
) -> Union['APILLM', 'LocalLLM']:
    """
    Legacy function that matches your current answer_generation.py pattern.
    Can be used as a drop-in replacement for the old get_llm() function.
    Supports [with_thinking] suffix.
    
    Args:
        model_name: Name of the model (may include [with_thinking])
        api_models: List of model names that should use API
        max_local_tokens: Max tokens for local models
        max_api_tokens: Max tokens for API models
        **kwargs: Additional arguments
        
    Returns:
        LLM instance
        
    Examples:
        llm = get_llm_from_list("gpt-4o", ["gpt-4o", "gpt-4.1"])
        llm = get_llm_from_list("gpt-4o[with_thinking]", ["gpt-4o"])
    """
    # Parse model name and thinking flag
    clean_model_name, enable_thinking = _parse_model_name(model_name)
    
    if clean_model_name in api_models:
        from llm_consistency.models.openai_api import OpenAIAPILLM
        llm = OpenAIAPILLM(
            model=clean_model_name,
            max_output_tokens=max_api_tokens,
            **kwargs
        )
    elif "gpt-oss" in clean_model_name:
        print("Using HFLocalLLM for gpt-oss model")
        print("*"*200)
        from llm_consistency.models.hf_local import HFLocalLLM
        llm = HFLocalLLM(
            model_id=clean_model_name,
            max_new_tokens=max_local_tokens,
            **kwargs
        )
    else:
        from llm_consistency.models.vllm_local import VLLMLocalLLM
        llm = VLLMLocalLLM(
            model_id=clean_model_name,
            max_tokens=max_local_tokens,
            gpu_memory_utilization=kwargs.pop("gpu_memory_utilization", 0.85),
            **kwargs
        )
    
    # Set thinking mode if requested
    if enable_thinking:
        llm.enable_thinking = True
    else:
        llm.enable_thinking = False
    
    return llm