"""Generation prompt templates maintained as part of HTEB.

Run configurations select transformations; these instructions define them.
"""

GENERATION_PROMPTS: dict[str, str] = {
    "paraphrasing": (
        "Rephrase the following text while keeping its original meaning. "
        "Answer in the same language as the input text. Only reply with one "
        "paraphrased text, with no explanations or notes."
    ),
    "expansion": (
        "Expand the following text with more detail and context while preserving "
        "its meaning. Keep questions as questions and statements as statements. "
        "Answer in the same language. Only reply with the expanded text."
    ),
    "summarise": (
        "Make the following text shorter while preserving its meaning. Keep "
        "questions as questions and statements as statements. Answer in the same "
        "language. Only reply with the summary."
    ),
    "style_change": (
        "Change the style of the following text. Rewrite informal text formally "
        "and formal text informally while preserving its meaning. Answer in the "
        "same language. Only reply with the rewritten text."
    ),
    "translation": (
        "Translate the following text from {source_language} to {target_language}. "
        "Provide only the translation, with no explanations or notes."
    ),
    "cross_translation": (
        "Translate the following text from {source_language} to {target_language}. "
        "Provide only the translation, with no explanations or notes."
    ),
    "backtranslate_forward": (
        "Translate the following text from {source_language} to {target_language}. "
        "Provide only the translation, with no explanations or notes."
    ),
    "backtranslate_backward": (
        "Translate the following text from {source_language} to {target_language}. "
        "Provide only the translation, with no explanations or notes."
    ),
}


def generation_prompt(stage: str, *, source_language: str, target_language: str) -> str:
    """Format an HTEB generation instruction using the resolved language names."""
    return (
        GENERATION_PROMPTS[stage]
        .replace("{source_language}", source_language)
        .replace("{target_language}", target_language)
    )
