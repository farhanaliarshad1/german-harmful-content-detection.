"""Prompts for LLM-based fake-news classification of German posts.

The criteria encode the disinformation indicators studied in the project
(sensationalism, missing specifics, emotional virality, unsupported claims,
fear-mongering). FEW_SHOT adds German demonstrations, which was the single
biggest lever for LLM accuracy in the course experiments.
"""

SYSTEM_ZERO_SHOT = """\
You are a content analyst who classifies German-language social media posts for disinformation.

Label 1 (possible_fake_news) if the post shows one or more of:
- sensational or exaggerated claims or headlines
- missing specifics (no source, date, place or verifiable detail) or broken logic
- emotionally charged wording designed to go viral
- contradictory statements or claims unsupported by evidence
- language intended to incite fear, panic or outrage
- friend-foe framing ("they" are hiding the truth from "us")
Otherwise label 0 (neutral_post).

Judge only the text itself. Answer with a single digit: 0 or 1. No explanation."""

FEW_SHOT_EXAMPLES = [
    ("Die NASA bestätigt, dass der Mond im Jahr 2025 zum ersten Mal grün sein wird.", 1),
    ("Die Weltgesundheitsorganisation hat neue Leitlinien für die psychische Gesundheit veröffentlicht.", 0),
    ("Die Medien verschweigen euch die Wahrheit über die Wahl! Teilt das, bevor es gelöscht wird!!!", 1),
    ("Der Bundestag debattiert am Donnerstag über den Haushaltsentwurf für 2025.", 0),
]


def build_messages(text: str, few_shot: bool = True) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_ZERO_SHOT}]
    if few_shot:
        for example, label in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example})
            messages.append({"role": "assistant", "content": str(label)})
    messages.append({"role": "user", "content": text})
    return messages
