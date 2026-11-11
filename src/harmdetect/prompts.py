"""Prompts for LLM-based classification of German posts on three tasks.

The prompt asks for all three labels at once ("a,b,c" = fake news, hate speech,
toxicity) and encodes the indicators studied in the project. The few-shot
variant adds German demonstrations, which was the single biggest lever for LLM
accuracy in the experiments.
"""

from __future__ import annotations

TASK_ORDER = ("fake_news", "hate_speech", "toxicity")

SYSTEM_PROMPT = """\
You are a content analyst who classifies German-language social media posts on three tasks.

1. Fake news / disinformation: 1 (possible_fake_news) if the post shows any of
   - sensational or exaggerated claims or headlines
   - missing specifics (no source, date, place or verifiable detail) or broken logic
   - emotionally charged wording designed to go viral
   - contradictory statements or claims unsupported by evidence
   - language intended to incite fear, panic or outrage, or friend-foe framing
   otherwise 0 (neutral_post).
2. Hate speech: 1 (possible_hate_speech) if the post attacks, demeans or stereotypes people
   because of a protected characteristic (origin, religion, gender, sexuality, disability, ...),
   explicitly or implicitly; otherwise 0.
3. Toxicity: 1 (possible_toxic) if the post is offensive, hostile, vulgar, insulting or threatening,
   even when it targets no group; otherwise 0.

Overlap rules: group-targeted hate that is also insulting -> hate 1 and toxicity 1;
a personal insult without group targeting -> hate 0, toxicity 1.

Judge only the text. Answer ONLY with three digits separated by commas in the order
fake_news,hate_speech,toxicity, e.g. 0,1,1. No explanation."""

FEW_SHOT_EXAMPLES = [
    ("Die NASA bestätigt, dass der Mond im Jahr 2025 zum ersten Mal grün sein wird.", "1,0,0"),
    ("Die Weltgesundheitsorganisation hat neue Leitlinien für die psychische Gesundheit veröffentlicht.", "0,0,0"),
    ("Frauen sind faul und leisten keinen Beitrag zur Gesellschaft.", "0,1,1"),
    ("Wir müssen zusammenarbeiten, um eine bessere Zukunft zu schaffen.", "0,0,0"),
    ("Du bist der schlimmste Mensch, den ich je getroffen habe. Ich hoffe, du versagst bei allem.", "0,0,1"),
    ("Die Medien verschweigen euch die Wahrheit über die Wahl! Teilt das, bevor es gelöscht wird!!!", "1,0,0"),
]


def build_messages(text: str, few_shot: bool = True) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for example, labels in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example})
            messages.append({"role": "assistant", "content": labels})
    messages.append({"role": "user", "content": text})
    return messages
