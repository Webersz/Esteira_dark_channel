"""Script generation with topic de-duplication.

Primary engine: Gemini (google-genai SDK, Structured Outputs), cascading models x keys and skipping
404/429/5xx instantly. If all of them fail, free OpenAI-compatible fallbacks (Groq / GitHub Models /
OpenRouter) are tried, validated against the same `VideoScript` schema.
"""
from __future__ import annotations

import difflib
import json
import logging
import random
import re
import time

import requests
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.config_loader import ChannelConfig, FallbackLLM, get_settings
from src.database import normalize_topic
from src.engine import mascot
from src.engine.schemas import CameraMove, TransitionSfx, VideoScript

log = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.78
MAX_ATTEMPTS = 4  # content-rule retries
CASCADE_ROUNDS = 3  # full passes over Gemini models x keys + fallbacks
ROUND_WAITS = (60, 180)  # sleep between rounds (rides out short "high demand" spikes)
SKIP_CODES = {400, 429, 500, 502, 503, 504}  # try the next key/model right away (400: e.g. no response_schema)
UNAVAILABLE_CODES = {403, 404}  # model not enabled for this key: never retried in this process
FALLBACK_TIMEOUT = 120
# Gemini 3 "thinking" shares max_output_tokens with the JSON: at default level it used ~4.8k of 8k
# tokens and sometimes truncated the script. "low" skips it, ~3x faster, scripts still valid.
THINKING_LEVEL = "low"
MAX_OUTPUT_TOKENS = 16384
TWIST_SCENE = 7  # 1-based, lands around the 25 s mark

# Rotated per video (anti format fatigue). Each rule is injected for scene 1 only.
HOOK_ARCHETYPES: dict[str, str] = {
    "IN_MEDIA_RES": "Start at the exact millisecond of catastrophe, failure or peak emotional breakdown, with zero "
                    "introductory context. Example: 'Faltavam exatos 4 segundos para a válvula explodir quando...'",
    "MITO_QUEBRADO": "Open by contradicting a belief the audience holds, to trigger cognitive dissonance. "
                     "Example: 'Se você acha que fazer isso vai salvar sua vida, você acabou de assinar sua sentença...'",
    "NUMERO_ESPECIFICO": "Open with a hyper-specific, bizarre financial or technical number. "
                         "Example: 'Por causa de 42 milhões de reais e uma linha de código esquecida às 3 da manhã...'",
    "CONFRONTO_DIRETO": "Address the viewer directly with an urgent command or physiological trigger. "
                        "Example: 'Não pisque pelos próximos 4 segundos e olhe para o canto dessa imagem...'",
}

_clients: dict[str, genai.Client] = {}
_dead_pairs: set[tuple[str, str]] = set()  # (model, key) that answered 403/404


class ScriptGenerationError(Exception):
    pass


def _get_client(api_key: str) -> genai.Client:
    if api_key not in _clients:
        _clients[api_key] = genai.Client(api_key=api_key)
    return _clients[api_key]


SYSTEM_INSTRUCTION = """You are the head writer of a top-performing faceless short-form video channel.
You write scripts engineered for maximum retention on YouTube Shorts and TikTok:
- Scene 1 is a pattern-interrupt hook that creates an open loop in the first 2 seconds (no greetings, no "hoje vamos").
- Every scene adds new information or tension; no filler, no repetition.
- Scene 7 (~25 s) is a mandatory retention reset: a sudden complication, betrayal or unexpected discovery
  ("Mas o que ninguém sabia é que..."). Re-hook again around scene 11.
- Scenes 16 and 1 form ONE continuous sentence (infinite loop): scene 16 delivers the climax/resolution in its
  first 7 words and ends with an open connector phrase ("...e tudo isso só aconteceu porque no começo...",
  "...e o motivo real disso tudo é que...") WITHOUT a final period, so it flows grammatically into scene 1.
- No "se inscreva" / "siga" calls anywhere: they break the loop.
- Narration is plain spoken text: no emojis, no markdown, no brackets, no stage directions, numbers written as words when short.
- Write narration, title, description and comment with FULL, correct accentuation and cedilla (é, ã, ç, ô...).
  The voice-over engine mispronounces unaccented words ("e" vs "é", "mascaras" vs "máscaras").
- Image prompts are in English, cinematic, vertical 9:16, describe a single clear focal subject, and never contain text.
  Every prompt names the era and place of the story (e.g. "Scotland 1900") so all images look like the same world.
  Real people are never shown with a visible face: use silhouettes, backs, hands or the objects they left behind.
  Prefer concrete evidence (the locked door, the logbook, the twisted railing) over generic scenery.
- Each scene has two images: `image_prompt` for the first half of the line, `image_prompt_b` for the second half,
  with a different framing (wide vs close-up vs detail).
- `callout` is a 1-3 word on-screen headline. Scene 1 always has one (the hook in a few words, e.g. "3 HOMENS SUMIRAM");
  add 5-7 more on scenes with a date, a number or a shocking fact; leave it empty everywhere else.
"""


def _build_prompt(channel: ChannelConfig, recent_topics: list[str], feedback: list[str], hook: str) -> str:
    per_scene = channel.target_words / channel.scenes
    mascot_rules = (
        f"\nMASCOT (appears as a separate animated sprite in about half of the scenes): {channel.character_description}."
        "\nFill `character_pose` for scenes where the mascot reacts/explains, using EXACTLY one of these pose keys: "
        f"{', '.join(mascot.pose_names(channel))}. Pick the key matching the emotion of the line. The `image_prompt` "
        "of those scenes must describe ONLY the background environment (no characters), because the mascot is "
        "composited on top."
        if channel.is_character_channel
        else "\nThis channel has NO mascot: `character_pose` must always be an empty string."
    )
    banned = "\n".join(f"- {t}" for t in recent_topics[:120]) or "- (none yet)"
    fixes = ("\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED. Fix these problems:\n" + "\n".join(f"- {f}" for f in feedback)) if feedback else ""
    return f"""CHANNEL: {channel.name}
NICHE: {channel.niche}
LANGUAGE OF NARRATION, TITLE, DESCRIPTION AND COMMENT: {channel.language}
EDITORIAL GUIDELINES: {channel.prompt_guidelines}
VISUAL STYLE (for image prompts): {channel.visual_style}{mascot_rules}

HOOK ARCHETYPE: {hook}
SCENE 1 RULE (strict): {HOOK_ARCHETYPES[hook]}
SCENE {TWIST_SCENE} RULE (strict): plot twist / pattern interrupt; set its sfx to "impact" and camera to "shake".
SCENE {channel.scenes} RULE (strict): climax in the first 7 words, ends with an open connector that loops into scene 1, no final period.

HARD REQUIREMENTS:
- Exactly {channel.scenes} scenes.
- Total narration between {round(channel.target_words * 0.92)} and {round(channel.target_words * 1.08)} words
  (about {per_scene:.0f} words per scene) so the voice-over lasts ~{channel.target_duration:.0f} seconds.
- Pick ONE fresh, specific topic. It must be clearly different from every topic already used below.

TOPICS ALREADY USED (do not repeat or paraphrase):
{banned}{fixes}
"""


def _too_similar(topic: str, recent_topics: list[str]) -> str | None:
    norm = normalize_topic(topic)
    for previous in recent_topics:
        if difflib.SequenceMatcher(None, norm, normalize_topic(previous)).ratio() >= SIMILARITY_THRESHOLD:
            return previous
    return None


_CLEAN_RE = re.compile(r"[\*\#_\[\]\(\)<>{}|~`]|[\U0001F000-\U0001FAFF☀-➿]")


def _sanitize(script: VideoScript) -> VideoScript:
    for scene in script.scenes:
        scene.narration = re.sub(r"\s+", " ", _CLEAN_RE.sub("", scene.narration)).strip()
        scene.image_prompt = scene.image_prompt.strip()
        scene.image_prompt_b = scene.image_prompt_b.strip()
        scene.callout = re.sub(r"\s+", " ", _CLEAN_RE.sub("", scene.callout)).strip().upper()[:28]
        scene.character_pose = scene.character_pose.strip()
    if script.scenes:
        # Loop: no falling "end of sentence" intonation on the last line
        script.scenes[-1].narration = script.scenes[-1].narration.rstrip(" .!")
    script.hashtags = [h if h.startswith("#") else f"#{h}" for h in (h.strip().replace(" ", "") for h in script.hashtags) if h]
    script.tags = [t.strip().lstrip("#") for t in script.tags if t.strip()]
    return script


def validate_script(script: VideoScript, channel: ChannelConfig, recent_topics: list[str]) -> list[str]:
    problems: list[str] = []
    if len(script.scenes) != channel.scenes:
        problems.append(f"You returned {len(script.scenes)} scenes; return exactly {channel.scenes}.")
    words = script.word_count
    lo, hi = round(channel.target_words * 0.85), round(channel.target_words * 1.15)
    if not lo <= words <= hi:
        problems.append(f"Narration has {words} words; it must have between {lo} and {hi}.")
    if any(len(s.narration.split()) < 4 for s in script.scenes):
        problems.append("Some scenes have almost no narration; every scene needs 8-13 words.")
    if any(not s.image_prompt for s in script.scenes):
        problems.append("Every scene needs a non-empty image_prompt.")
    similar = _too_similar(script.topic, recent_topics)
    if similar:
        problems.append(f"Topic '{script.topic}' is too similar to already used topic '{similar}'. Choose another subject.")
    return problems


def _gemini_once(model: str, api_key: str, prompt: str) -> VideoScript:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=VideoScript,
        temperature=1.0,
        top_p=0.95,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
    )
    response = _get_client(api_key).models.generate_content(model=model, contents=prompt, config=config)
    if isinstance(response.parsed, VideoScript):
        return response.parsed
    # Fallback: parse raw JSON text (gives a precise error if malformed)
    return VideoScript.model_validate_json(response.text or "")


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    return text


def _fallback_once(llm: FallbackLLM, prompt: str) -> VideoScript:
    schema = json.dumps(VideoScript.model_json_schema(), ensure_ascii=False)
    system = (
        f"{SYSTEM_INSTRUCTION}\nReply ONLY with one valid JSON object that strictly follows this JSON Schema, "
        f"with no markdown fences and no extra text:\n{schema}"
    )
    response = requests.post(
        llm.url,
        headers={"Authorization": f"Bearer {llm.api_key}", "Content-Type": "application/json"},
        json={
            "model": llm.model,
            "temperature": 1.0,
            "max_tokens": 8192,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        },
        timeout=FALLBACK_TIMEOUT,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"] or ""
    return VideoScript.model_validate_json(_strip_fences(content))


def _call_llm(prompt: str) -> VideoScript:
    """Gemini models x keys, then free fallbacks; repeats the whole cascade CASCADE_ROUNDS times.

    Schema violations (pydantic ValidationError, a ValueError) propagate so `generate_script` can feed
    them back to the model.
    """
    settings = get_settings()
    keys, models, fallbacks = settings.gemini_api_keys, settings.gemini_models, settings.fallback_llms
    if not keys and not fallbacks:
        raise ScriptGenerationError("No LLM configured: set GEMINI_API_KEYS (or GEMINI_API_KEY) or a fallback key.")
    errors: list[str] = []
    for round_no in range(CASCADE_ROUNDS):
        for model in models:
            for idx, key in enumerate(keys, start=1):
                if (model, key) in _dead_pairs:
                    continue
                try:
                    script = _gemini_once(model, key, prompt)
                    log.info("Script via Gemini %s (key #%d)", model, idx)
                    return script
                except genai_errors.APIError as exc:
                    code = getattr(exc, "code", None)
                    errors.append(f"gemini {model} key#{idx}: HTTP {code}")
                    if code in UNAVAILABLE_CODES:
                        _dead_pairs.add((model, key))
                    elif code not in SKIP_CODES:
                        raise ScriptGenerationError(f"Gemini API error: {exc}") from exc
                    log.warning("Gemini %s (key #%d) HTTP %s, trying next", model, idx, code)
        for llm in fallbacks:
            try:
                script = _fallback_once(llm, prompt)
                log.info("Script via fallback %s (%s)", llm.name, llm.model)
                return script
            except (requests.RequestException, KeyError, IndexError, TypeError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                errors.append(f"{llm.name}: {status or type(exc).__name__}")
                log.warning("Fallback %s failed (%s), trying next", llm.name, status or exc)
        if round_no < CASCADE_ROUNDS - 1:
            wait = ROUND_WAITS[min(round_no, len(ROUND_WAITS) - 1)]
            log.warning("Every LLM failed (round %d/%d), retrying in %ss", round_no + 1, CASCADE_ROUNDS, wait)
            time.sleep(wait)
    raise ScriptGenerationError(f"All LLMs failed: {'; '.join(errors[-12:])}")


def generate_script(channel: ChannelConfig, recent_topics: list[str], extra_feedback: list[str] | None = None) -> VideoScript:
    """Generate a validated 16-scene script. Retries with model-facing feedback on rule violations."""
    feedback = list(extra_feedback or [])
    last_problems: list[str] = []
    hook = random.choice(list(HOOK_ARCHETYPES))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            script = _sanitize(_call_llm(_build_prompt(channel, recent_topics, feedback, hook)))
        except ValueError as exc:  # pydantic ValidationError subclasses ValueError
            log.warning("[%s] script attempt %d: invalid JSON/schema: %s", channel.id, attempt, str(exc)[:300])
            feedback = [f"Your JSON did not match the schema: {str(exc)[:400]}"]
            continue
        if len(script.scenes) > channel.scenes:
            script.scenes = script.scenes[: channel.scenes]  # cheap fix, then re-check word budget
        last_problems = validate_script(script, channel, recent_topics)
        if not last_problems:
            script.hook_archetype = hook
            twist = script.scenes[TWIST_SCENE - 1]  # enforced regardless of what the model chose
            twist.sfx, twist.camera = TransitionSfx.IMPACT, CameraMove.SHAKE
            log.info("[%s] script OK on attempt %d: %s (%d words, hook %s)", channel.id, attempt, script.topic,
                     script.word_count, hook)
            return script
        log.warning("[%s] script attempt %d rejected: %s", channel.id, attempt, last_problems)
        feedback = last_problems
    raise ScriptGenerationError(f"Could not produce a valid script after {MAX_ATTEMPTS} attempts: {last_problems}")
