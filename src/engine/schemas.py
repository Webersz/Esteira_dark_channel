"""Pydantic models used as Gemini Structured Output `response_schema`.

Keep these schema-friendly for google-genai: plain str/list/enum fields, no unions or
validators that raise during parsing. Business rules (scene count, word budget) are
checked in `script_generator.validate_script` so failures can be fed back to the model.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CameraMove(str, Enum):
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    SHAKE = "shake"


class TransitionSfx(str, Enum):
    WHOOSH = "whoosh"
    IMPACT = "impact"
    POP = "pop"
    NONE = "none"


class Scene(BaseModel):
    narration: str = Field(
        description="Spoken voice-over for this scene only, 9-12 words, no emojis, no stage directions."
    )
    image_prompt: str = Field(
        description="English prompt for a vertical 9:16 image that literally illustrates the narration. "
        "Describe subject, setting, lighting and camera angle. Never ask for text, letters or logos."
    )
    character_pose: str = Field(
        description="English description of the mascot character's action/expression in this scene "
        "(e.g. 'pointing up excitedly, mouth open'). Empty string if the mascot should not appear "
        "or the channel has no mascot."
    )
    broll_query: str = Field(description="2-3 English keywords to search stock footage for this scene.")
    camera: CameraMove = Field(description="Camera movement that best fits the emotion of the scene.")
    sfx: TransitionSfx = Field(
        description="Sound effect at the start of the scene: impact for reveals/shocks, whoosh for fast "
        "cuts, pop for light/funny moments, none for calm moments."
    )


class VideoScript(BaseModel):
    hook_archetype: str = Field(description="Echo the HOOK ARCHETYPE name given in the prompt.")
    topic: str = Field(description="Short unique topic label (max 12 words) used for de-duplication.")
    title: str = Field(description="Curiosity-driven title in the channel language, max 70 characters, no hashtags.")
    description: str = Field(description="2-3 sentence description in the channel language for YouTube/TikTok.")
    tags: list[str] = Field(description="8-12 search tags, no '#'.")
    hashtags: list[str] = Field(description="3-5 hashtags including '#'.")
    pinned_comment: str = Field(
        description="Engagement comment posted by the channel: a provocative question that invites replies."
    )
    scenes: list[Scene] = Field(description="Exactly 16 scenes in narrative order.")

    @property
    def narration(self) -> str:
        return " ".join(scene.narration.strip() for scene in self.scenes)

    @property
    def word_count(self) -> int:
        return len(self.narration.split())
