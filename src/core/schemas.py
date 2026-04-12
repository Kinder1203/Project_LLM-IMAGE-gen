from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict


RequestInputType = Literal["text", "image", "modification", "image_only", "image_and_text"]
CanonicalInputType = Literal["text", "image_only", "image_and_text"]
ResponseStatus = Literal["success", "failed", "waiting_for_user", "waiting_for_user_edit"]


def _has_content(value: Optional[str]) -> bool:
    return bool((value or "").strip())


def normalize_input_type(
    prompt: Optional[str],
    image_url: Optional[str],
) -> CanonicalInputType:
    has_prompt = _has_content(prompt)
    has_image = _has_content(image_url)

    if has_image and has_prompt:
        return "image_and_text"
    if has_image:
        return "image_only"
    return "text"


class PipelineRequest(BaseModel):
    input_type: RequestInputType = Field(
        "text",
        description="Raw input label. Runtime normalizes it into the canonical type from payload shape.",
    )
    prompt: Optional[str] = Field(None, description="Ring generation or customization prompt.")
    image_url: Optional[str] = Field(
        None,
        description="Source ring image URL or ComfyUI input filename.",
    )
    engraving_reference_image_url: Optional[str] = Field(
        None,
        description="Optional engraving or pattern reference image for edit scenarios.",
    )
    gemstone_reference_image_url: Optional[str] = Field(
        None,
        description="Optional gemstone reference image for edit scenarios.",
    )

    thread_id: str = Field(..., description="LangGraph thread id. Required for every request.")
    action: Literal["start", "accept_base", "request_customization"] = Field(
        "start",
        description="Pipeline action.",
    )
    customization_prompt: Optional[str] = Field(
        None,
        description="Customization instruction used for follow-up edits.",
    )

    @model_validator(mode="after")
    def canonicalize_input_type(self) -> "PipelineRequest":
        self.input_type = normalize_input_type(self.prompt, self.image_url)

        if not _has_content(self.thread_id):
            raise ValueError("thread_id is required and must not be blank.")

        if self.action == "start" and not (_has_content(self.prompt) or _has_content(self.image_url)):
            raise ValueError("start action requires either a prompt, an image_url, or both.")

        if self.action == "request_customization" and not _has_content(self.customization_prompt):
            raise ValueError("request_customization action requires customization_prompt.")

        return self


class PipelineResponse(BaseModel):
    status: ResponseStatus
    optimized_image_urls: List[str] = Field(
        default_factory=list,
        description="Validated multi-view image URLs.",
    )
    message: str = Field(..., description="User-facing status message.")
    base_image_url: Optional[str] = None


class AgentState(TypedDict):
    input_type: str
    user_prompt: str
    user_image: str
    intent: str
    rag_context: str
    synthesized_prompt: str

    base_ring_image_url: str
    base_ring_image_ref: str
    customization_prompt: str
    engraving_reference_image_url: str
    gemstone_reference_image_url: str
    customization_context: str
    customization_kind: str
    expected_engraving_text: str
    edited_ring_image_url: str
    edited_ring_image_ref: str

    validation_reason: str
    guardrail_result: str
    generation_result: str
    retry_count: int

    current_image_urls: List[str]

    is_valid: bool
    validation_feedback: str

    final_output_urls: List[str]
    status_message: str
