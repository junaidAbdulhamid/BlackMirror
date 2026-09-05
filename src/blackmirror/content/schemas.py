"""Content intelligence contracts.

Phase 4 answers "what is happening in the stimulus?" as timestamped, typed,
traceable data — never a free-form paragraph. Every derived statement carries
provenance (which model produced it, with what confidence) so a downstream
claim can always be traced back to the tool that made it.

Sizing rule, inherited from Phase 1: no bulk numeric arrays in JSON. Dense
time-series live in `features.npz`; these models describe and point at them.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _TimedInterval(BaseModel):
    """Shared half-open interval contract used by every temporal record."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered_interval(self) -> _TimedInterval:
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        return self


class _DurationInterval(_TimedInterval):
    """Interval whose persisted duration must agree with its boundaries."""

    duration: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent_duration(self) -> _DurationInterval:
        expected = self.end_time - self.start_time
        if abs(self.duration - expected) > 1e-3:
            raise ValueError("duration must equal end_time - start_time")
        return self

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class AnalysisSource(StrEnum):
    """Which tool produced a piece of information.

    Recorded on every derived field so a surprising result can be attributed to
    a specific model rather than to "the pipeline".
    """

    FFPROBE = "ffprobe"
    SCENE_DETECT = "pyscenedetect"
    FRAME_DIFFERENCE = "frame_difference"
    CLIP_ZERO_SHOT = "clip_zero_shot"
    RAPID_OCR = "rapid_ocr"
    OBJECT_DETECTION = "object_detection"
    SIGNAL_DSP = "signal_dsp"
    AUDIO_TAGGING = "audio_tagging"
    SPEAKER_EMBEDDING = "speaker_embedding"
    TRIBE_EVENTS = "tribe_events"
    WHISPERX = "whisperx"
    SENTENCE_EMBEDDING = "sentence_embedding"
    LEXICON_RULE = "lexicon_rule"
    HEURISTIC = "heuristic"


class Provenance(BaseModel):
    """How one derived value came to exist."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    source: AnalysisSource
    model_id: str | None = Field(
        default=None, description="Concrete model/algorithm identifier, where one applies."
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model-reported confidence. None when the tool reports none — "
        "never a fabricated number.",
    )
    notes: str | None = None


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


class VideoMetadata(BaseModel):
    """Container facts, read from ffprobe rather than assumed."""

    model_config = ConfigDict(frozen=True)

    duration_seconds: float = Field(gt=0)
    width: int | None = None
    height: int | None = None
    fps: float | None = Field(default=None, gt=0)
    frame_count: int | None = Field(default=None, ge=0)
    video_codec: str | None = None
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    has_video: bool = True
    has_audio: bool = True


# ---------------------------------------------------------------------------
# Visual
# ---------------------------------------------------------------------------


class TransitionKind(StrEnum):
    """How a shot began.

    A hard CUT and a FADE are different editorial devices, and a detector that
    reports only cuts silently omits the fades. Measured on the Sintel trailer:
    frame-difference detection found 11 boundaries and missed three fade-to-black
    transitions at 0.58 s, 10.83 s and 15.67 s, where luminance collapses to
    ~0 and consecutive frames therefore barely differ.
    """

    CUT = "cut"
    #: Luminance collapses toward black (or rises from it) across several frames.
    FADE = "fade"
    #: Content changes progressively rather than in one frame (dissolve, wipe).
    GRADUAL = "gradual"
    #: The opening of the stimulus, which is a boundary but not a transition.
    START = "start"
    UNKNOWN = "unknown"


class ShotSegment(_DurationInterval):
    """One continuous camera take, bounded by cuts."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    frame_start: int | None = Field(default=None, ge=0)
    frame_end: int | None = Field(default=None, ge=0)
    keyframe_paths: tuple[str, ...] = Field(
        default=(), description="Run-relative paths to representative frames."
    )
    transition_in: TransitionKind = Field(
        default=TransitionKind.UNKNOWN,
        description="How this shot began. See TransitionKind.",
    )
    detected_by: tuple[str, ...] = Field(
        default=(),
        description=(
            "Which detectors independently found this shot's opening boundary. "
            "Agreement between detectors is corroboration, not proof."
        ),
    )


class SceneSegment(_DurationInterval):
    """A group of consecutive shots forming one semantic unit."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    shot_indices: tuple[int, ...] = ()
    description: str | None = None
    keyframe_path: str | None = None
    grouping_reason: str | None = Field(
        default=None, description="Why these shots were grouped — the algorithm is documented."
    )


class FrameKind(StrEnum):
    """What sort of frame this is, before asking what is in it.

    Determined from evidence (OCR text coverage, image entropy), not from a CLIP
    prompt: asking a zero-shot classifier "is this a title card?" was measured to
    misfire badly on dark photographic frames.
    """

    PHOTOGRAPHIC = "photographic"
    GRAPHIC = "graphic"
    BLANK = "blank"
    UNKNOWN = "unknown"


class VisualAttributes(BaseModel):
    """Structured scene content for one keyframe.

    Deliberately structured rather than prose: downstream comparison needs
    fields it can diff, not sentences it must parse.
    """

    model_config = ConfigDict(frozen=True)

    time: float = Field(ge=0)
    keyframe_path: str | None = None
    frame_kind: FrameKind = Field(
        default=FrameKind.UNKNOWN,
        description="Graphic and blank frames have no meaningful 'setting'.",
    )
    setting: str | None = Field(
        default=None,
        description="None when the classifier abstained or the frame is not photographic.",
    )
    setting_confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Margin-derived, NOT softmax. Softmax over a closed label set is "
        "inflated: it must sum to 1 even when no label fits.",
    )
    setting_margin: float | None = Field(
        default=None,
        description="Raw cosine gap between the top two labels. Small means the "
        "classifier had no real preference.",
    )
    setting_null_gap: float | None = Field(
        default=None,
        description=(
            "How much the best setting label outscored a bank of control prompts "
            "describing nothing visual. The top-two margin measures whether the "
            "model can separate its labels; this measures whether ANY of them fit "
            "better than nonsense. They fail on different frames, so both gate the "
            "reported setting."
        ),
    )
    abstained: bool = Field(
        default=False,
        description="True when no label was confident enough to report. An honest "
        "absence, not a failure.",
    )
    abstain_reason: str | None = None
    subjects: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    person_present: bool | None = None
    text_present: bool | None = None
    shot_scale: str | None = Field(
        default=None, description="close-up / medium / wide, as a zero-shot label."
    )
    labels: dict[str, float] = Field(
        default_factory=dict,
        description="Raw label -> similarity score, so a ranking can be re-derived.",
    )
    provenance: Provenance


class TextOverlay(_TimedInterval):
    """On-screen text, temporally deduplicated across consecutive frames."""

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    occurrences: int = Field(default=1, ge=1)
    provenance: Provenance


class ObjectAppearance(BaseModel):
    """When a recognised object was on screen, and for how long."""

    model_config = ConfigDict(frozen=True)

    label: str
    first_seen: float = Field(ge=0)
    last_seen: float = Field(ge=0)
    screen_time_seconds: float = Field(ge=0)
    detection_count: int = Field(ge=1)
    mean_confidence: float | None = Field(default=None, ge=0, le=1)
    provenance: Provenance


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class AudioSegmentType(StrEnum):
    SPEECH = "speech"
    MUSIC = "music"
    SILENCE = "silence"
    OTHER = "other"


class AudioSegment(_TimedInterval):
    """A stretch of audio with one dominant character."""

    model_config = ConfigDict(frozen=True)

    segment_type: AudioSegmentType
    mean_energy: float = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    provenance: Provenance


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------


class WordTiming(_TimedInterval):
    """One spoken word and when it was said."""

    model_config = ConfigDict(frozen=True)

    text: str


class TranscriptSegment(_TimedInterval):
    """A sentence or utterance with timing, and its words where available."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    text: str
    speaker: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    overlap_probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Probability that more than one person speaks during this utterance. "
            "Diarization assigns exactly one speaker, so an elevated value means "
            "`speaker` may name only the dominant voice. None means it was not "
            "scored. This is a continuous score, not a detection: the underlying "
            "model's two-speaker classes rank real overlap correctly but never win "
            "a hard argmax on measured material."
        ),
    )
    words: tuple[WordTiming, ...] = ()
    provenance: Provenance


class ContentTone(StrEnum):
    """Tone OF THE CONTENT. Never a claim about a viewer's state."""

    INFORMATIONAL = "informational"
    MOTIVATIONAL = "motivational"
    URGENT = "urgent"
    NARRATIVE = "narrative"
    INSTRUCTIONAL = "instructional"
    CONVERSATIONAL = "conversational"
    HUMOROUS = "humorous"
    UNKNOWN = "unknown"


class StructuralRole(StrEnum):
    """Coarse position in the content's own arc."""

    INTRODUCTION = "introduction"
    DEVELOPMENT = "development"
    CONCLUSION = "conclusion"
    UNKNOWN = "unknown"


class LanguageAnalysis(_TimedInterval):
    """Structured reading of one transcript segment."""

    model_config = ConfigDict(frozen=True)

    segment_index: int = Field(ge=0)
    text: str
    tone: ContentTone = ContentTone.UNKNOWN
    tone_confidence: float | None = Field(default=None, ge=0, le=1)
    is_question: bool = False
    is_call_to_action: bool = False
    cta_confidence: float | None = Field(default=None, ge=0, le=1)
    contains_numeric_claim: bool = False
    urgency_markers: tuple[str, ...] = ()
    structural_role: StructuralRole = StructuralRole.UNKNOWN
    provenance: Provenance


class CallToAction(_TimedInterval):
    """A likely call to action, from speech or on-screen text."""

    model_config = ConfigDict(frozen=True)

    text: str
    modality: str = Field(description="'speech' or 'on_screen_text'")
    confidence: float | None = Field(default=None, ge=0, le=1)
    matched_pattern: str | None = None
    provenance: Provenance


# ---------------------------------------------------------------------------
# Fused events
# ---------------------------------------------------------------------------


class ContentEventType(StrEnum):
    """Kept deliberately small. More types means more ways to disagree."""

    SHOT = "shot"
    SCENE = "scene"
    SPEECH_SEGMENT = "speech_segment"
    TEXT_OVERLAY = "text_overlay"
    MUSIC_SEGMENT = "music_segment"
    SILENCE = "silence"
    CTA = "cta"
    FUSED_INTERVAL = "fused_interval"


class ContentEvent(_TimedInterval):
    """One interval of the stimulus, described across whatever modalities apply.

    Not every event carries every modality — a silence has no speech, a text
    overlay has no music. Absent means "not present", not "not analysed";
    `modalities` records which pipelines actually contributed.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    event_type: ContentEventType
    modalities: tuple[str, ...] = ()

    visual_description: str | None = None
    speech_text: str | None = None
    on_screen_text: tuple[str, ...] = ()
    audio_description: str | None = None
    semantic_description: str | None = None

    objects: tuple[str, ...] = ()
    speakers: tuple[str, ...] = ()

    visual_features: dict[str, float] = Field(default_factory=dict)
    audio_features: dict[str, float] = Field(default_factory=dict)
    language_features: dict[str, float | bool | str] = Field(default_factory=dict)

    shot_indices: tuple[int, ...] = ()
    scene_index: int | None = None

    confidence: float | None = Field(default=None, ge=0, le=1)
    provenance: tuple[Provenance, ...] = ()

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ---------------------------------------------------------------------------
# Neural association
# ---------------------------------------------------------------------------


class NeuralContentAssociation(BaseModel):
    """Content that occurred around the same stimulus interval as a neural event.

    TEMPORAL CO-OCCURRENCE ONLY. This asserts that certain content features were
    present near a predicted response change — not that they produced it. The
    field names, the notice, and every UI string that renders this must preserve
    that distinction.
    """

    model_config = ConfigDict(frozen=True)

    neural_event_id: str
    neural_event_type: str
    neural_time: float
    neural_score: float

    window_start: float = Field(ge=0)
    window_end: float = Field(ge=0)
    context_window_seconds: float = Field(
        ge=0, description="Half-width applied around the neural event time."
    )

    content_event_ids: tuple[str, ...] = ()
    visual_context: tuple[str, ...] = ()
    speech_context: tuple[str, ...] = ()
    on_screen_text_context: tuple[str, ...] = ()
    audio_context: tuple[str, ...] = ()
    semantic_context: tuple[str, ...] = ()

    interpretation: str = Field(
        default=(
            "These content features occurred around the same stimulus interval as a "
            "predicted cortical response change. This is temporal co-occurrence, not "
            "evidence that the content produced the response."
        )
    )


class FeatureCorrelation(BaseModel):
    """Exploratory correlation between a content feature and a neural metric."""

    model_config = ConfigDict(frozen=True)

    content_feature: str
    neural_metric: str
    method: str = Field(description="'pearson' or 'spearman'")
    coefficient: float
    p_value: float | None = None
    sample_count: int = Field(ge=0)
    lag_seconds: float = Field(
        default=0.0,
        description="Exploratory temporal offset applied before correlating. Phase 1 "
        "outputs are already stimulus-aligned; a non-zero result is not a BOLD-delay "
        "estimate and is not evidence of causation.",
    )
    lags_tested: int = Field(
        default=1,
        ge=1,
        description="How many offsets were searched. The reported one is the best of "
        "these, which biases the coefficient upward.",
    )
    comparisons: int = Field(
        default=1,
        ge=1,
        description="Total correlations computed in this run (pairs x offsets) — the "
        "multiple-comparison burden a reader must account for.",
    )
    p_value_bonferroni: float | None = Field(
        default=None,
        description="p_value x comparisons, capped at 1. Conservative, but it is the "
        "honest scale of the multiplicity: with hundreds of tests on ~10 samples, "
        "almost nothing should survive.",
    )
    q_value_bh: float | None = Field(
        default=None,
        description=(
            "Benjamini-Hochberg FDR-adjusted p-value, computed over EVERY test "
            "performed (pairs x offsets), not only the reported best-per-pair. "
            "It controls the expected PROPORTION of false positives among results "
            "called significant; Bonferroni controls the probability of ANY false "
            "positive. Both are reported because they answer different questions, "
            "and for exploratory work FDR is the usual criterion."
        ),
    )
    interpretation: str = Field(
        default=(
            "Exploratory correlation over a small number of samples, reported at the "
            "best of several lags. The raw p-value ignores both that selection and the "
            "multiple-comparison burden; q_value_bh (FDR) and p_value_bonferroni "
            "show the honest scale. Not a causal effect."
        )
    )


# ---------------------------------------------------------------------------
# Metrics, arrays, result
# ---------------------------------------------------------------------------


class ContentMetrics(BaseModel):
    """Quantitative summary — the numbers Phase 5 will diff between variants."""

    model_config = ConfigDict(frozen=True)

    duration_seconds: float = Field(ge=0)
    shot_count: int = Field(ge=0)
    mean_shot_duration: float | None = Field(default=None, ge=0)
    median_shot_duration: float | None = Field(default=None, ge=0)
    cuts_per_minute: float | None = Field(default=None, ge=0)
    scene_count: int = Field(ge=0)

    speech_fraction: float | None = Field(default=None, ge=0, le=1)
    silence_fraction: float | None = Field(default=None, ge=0, le=1)
    music_fraction: float | None = Field(default=None, ge=0, le=1)
    word_count: int = Field(default=0, ge=0)
    words_per_minute: float | None = Field(default=None, ge=0)

    text_overlay_count: int = Field(default=0, ge=0)
    cta_count: int = Field(default=0, ge=0)
    first_cta_time: float | None = None

    mean_motion: float | None = Field(default=None, ge=0)
    mean_audio_energy: float | None = Field(default=None, ge=0)
    mean_brightness: float | None = Field(default=None, ge=0)

    content_event_count: int = Field(default=0, ge=0)


class ContentArrayArtifact(BaseModel):
    """Points at the dense time-series bundle."""

    model_config = ConfigDict(frozen=True)

    path: str
    keys: dict[str, str]
    frame_count: int = Field(ge=0)
    feature_names: tuple[str, ...] = ()


class ContentAnalysisMetadata(BaseModel):
    """Everything needed to reproduce, cache, and audit an analysis."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    analysis_version: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    stimulus_sha256: str
    cache_key: str
    models: dict[str, str] = Field(
        default_factory=dict, description="Stage -> model identifier actually used."
    )
    configuration: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    warnings: tuple[str, ...] = Field(
        default=(),
        description="Modalities that failed or degraded. A partial analysis is "
        "returned with warnings rather than hidden or faked.",
    )
    failed_stages: tuple[str, ...] = Field(
        default=(),
        description="Stages that raised and were isolated. Such results are not cacheable.",
    )
    api_cost: dict[str, float] = Field(
        default_factory=dict,
        description="Token/request counts and estimated cost when a paid API is used. "
        "Empty when everything ran locally.",
    )
    interpretation_notice: str = Field(
        default=(
            "Automated description of the stimulus itself. Content labels are model "
            "outputs with stated confidence, not ground truth, and say nothing about "
            "any viewer's perception, attention, or intent."
        )
    )


class ContentAnalysisResult(BaseModel):
    """The Phase 4 contract, persisted as content_analysis/metadata.json."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    run_id: str
    stimulus_id: str
    stimulus_filename: str
    media: VideoMetadata
    modalities_analysed: tuple[str, ...] = ()

    shots: tuple[ShotSegment, ...] = ()
    scenes: tuple[SceneSegment, ...] = ()
    visual_attributes: tuple[VisualAttributes, ...] = ()
    text_overlays: tuple[TextOverlay, ...] = ()
    objects: tuple[ObjectAppearance, ...] = ()

    audio_segments: tuple[AudioSegment, ...] = ()
    audio_events: tuple[dict[str, float | str], ...] = Field(
        default=(),
        description="Discrete audio events (applause, impacts, vehicles) with "
        "AudioSet labels and confidences. Medium labels like Music are reported "
        "as segments instead.",
    )

    transcript: tuple[TranscriptSegment, ...] = ()
    language: tuple[LanguageAnalysis, ...] = ()
    calls_to_action: tuple[CallToAction, ...] = ()

    events: tuple[ContentEvent, ...] = ()
    associations: tuple[NeuralContentAssociation, ...] = ()
    correlations: tuple[FeatureCorrelation, ...] = ()

    metrics: ContentMetrics
    arrays: ContentArrayArtifact | None = None
    metadata: ContentAnalysisMetadata
