"""Deterministic candidate construction.

Every cache below Phase 9 assumes that the same edit on the same source yields
the same bytes, so determinism is tested directly rather than assumed. The
content-addressed filename is tested too, because the feature cache under TRIBE
keys on path rather than content: a reused path with new content would score
one candidate using another's features and nothing would report it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from blackmirror.materialization.adapter import binding_from_result
from blackmirror.materialization.ffmpeg import (
    MaterializationError,
    build_command,
    candidate_key,
    materialize,
)
from blackmirror.materialization.schemas import (
    EditOperation,
    EditStep,
    MaterializationPlan,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="materialization requires ffmpeg and ffprobe on PATH",
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny deterministic clip with both streams, built once per module."""
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-nostdin", "-fflags", "+bitexact",
            "-f", "lavfi", "-i", "testsrc=size=128x96:rate=10:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-x264-params", "threads=1", "-c:a", "aac", "-b:a", "64k",
            "-map_metadata", "-1", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _plan(clip: Path, **steps: float) -> MaterializationPlan:
    return MaterializationPlan(
        source_path=clip,
        steps=tuple(EditStep(operation=EditOperation(k), value=v) for k, v in steps.items()),
    )


class TestPlanValidation:
    def test_rejects_a_plan_that_would_reproduce_the_source(self, clip: Path) -> None:
        """A neutral plan costs a full inference to learn nothing."""
        with pytest.raises(ValueError, match="byte for byte"):
            _plan(clip, audio_gain_db=0.0, contrast=1.0)

    def test_rejects_a_repeated_operation(self, clip: Path) -> None:
        with pytest.raises(ValueError, match="at most once"):
            MaterializationPlan(
                source_path=clip,
                steps=(
                    EditStep(operation=EditOperation.BRIGHTNESS, value=0.1),
                    EditStep(operation=EditOperation.BRIGHTNESS, value=0.2),
                ),
            )

    def test_rejects_values_outside_the_declared_range(self, clip: Path) -> None:
        with pytest.raises(ValueError, match=r"within \[-30.0, 30.0\]"):
            EditStep(operation=EditOperation.AUDIO_GAIN_DB, value=99.0)

    def test_fingerprint_ignores_neutral_steps_and_ordering(self, clip: Path) -> None:
        """Two plans that do the same thing are the same candidate.

        Without this, a search would evaluate the identical edit twice because
        one description happened to carry a do-nothing step.
        """
        a = _plan(clip, audio_gain_db=3.0, contrast=1.0)
        b = _plan(clip, audio_gain_db=3.0)

        assert a.fingerprint() == b.fingerprint()

    def test_different_edits_have_different_fingerprints(self, clip: Path) -> None:
        assert _plan(clip, audio_gain_db=3.0).fingerprint() != _plan(
            clip, audio_gain_db=3.5
        ).fingerprint()


class TestMaterialization:
    def test_produces_a_file_that_differs_from_its_source(
        self, clip: Path, tmp_path: Path
    ) -> None:
        result = materialize(_plan(clip, audio_gain_db=6.0), tmp_path)

        assert result.variant_path.is_file()
        assert result.variant_sha256 != result.source_sha256
        assert result.reused_existing is False

    def test_is_deterministic(self, clip: Path, tmp_path: Path) -> None:
        """The property every cache in Phase 9 depends on."""
        first = materialize(_plan(clip, audio_gain_db=6.0), tmp_path / "a")
        second = materialize(_plan(clip, audio_gain_db=6.0), tmp_path / "b")

        assert first.variant_sha256 == second.variant_sha256

    def test_video_edits_are_deterministic_too(self, clip: Path, tmp_path: Path) -> None:
        """Re-encoding is the harder case: x264 is threaded by default."""
        first = materialize(_plan(clip, brightness=0.1, contrast=1.2), tmp_path / "a")
        second = materialize(_plan(clip, brightness=0.1, contrast=1.2), tmp_path / "b")

        assert first.variant_sha256 == second.variant_sha256

    def test_filename_is_content_addressed(self, clip: Path, tmp_path: Path) -> None:
        """Different edits must never share a path.

        The feature cache under TRIBE is keyed on file path and time range, not
        content. Two candidates at one path would silently share features.
        """
        louder = materialize(_plan(clip, audio_gain_db=6.0), tmp_path)
        quieter = materialize(_plan(clip, audio_gain_db=-6.0), tmp_path)

        assert louder.variant_path != quieter.variant_path
        assert louder.candidate_key != quieter.candidate_key

    def test_identical_plan_reuses_the_existing_file(
        self, clip: Path, tmp_path: Path
    ) -> None:
        first = materialize(_plan(clip, audio_gain_db=6.0), tmp_path)
        mtime = first.variant_path.stat().st_mtime_ns

        second = materialize(_plan(clip, audio_gain_db=6.0), tmp_path)

        assert second.reused_existing is True
        assert second.variant_path.stat().st_mtime_ns == mtime
        assert second.variant_sha256 == first.variant_sha256

    def test_speed_changes_duration_and_gain_does_not(
        self, clip: Path, tmp_path: Path
    ) -> None:
        gain = materialize(_plan(clip, audio_gain_db=6.0), tmp_path)
        faster = materialize(_plan(clip, speed=2.0), tmp_path)

        assert abs(gain.duration_change_seconds) < 0.05
        assert faster.duration_change_seconds < -0.2
        assert faster.plan.changes_duration is True
        assert gain.plan.changes_duration is False

    def test_missing_source_is_reported_clearly(self, tmp_path: Path) -> None:
        plan = MaterializationPlan(
            source_path=tmp_path / "absent.mp4",
            steps=(EditStep(operation=EditOperation.AUDIO_GAIN_DB, value=3.0),),
        )

        with pytest.raises(MaterializationError, match="does not exist"):
            materialize(plan, tmp_path)

    def test_untouched_stream_is_copied_not_re_encoded(self, clip: Path) -> None:
        """A candidate must differ from its parent only as the plan says.

        Re-encoding an unedited stream would change its bytes for no reason and
        put an uncontrolled difference into the comparison.
        """
        audio_only = build_command(
            clip, clip.with_name("out.mp4"), _plan(clip, audio_gain_db=3.0), has_audio=True
        )
        visual_only = build_command(
            clip, clip.with_name("out.mp4"), _plan(clip, brightness=0.1), has_audio=True
        )

        assert "-c:v" in audio_only and audio_only[audio_only.index("-c:v") + 1] == "copy"
        assert "-c:a" in visual_only and visual_only[visual_only.index("-c:a") + 1] == "copy"

    def test_candidate_key_binds_the_edit_to_the_exact_source(
        self, clip: Path, tmp_path: Path
    ) -> None:
        """The same edit on different content is a different candidate."""
        plan = _plan(clip, audio_gain_db=3.0)

        assert candidate_key(plan, "a" * 64) != candidate_key(plan, "b" * 64)


class TestPhase8Binding:
    def test_binding_records_machine_provenance_not_human(
        self, clip: Path, tmp_path: Path
    ) -> None:
        """A generated file must not be recorded as user-supplied media."""
        result = materialize(_plan(clip, audio_gain_db=6.0), tmp_path)

        binding = binding_from_result(result)

        assert binding.adapter_id == "ffmpeg-materialized"
        assert binding.method == "deterministic_ffmpeg_edit"
        assert binding.tool is not None and binding.tool.startswith("ffmpeg")
        assert binding.parameters["audio_gain_db"] == 6.0
        assert binding.source_sha256 == result.source_sha256
        assert binding.variant_sha256 == result.variant_sha256

    def test_binding_is_accepted_by_the_phase_8_schema(
        self, clip: Path, tmp_path: Path
    ) -> None:
        """Construction alone proves it: the model validates distinct content."""
        result = materialize(_plan(clip, brightness=0.15), tmp_path)

        binding = binding_from_result(result)

        assert binding.source_sha256 != binding.variant_sha256
        assert binding.dry_run_identical_allowed is False


class TestSearchSpaceIntegration:
    """A sampled point must survive all the way to a real file.

    This is the join the whole phase depends on: if a search space can produce
    a value the materializer rejects, the failure surfaces only after a
    strategy has already committed budget to it.
    """

    def test_a_sampled_point_becomes_a_playable_variant(
        self, clip: Path, tmp_path: Path
    ) -> None:
        from random import Random

        from blackmirror.materialization.schemas import EditOperation as Op
        from blackmirror.search.space import ContentSearchSpace, continuous

        space = ContentSearchSpace(
            parameters=(
                continuous("gain", Op.AUDIO_GAIN_DB, 3.0, 6.0),
                continuous("brightness", Op.BRIGHTNESS, 0.05, 0.2),
            )
        )
        values = space.sample(Random(3))

        plan = MaterializationPlan(source_path=clip, steps=space.to_edit_steps(values))
        result = materialize(plan, tmp_path)

        assert result.variant_path.is_file()
        assert result.variant_sha256 != result.source_sha256
        assert result.variant_duration_seconds > 0

    def test_every_sampled_point_in_the_space_is_materializable(
        self, clip: Path, tmp_path: Path
    ) -> None:
        """Sampled 200 times: no value the materializer would refuse.

        Only the plan is built, not the file: constructing it runs the same
        bounds and neutrality checks that would reject a bad point, without
        200 ffmpeg invocations.
        """
        from random import Random

        from blackmirror.materialization.schemas import EditOperation as Op
        from blackmirror.search.space import ContentSearchSpace, continuous

        space = ContentSearchSpace(
            parameters=(continuous("gain", Op.AUDIO_GAIN_DB, 1.0, 6.0),)
        )
        rng = Random(11)

        for _ in range(200):
            plan = MaterializationPlan(
                source_path=clip, steps=space.to_edit_steps(space.sample(rng))
            )
            assert plan.effective_steps
