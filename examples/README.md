# Example stimuli

Media files are **not committed** (`examples/media/` is gitignored): they are
large, and most real content carries licensing we cannot redistribute.

## Generate a synthetic test clip

No copyright concerns, deterministic, useful for smoke-testing the pipeline:

```bash
mkdir -p examples/media
ffmpeg -y \
  -f lavfi -i "testsrc2=size=320x240:rate=24:duration=20" \
  -f lavfi -i "sine=frequency=440:duration=20" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
  examples/media/synthetic_20s.mp4
```

This clip contains **no speech**. TRIBE's transcription step therefore produces
no `Word` events and the text extractor is dropped — which conveniently means it
runs without gated Llama-3.2 access, but also means it exercises only two of the
model's three modalities.

## A clip with speech

To exercise the full trimodal path you need spoken content and `HF_TOKEN`
configured. Meta's own demo uses the Sintel trailer (Creative Commons):

```bash
curl -L -o examples/media/sintel_trailer.mp4 \
  https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4
```

Trim to a window that actually contains speech — CPU cost scales with duration,
and only spoken content exercises the text modality. The densest speech in this
trailer starts at 41.5 s:

```bash
ffmpeg -y -ss 41.5 -i examples/media/sintel_trailer.mp4 -t 4 \
       -c:v libx264 -pix_fmt yuv420p -c:a aac \
       examples/media/sintel_speech_4s.mp4
```

That 4-second clip yields 10 `Word` events and is the reference stimulus used in
`docs/phase1_benchmark.md`.

## A text stimulus

```bash
printf 'The product arrives tomorrow. You will not believe what happens next.\n' \
  > examples/media/script.txt
```

TRIBE synthesises speech from text (gTTS) and transcribes it back to recover
word timings, so text stimuli always require working transcription.

## Running one

```bash
blackmirror predict examples/media/synthetic_20s.mp4
blackmirror predict --backend mock examples/media/synthetic_20s.mp4   # no model needed
```

## Supported formats

| Modality | Extensions |
|---|---|
| Video | `.mp4 .avi .mkv .mov .webm` |
| Audio | `.wav .mp3 .flac .ogg` |
| Text | `.txt` |

## Cost warning

On CPU, expect **tens of minutes** for a ~10 s clip, dominated by V-JEPA2 ViT-g
video encoding. Start short. On first use the frozen encoders download over
10 GB.
