# Contrastive data curation

A 34-second, 1080p animation illustrating the refund example in the repository README. The video uses on-screen explanations, has no logo or product branding, and contains no audio track.

1. Start with a fixed authorization rule and two necessary facts.
2. Copy the example, changing only the signer from Mira to Noah.
3. Change the correct label from true to false.
4. Check that removing either necessary fact makes the answer unknown. These deletion checks are not additional false-labeled training examples.
5. Keep the contrast pair together when splitting training and evaluation data.

This is a conceptual illustration, not a recorded model run or a complete pipeline walkthrough.

Files: `contrastive-curation.mp4`, `poster.jpg`, and `storyboard.jpg`.

Rebuild with Pillow, NumPy, FFmpeg, and the macOS fonts referenced by the source:

```sh
python assets/video/source/contrastive.py --ffmpeg /path/to/ffmpeg
```

The renderer reuses drawing helpers from `assets/video/source/render.py`. Pass `--preview` to produce stills only.
