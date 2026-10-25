# Release provenance

This release is based on [adnanqidwai/RUBICON_ARR](https://github.com/adnanqidwai/RUBICON_ARR) at commit `59bc3beb0511b0f873cd31cf19de54030ca2c82f` (March 17, 2026).
The development repository is [sri-ja/hungary](https://github.com/sri-ja/hungary); the compared snapshot was `5230746eeaab29a6c693e0530b68d9ace63aa774`.

The puzzle engines, generators, oracle logic and prompt construction code are preserved from the artifact. Release changes improve installation, dataset distribution, evaluation launchers, documentation and validation. Packaged dataset JSON and image bytes are unchanged; `dataset-manifest.json` records their SHA-256 hashes. The Hugging Face Parquet representation embeds the same image bytes and serializes heterogeneous ground-truth and metadata objects as JSON strings.

The source snapshots did not include a license file. This release does not invent or assign a license; public availability alone does not grant a reuse license. Contact the maintainers through repository issues for licensing questions.

The banner is conceptual AI-generated artwork, not a benchmark example or result.

Generated demo-output images from the source checkout are omitted from the code release; the demo commands regenerate them. The original evaluation images are distributed in the dataset archive.
