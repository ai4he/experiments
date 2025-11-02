# Multimodal LLMs Teaching Materials

This repository contains generators for the lecture slide deck and the two Jupyter notebooks used in the Applied Data Science course at Clemson University.

## Regenerating artifacts

The presentation is not checked into version control because binary files cannot be attached to the PR workflow. Regenerate it locally with:

```bash
python generate_pptx.py --output Multimodal_LLMs_Lecture.pptx
```

The lab and homework notebooks can be regenerated with:

```bash
python generate_notebooks.py --output-dir .
```

These commands default to deterministic timestamps so the outputs remain reproducible across runs.
