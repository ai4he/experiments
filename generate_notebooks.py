"""Generators for the Multimodal LLM lab and homework notebooks."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent


def make_markdown(text: str) -> dict:
    text = dedent(text).lstrip("\n")
    lines = text.splitlines(True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def make_code(text: str) -> dict:
    text = dedent(text).lstrip("\n")
    lines = text.splitlines(True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if lines:
        leading_spaces = len(lines[0]) - len(lines[0].lstrip(" "))
        if leading_spaces:
            prefix = " " * leading_spaces
            lines = [line[len(prefix):] if line.startswith(prefix) else line for line in lines]
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


DEFAULT_METADATA = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python", "version": "3.10"},
}


def build_lab_notebook(date_str: str | None = None) -> dict:
    date_str = date_str or datetime.now(UTC).date().isoformat()
    cells = [
        make_markdown(
            f"""
            # Multimodal Large Language Models — Lab Notebook
            Applied Data Science (CPSC 8xxx) · Clemson University
            Instructor: _[Your Name]_
            Date: {date_str}

            This notebook accompanies the 75-minute lecture on multimodal large language models. It is designed to run on the Clemson Palmetto cluster and demonstrates how to stage data, configure training jobs, and fine-tune state-of-the-art multimodal foundation models.
            """
        ),
        make_markdown(
            """
            ## Learning Outcomes
            By working through this lab you will:

            - Stand up a reproducible multimodal training environment on Palmetto.
            - Implement data pipelines that combine text, vision, and audio modalities.
            - Train a compact multimodal encoder-decoder model from scratch on a curated dataset.
            - Fine-tune open-source any-to-any assistants using parameter-efficient techniques.
            - Evaluate, profile, and monitor multimodal workloads responsibly.
            """
        ),
        make_markdown(
            """
            ## Cluster Prerequisites
            - Palmetto account with access to GPU partitions (preferably `gpu-a100` or `gpu-v100`).
            - An interactive or batch allocation via SLURM.
            - Conda (Miniconda or Mamba) installed under your home directory.
            - Sufficient quota on `/scratch1` for temporary datasets (~200 GB recommended).
            """
        ),
        make_markdown(
            """
            ## Session Outline
            1. Environment bootstrap on Palmetto
            2. Data staging & multimodal manifest creation
            3. Training a compact vision-language model from scratch
            4. Fine-tuning LLaVA-style instruction followers
            5. Extending to audio-text and video-text adapters
            6. Evaluation, monitoring, and safety audits
            """
        ),
        make_code(
            """
            # Check GPU availability (run inside an interactive GPU session)
            !nvidia-smi
            """
        ),
        make_markdown(
            """
            ### 1. Environment Bootstrap
            Use a dedicated Conda environment per experiment to keep dependencies isolated. The snippet below assumes the CUDA 12 toolchain on Palmetto. Adjust the module versions as needed.
            """
        ),
        make_code(
            """
            %%bash
            module purge
            module load gcc/11.3.0 cuda/12.1.0 cudnn/8.9.2.26
            source $HOME/miniconda3/etc/profile.d/conda.sh
            conda create -y -n multimodal-llm python=3.10
            conda activate multimodal-llm
            pip install --upgrade pip
            pip install 'torch==2.1.*' --index-url https://download.pytorch.org/whl/cu121
            pip install accelerate transformers datasets einops timm bitsandbytes peft open_clip_torch torchvision torchaudio
            pip install soundfile decord webdataset pillow scipy wandb mlflow rich
            """
        ),
        make_markdown(
            """
            > **Tip:** For repeated jobs, bake the environment into a Palmetto Singularity container or use Conda-pack to speed up startup.
            """
        ),
        make_markdown(
            """
            ### 2. Data Staging & Manifest Creation
            For this lab we will sample a small but diverse multimodal corpus:
            - Image-text pairs: subset of [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/)
            - Instructional dialogues: LLaVA-Instruct 150K
            - Audio-caption pairs: AudioCaps + synthetic prompts

            Stage the raw archives onto `/scratch1/$USER/datasets/multimodal` using `rclone`, `aws s3 cp`, or the Clemson Globus endpoint.
            """
        ),
        make_code(
            """
            import os
            from pathlib import Path

            data_root = Path('/scratch1') / os.environ.get('USER', 'student') / 'datasets' / 'multimodal'
            print('Dataset root:', data_root)

            data_root.mkdir(parents=True, exist_ok=True)

            image_text_manifest = data_root / 'laion_subset.tsv'
            if not image_text_manifest.exists():
                with image_text_manifest.open('w') as f:
                    f.write('url\tcaption\n')
                    f.write('https://example.org/image1.jpg\tA robot assembling solar panels on campus.\n')
                    f.write('https://example.org/image2.jpg\tStudents collaborating with an interactive whiteboard.\n')
            print('Created manifest:', image_text_manifest)
            """
        ),
        make_markdown(
            """
            #### Download helpers
            Use `multiprocessing` or SLURM array jobs to fetch shards in parallel. The following utility script expects a TSV manifest and stores files under `/scratch1`.
            """
        ),
        make_code(
            """
            import concurrent.futures as futures
            import requests

            session = requests.Session()
            session.headers['User-Agent'] = 'Clemson-Multimodal-Lab/1.0'

            def download_example(row):
                url, caption = row
                target = data_root / 'images' / Path(url).name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    return target
                try:
                    resp = session.get(url, timeout=10)
                    resp.raise_for_status()
                    target.write_bytes(resp.content)
                    return target
                except Exception as exc:
                    print('Failed to fetch', url, exc)
                    return None

            rows = [('https://example.org/image1.jpg', 'A robot assembling solar panels on campus.')]
            with futures.ThreadPoolExecutor(max_workers=8) as executor:
                for result in executor.map(download_example, rows):
                    print('Downloaded ->', result)
            """
        ),
        make_markdown(
            """
            ### 3. Warm-Up: Zero-Shot CLIP Retrieval
            Before training from scratch, verify the environment by running inference with OpenCLIP.
            """
        ),
        make_code(
            """
            import torch
            import open_clip
            from PIL import Image
            import requests
            from io import BytesIO

            model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device='cuda')
            tokenizer = open_clip.get_tokenizer('ViT-H-14')

            candidate_urls = [
                'https://images.unsplash.com/photo-1522199994827-8f68f1d1d8c5',
                'https://images.unsplash.com/photo-1581092152835-30ab079f19b9',
            ]
            images = []
            for url in candidate_urls:
                img = Image.open(BytesIO(requests.get(url).content)).convert('RGB')
                images.append(preprocess(img))
            image_tensor = torch.stack(images).cuda()
            texts = tokenizer([
                'Students collaborating with augmented reality interfaces',
                'Industrial robot arm assembling circuit boards'
            ])
            text_tensor = texts.cuda()
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text_tensor)
                logits = (image_features @ text_features.T) / model.logit_scale.exp()
            print('Similarity matrix:', logits.softmax(dim=-1).cpu())
            """
        ),
        make_markdown(
            """
            ### 4. Training a Compact Vision-Language Model from Scratch
            We construct a dual-encoder similar to CLIP but sized for a classroom-scale dataset.
            """
        ),
        make_code(
            """
            from dataclasses import dataclass
            import os
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torchvision import transforms
            from datasets import load_dataset

            @dataclass
            class TrainingConfig:
                vision_model: str = 'openai/clip-vit-base-patch16'
                text_model: str = 'distilbert-base-uncased'
                batch_size: int = 256
                lr: float = 5e-4
                warmup_steps: int = 500
                total_steps: int = 5000
                log_every: int = 50
                output_dir: str = f"/scratch1/{os.environ.get('USER', 'student')}/experiments/clip-scratch"
                bf16: bool = True

            config = TrainingConfig()
            print(config)

            dataset = load_dataset('laion/laion400m', split='train[:0.02%]', streaming=True)

            vision_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
            """
        ),
        make_code(
            """
            from transformers import AutoModel, AutoTokenizer

            vision_encoder = AutoModel.from_pretrained(config.vision_model)
            text_encoder = AutoModel.from_pretrained(config.text_model)
            text_tokenizer = AutoTokenizer.from_pretrained(config.text_model)

            vision_encoder.train()
            text_encoder.train()

            projector = nn.Linear(vision_encoder.config.hidden_size, text_encoder.config.hidden_size)

            if torch.cuda.is_available():
                vision_encoder.cuda()
                text_encoder.cuda()
                projector.cuda()
            """
        ),
        make_code(
            """
            import itertools
            from accelerate import Accelerator

            accelerator = Accelerator(mixed_precision='bf16' if config.bf16 else 'no')

            params = itertools.chain(vision_encoder.parameters(), text_encoder.parameters(), projector.parameters())
            optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=0.01)

            temperature = torch.nn.Parameter(torch.tensor(0.07))
            if torch.cuda.is_available():
                temperature = temperature.to(accelerator.device)

            vision_encoder, text_encoder, projector, optimizer, temperature = accelerator.prepare(
                vision_encoder, text_encoder, projector, optimizer, temperature
            )

            step = 0
            for batch in dataset.take(config.total_steps):
                images = batch['image']
                captions = batch['caption']
                pixel_values = torch.stack([vision_transform(img) for img in images])
                tokenized = text_tokenizer(list(captions), return_tensors='pt', padding=True, truncation=True)
                pixel_values = pixel_values.to(accelerator.device)
                tokenized = {k: v.to(accelerator.device) for k, v in tokenized.items()}

                image_feats = vision_encoder(pixel_values=pixel_values).pooler_output
                text_feats = text_encoder(**tokenized).last_hidden_state[:, 0]
                image_feats = projector(image_feats)

                image_feats = F.normalize(image_feats, dim=-1)
                text_feats = F.normalize(text_feats, dim=-1)
                logits = (image_feats @ text_feats.t()) / temperature.exp()
                targets = torch.arange(logits.size(0), device=logits.device)
                loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if step % config.log_every == 0:
                    accelerator.print(f"step={step} loss={loss.item():.4f}")
                step += 1
                if step >= config.total_steps:
                    break

            accelerator.print('Training completed')
            """
        ),
        make_markdown(
            """
            ### 5. Checkpointing & Evaluation
            Persist checkpoints to `/scratch1/$USER/experiments` and periodically evaluate on COCO or Flickr30k. Use Weights & Biases or MLflow for tracking.
            """
        ),
        make_code(
            """
            from pathlib import Path

            save_dir = Path(config.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            accelerator.wait_for_everyone()
            unwrap_vision = accelerator.unwrap_model(vision_encoder)
            unwrap_text = accelerator.unwrap_model(text_encoder)

            accelerator.save_state(save_dir / 'accelerate_state')
            unwrap_vision.save_pretrained(save_dir / 'vision_encoder')
            unwrap_text.save_pretrained(save_dir / 'text_encoder')
            """
        ),
        make_markdown(
            """
            ### 6. Visual Instruction Fine-Tuning (LLaVA-1.5 style)
            We reuse the pretrained CLIP vision tower, attach a projection MLP, and supervise with multimodal conversation data.
            """
        ),
        make_code(
            """
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM

            llm = AutoModelForCausalLM.from_pretrained('liuhaotian/llava-v1.5-7b', device_map='auto', torch_dtype=torch.bfloat16)
            vision_projector = nn.Sequential(
                nn.Linear(vision_encoder.config.hidden_size, llm.config.hidden_size),
                nn.GELU(),
                nn.Linear(llm.config.hidden_size, llm.config.hidden_size)
            )
            vision_projector = get_peft_model(vision_projector, LoraConfig(r=8, lora_alpha=16, target_modules=['0', '2']))
            """
        ),
        make_code(
            """
            batch = {
                'pixel_values': torch.randn(2, 3, 224, 224, device=llm.device),
                'input_ids': torch.ones((2, 128), dtype=torch.long, device=llm.device),
                'attention_mask': torch.ones((2, 128), dtype=torch.long, device=llm.device),
                'labels': torch.ones((2, 128), dtype=torch.long, device=llm.device),
            }
            outputs = llm(**batch)
            loss = outputs.loss
            loss.backward()
            """
        ),
        make_markdown(
            """
            ### 7. Audio-Text Alignment Adapter
            We extend the framework to speech prompts using Whisper encoder features.
            """
        ),
        make_code(
            """
            from transformers import WhisperProcessor, WhisperModel

            processor = WhisperProcessor.from_pretrained('openai/whisper-small')
            whisper = WhisperModel.from_pretrained('openai/whisper-small')

            waveform = torch.randn(1, 16000 * 10)
            inputs = processor(waveform, sampling_rate=16000, return_tensors='pt')
            audio_features = whisper.encoder(inputs.input_features).last_hidden_state.mean(dim=1)
            audio_to_text = nn.Linear(audio_features.size(-1), llm.config.hidden_size)
            audio_embeddings = audio_to_text(audio_features)
            print(audio_embeddings.shape)
            """
        ),
        make_markdown(
            """
            ### 8. Video Token Resampler
            For video-text tasks, subsample frames and use a temporal adapter.
            """
        ),
        make_code(
            """
            import decord
            from decord import VideoReader

            vr = VideoReader('sample.mp4', num_threads=1)
            frame_ids = list(range(0, len(vr), max(len(vr)//16, 1)))[:16]
            frames = vr.get_batch(frame_ids).asnumpy()
            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2) / 255.0

            temporal_pool = nn.Conv1d(in_channels=frames_tensor.shape[0], out_channels=8, kernel_size=1)
            video_tokens = temporal_pool(frames_tensor.view(frames_tensor.shape[0], -1, frames_tensor.shape[1]*frames_tensor.shape[2]*frames_tensor.shape[3]))
            print('Video tokens shape:', video_tokens.shape)
            """
        ),
        make_markdown(
            """
            ### 9. Evaluation & Safety Audits
            - **Retrieval**: Recall@K on MSCOCO, Flickr30k.
            - **Instruction following**: MMBench, MMMU, ScienceQA.
            - **Audio/Text**: WER, BLEU, MOS.
            - **Bias/Safety**: Multimodal SafetyBench, RealToxicityPrompts with image perturbations.

            Log responsible AI metrics in the same experiment tracking dashboard.
            """
        ),
        make_markdown(
            """
            ### 10. Homework Preview
            1. Fine-tune the CLIP-like model on a Clemson campus dataset.
            2. Adapt LLaVA to support chart understanding using the ChartQA dataset.
            3. Train an audio-text adapter for campus tour narration.

            Refer to the dedicated homework notebook for detailed instructions and grading rubrics.
            """
        ),
    ]

    return {"cells": cells, "metadata": DEFAULT_METADATA, "nbformat": 4, "nbformat_minor": 5}


def build_homework_notebook(date_str: str | None = None) -> dict:
    cells = [
        make_markdown(
            """
            # Homework · Multimodal Large Language Models
            **Course:** Applied Data Science (CPSC 8xxx)
            **Due:** _One week after the lab session_
            **Submission:** Push to your private GitLab repository and submit a Palmetto job report.

            This homework builds on the lecture and lab notebook. You will fine-tune multimodal foundation models across three modality pairings. Each exercise must include:
            - SLURM script or `srun` command used on Palmetto.
            - Training/evaluation logs (TensorBoard, W&B, or MLflow).
            - Short written summary (1–2 paragraphs) interpreting results and challenges.
            """
        ),
        make_markdown(
            """
            ---
            ## Environment Checklist
            - Use the same `multimodal-llm` Conda environment from the lab notebook.
            - Reserve GPUs with at least 24 GB memory (A100 preferred).
            - Store intermediate checkpoints under `/scratch1/$USER/hw3-multimodal`.
            """
        ),
        make_markdown(
            """
            ---
            ## Exercise 1 · Vision-Language Fine-Tuning (20 pts)
            **Goal:** Fine-tune the CLIP-style dual encoder on the [Clemson Campus Scenes](https://example.org) dataset and evaluate zero-shot transfer to COCO.

            **Requirements:**
            - Implement balanced sampling to mitigate class imbalance between campus landmarks.
            - Apply parameter-efficient fine-tuning (LoRA on the projector or QLoRA on the text tower).
            - Report Recall@1/5/10 on COCO validation and Clemson Campus validation splits.
            - Analyze modality alignment drift compared to the pretraining checkpoint.
            """
        ),
        make_code(
            """
            # TODO: configure paths
            from pathlib import Path
            import os

            DATA_ROOT = Path('/scratch1') / os.environ.get('USER', 'student') / 'hw3-multimodal'
            DATA_ROOT.mkdir(parents=True, exist_ok=True)

            # TODO: implement dataset loader and balanced sampling strategy
            """
        ),
        make_code(
            """
            # TODO: load pretrained encoders and attach LoRA adapters
            from transformers import AutoModel
            from peft import LoraConfig, get_peft_model

            vision_encoder = AutoModel.from_pretrained('openai/clip-vit-base-patch16')
            text_encoder = AutoModel.from_pretrained('distilbert-base-uncased')

            lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=['query', 'value'])
            vision_encoder = get_peft_model(vision_encoder, lora_config)
            text_encoder = get_peft_model(text_encoder, lora_config)
            """
        ),
        make_code(
            """
            # TODO: training loop
            # Use Accelerate or PyTorch Lightning if preferred
            """
        ),
        make_code(
            """
            # TODO: compute retrieval metrics and save a JSON report
            """
        ),
        make_markdown(
            """
            ---
            ## Exercise 2 · Audio-Text Instruction Alignment (25 pts)
            **Goal:** Adapt Whisper-small + LLaMA-2-7B-chat to follow spoken instructions and output textual answers.

            **Dataset Suggestions:** AudioCaps, Spoken-SQuAD, campus tour recordings. Combine with synthetic speech generated via Torchaudio + TTS for augmentation.

            **Requirements:**
            - Implement a projection layer from Whisper encoder embeddings to the LLaMA hidden size.
            - Apply LoRA to the language model _or_ freezing strategy plus adapter.
            - Evaluate on a held-out set with WER and instruction-following accuracy (exact match or ROUGE-L).
            - Discuss robustness to noisy backgrounds recorded on campus.
            """
        ),
        make_code(
            """
            # TODO: prepare audio dataset manifest and dataloader
            """
        ),
        make_code(
            """
            # TODO: build audio-to-text adapter and fine-tuning loop
            """
        ),
        make_code(
            """
            # TODO: evaluation metrics (WER, instruction accuracy)
            """
        ),
        make_markdown(
            """
            ---
            ## Exercise 3 · Any-to-Any Generation (35 pts)
            **Goal:** Extend an open-source multimodal assistant (e.g., LLaVA-Next, InstructBLIP, or Qwen-VL) to support **chart-to-audio** and **audio-to-image** tasks via tool augmentation.

            **Requirements:**
            - Add at least two modality adapters (e.g., Chart OCR encoder + audio synthesizer).
            - Implement routing logic that selects the correct adapter based on the prompt.
            - Demonstrate two end-to-end examples per new capability.
            - Benchmark against a baseline without the new adapters.
            - Provide an ablation table analyzing latency and GPU memory usage under Palmetto scheduling constraints.
            """
        ),
        make_code(
            """
            # TODO: design routing policy for any-to-any interactions
            """
        ),
        make_code(
            """
            # TODO: integrate tool calls (e.g., image generation API, TTS)
            """
        ),
        make_markdown(
            """
            ---
            ## Deliverables Checklist
            - [ ] Completed code cells for all exercises.
            - [ ] `README.md` in your repository describing data sources and ethical considerations.
            - [ ] SLURM submission scripts (`*.sbatch`).
            - [ ] Evaluation summaries in `reports/`.
            - [ ] Short reflection (submit via Canvas) covering lessons learned and next steps.
            """
        ),
        make_markdown(
            """
            ---
            ## Grading Rubric
            | Component | Points | Criteria |
            | --- | --- | --- |
            | Exercise 1 | 20 | Completeness, retrieval metrics, analysis |
            | Exercise 2 | 25 | Adapter design, instruction accuracy, robustness discussion |
            | Exercise 3 | 35 | Tool integration, demonstrations, ablation |
            | Engineering Report | 10 | Clarity of SLURM scripts, logging, reproducibility |
            | Responsible AI Reflection | 10 | Bias/safety analysis, mitigation proposals |

            Late policy: -10% per day (max 3 days). Collaborations must be declared.
            """
        ),
    ]

    return {"cells": cells, "metadata": DEFAULT_METADATA, "nbformat": 4, "nbformat_minor": 5}


def write_notebook(path: Path | str, notebook: dict) -> None:
    path = Path(path)
    with path.open("w") as fh:
        json.dump(notebook, fh, indent=2)


def _parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab",
        type=Path,
        default=Path("Multimodal_LLMs_Lab.ipynb"),
        help="Output path for the lab notebook (default: %(default)s)",
    )
    parser.add_argument(
        "--homework",
        type=Path,
        default=Path("Multimodal_LLMs_Homework.ipynb"),
        help="Output path for the homework notebook (default: %(default)s)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional ISO date to embed in the lab heading (default: today)",
    )
    return parser


def main() -> None:
    args = _parse_args().parse_args()
    lab_notebook = build_lab_notebook(args.date)
    homework_notebook = build_homework_notebook(args.date)
    write_notebook(args.lab, lab_notebook)
    write_notebook(args.homework, homework_notebook)


if __name__ == "__main__":
    main()
