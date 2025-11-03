"""Utility for generating the Multimodal LLM lecture deck."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from zipfile import ZipFile

slides = [
    ("Multimodal Large Language Models", ["• Applied Data Science – Clemson University", "• Instructor: [Your Name]", "• Lecture Duration: 75 minutes"]),
    ("Today's Flow", ["• Motivation & background", "• Architectures across modality pairings", "• Training from scratch vs. adapting foundation models", "• Evaluation, deployment, and responsible use", "• Preview of lab + homework"]),
    ("Learning Objectives", ["• Understand theoretical foundations behind multimodal modeling", "• Compare architectures for single, mixed, and any-to-any modality mappings", "• Design training pipelines on large-scale compute clusters", "• Plan fine-tuning strategies tailored to downstream tasks"]),
    ("Recap: Transformer Essentials", ["• Attention as modality-agnostic aggregator", "• Positional & modality embeddings", "• Sequence-to-sequence vs. decoder-only reuse", "• Scaling considerations already mastered"]),
    ("Why Multimodal Now?", ["• Explosion of aligned image/text, audio/text, and video/text corpora", "• Hardware & distributed training maturity", "• Emergence of foundation models requiring holistic perception", "• User demand for richer assistants (education, accessibility, robotics)"]),
    ("Timeline of Multimodal Milestones", ["• 2013–2015: Neural captioning and VQA pioneers", "• 2018–2020: CLIP, ALIGN, UNITER unify vision-language embeddings", "• 2021–2022: Flamingo, PaLI, AudioLM expand modalities", "• 2023–2024: Any-to-any assistants (GPT-4V, Gemini, Kosmos-2)"]),
    ("Taxonomy of Modalities", ["• Text & code (discrete tokens)", "• Vision (images, video clips, 3D point clouds)", "• Audio & speech", "• Structured sensors (tabular, time series, robotics proprioception)", "• Action spaces (control tokens, tool APIs)"]),
    ("Single-Modality Encoders: Text", ["• Pretrained LLM backbones (GPT-NeoX, LLaMA, T5)", "• Sentence-level pooling (mean, [CLS], attention pooling)", "• Domain adaptation via continued pretraining", "• Tokenizer implications for multilingual coverage"]),
    ("Single-Modality Encoders: Vision", ["• ViT / Swin Transformers as plug-and-play components", "• Patch embeddings vs. convolutional stems", "• Self-supervised pretraining (MAE, DINO, iBOT)", "• Parameter counts and memory footprints"]),
    ("Single-Modality Encoders: Audio", ["• Spectrogram tokenization (Mel, log-mel)", "• Conformer & wav2vec 2.0 backbones", "• HuBERT, Whisper as general-purpose encoders", "• Temporal resolution vs. cost trade-offs"]),
    ("Beyond the Big Three", ["• Graphs and relational data via Graphormer", "• 3D point clouds (Point-BERT, PointNeXt)", "• Multisensor fusion in robotics", "• Text-structured hybrids (tables, charts)"]),
    ("Cross-Modal Motivation", ["• Grounding language with perception", "• Few-shot generalization via shared embedding spaces", "• Enabling embodied decision making", "• Reducing hallucinations through multimodal grounding"]),
    ("Representation Learning Concepts", ["• Joint vs. coordinated embedding spaces", "• Contrastive alignment and mutual information maximization", "• Shared latent variable models", "• Conditional generation with modality-specific decoders"]),
    ("Embedding Geometry", ["• Temperature scaling in contrastive losses", "• Feature normalization and spherical manifolds", "• Modality-specific scaling factors", "• Measuring alignment: CKA, centered cosine similarity"]),
    ("Pretraining Paradigms", ["• Contrastive (InfoNCE, CLIP loss)", "• Matching & ranking (ITM, MIL-NCE)", "• Generative modeling (encoder-decoder, diffusion, masked modeling)", "• Reinforced or instruction-conditioned pretraining"]),
    ("Architecture Overview", ["• Encoder-encoder (dual towers)", "• Encoder-decoder (fusion encoder + autoregressive decoder)", "• Unified token transformers", "• Adapter-heavy foundation models"]),
    ("Early Fusion", ["• Concatenate raw or embedded modalities before Transformer", "• Useful when modalities share temporal alignment", "• Risks: modality dominance, scaling to high dimensions", "• Example: Multimodal BERT (MMBT)"]),
    ("Late Fusion", ["• Independent modality experts with shallow combination", "• Simple averaging, gating, or product of experts", "• Strength: modularity and pretraining reuse", "• Weakness: limited cross-modal interaction until late"]),
    ("Mid-Fusion & Co-Attention", ["• Hierarchical attention layers exchange features", "• Co-attention blocks (ViLBERT, LXMERT)", "• Cross-modal transformers as relation learners", "• Configurable depth per modality"]),
    ("Encoder-Decoder Patterns", ["• Vision encoder feeding language decoder (e.g., Flamingo)", "• Unified memory tokens bridging modalities", "• Prefix-tuning for decoder-only LLMs", "• Handling long context windows and multi-image prompts"]),
    ("Mixture-of-Experts for Modalities", ["• Routing tokens to modality-specialized experts", "• Switch Transformer and Router layers", "• Balancing load: auxiliary losses", "• Scaling to any-to-any by adding experts"]),
    ("Any-to-Any Interfaces", ["• Single backbone generating arbitrary modality outputs", '• Multitask instruction format ("Describe", "Answer", "Transcribe")', "• Tool use for non-text outputs (image generation APIs)", "• Case study: Gemini, Kosmos-2, LLaVA-Next"]),
    ("Tokenization Strategies", ["• Byte-pair encodings for text and code", "• Patchifying images into pseudo-tokens", "• Audio tokens via learned vector quantizers", "• Neural codecs (SoundStream, EnCodec) for generative tasks"]),
    ("Continuous Feature Adapters", ["• Projectors mapping modality embeddings to LLM dimension", "• Gated cross-attention modules", "• Low-rank adapters per modality", "• Temporal pooling modules for video"]),
    ("Data Curation", ["• Sourcing aligned pairs (CC3M, LAION-5B, WebVid2.5M)", "• Cleaning: caption quality filters, deduplication", "• Copyright and licensing considerations", "• Balanced sampling across modalities"]),
    ("Scaling Laws & Data Mixing", ["• Power-law behavior across modalities", "• Compute-optimal mixing ratios", "• Curriculum schedules (warm-up with single modality)", "• Estimating effective data diversity"]),
    ("Synthetic & Augmented Data", ["• Prompting LLMs for captions / conversations", "• Self-training with pseudo-labels", "• Generating contrastive negatives", "• Simulation environments for robotics data"]),
    ("Alignment Objectives", ["• InfoNCE contrastive loss details", "• Matching losses (binary cross-entropy)", "• Multi-positive pairing strategies", "• Temperature tuning and logit scaling"]),
    ("Generative Objectives", ["• Sequence-to-sequence cross-entropy", "• Masked multimodal modeling (MIM, MLM)", "• Diffusion for images conditioned on text", "• Autoregressive latent modeling (VQ-VAE tokens)"]),
    ("Training From Scratch: Pipeline", ["• Stage 0: dataset staging on Palmetto (Globus, Lustre)", "• Stage 1: tokenizer + modality preprocessors", "• Stage 2: distributed trainer (DeepSpeed, FSDP)", "• Stage 3: checkpointing, evaluation, monitoring"]),
    ("Palmetto Cluster Setup", ["• GPU partitions (A100, V100 nodes)", "• SLURM job submission templates", "• Module system for CUDA/cuDNN", "• Shared filesystems and scratch usage policies"]),
    ("Resource Planning", ["• Estimate GPU hours via scaling laws", "• Memory profiling for modality adapters", "• Network bandwidth for data streaming", "• Fallback strategies when jobs preempted"]),
    ("Optimization Techniques", ["• Mixed precision (bf16) for stability", "• Gradient checkpointing for long contexts", "• ZeRO and activation offloading", "• Large-batch optimization with AdamW and Lion"]),
    ("Batching Across Modalities", ["• Padding strategies for variable sequence lengths", "• Bucketed batching for video frame counts", "• Curriculum mixing within dataloaders", "• Asynchronous dataloading on shared storage"]),
    ("Evaluation: Retrieval Tasks", ["• Zero-shot image-text retrieval", "• Recall@K, median rank, CLIP score", "• In-batch negatives vs. memory banks", "• Cross-dataset generalization (COCO, Flickr30k)"]),
    ("Evaluation: Generation Tasks", ["• Captioning metrics (CIDEr, SPICE, BLEU)", "• VQA accuracy and programmatic evaluation", "• Audio transcription WER/CER", "• Human preference studies for dialogue"]),
    ("Responsible Evaluation", ["• Robustness to perturbations", "• Bias and stereotype audits", "• Adversarial prompt testing", "• Safety classifications and refusal rates"]),
    ("Fine-Tuning Goals", ["• Domain adaptation vs. instruction alignment", "• Parameter efficiency vs. accuracy", "• Latency constraints in deployment", "• Interleaving modalities in prompts"]),
    ("Instruction Tuning", ["• Collect multimodal instruction datasets (LLaVA-Instruct)", "• Supervised fine-tuning with conversation format", "• Ensuring coverage of tool-invocation commands", "• Evaluating with multimodal benchmarks (MMBench, MMMU)"]),
    ("Parameter-Efficient Techniques", ["• LoRA/QLoRA on projector and attention layers", "• Adapters for modality encoders", "• Prompt tuning with pseudo tokens", "• BitFit for bias-only updates"]),
    ("Modality-Specific Adapters", ["• Vision adapters (Resampler tokens, Perceiver)", "• Audio adapters with convolutional front-ends", "• Temporal adapters for video windows", "• Routing policies conditioned on modality tokens"]),
    ("Reinforcement & Preference Optimization", ["• RLHF with multimodal preference data", "• Direct Preference Optimization (DPO)", "• Contrastive preference learning", "• Stabilizing training with reward models"]),
    ("Continual & Lifelong Learning", ["• Streaming data updates on Palmetto", "• Replay buffers across modalities", "• Regularization (EWC, LwF)", "• Monitoring catastrophic forgetting"]),
    ("Case Study: CLIP", ["• Dual-encoder architecture", "• Large-scale web image-text pairs", "• Contrastive pretraining workflow", "• Zero-shot transfer and limitations"]),
    ("Case Study: Flamingo", ["• Frozen ViT + frozen LLM", "• Perceiver resampler bridging modalities", "• Few-shot in-context learning with multimodal tokens", "• Scaling to video inputs"]),
    ("Case Study: LLaVA", ["• Visual instruction tuning pipeline", "• Projector module for vision to language", "• Data mixture (synthetic + human)", "• Open-sourced weights for experimentation"]),
    ("Case Study: Audio-Text Models", ["• Whisper encoder + GPT-style decoder", "• Speech-to-text and speech translation", "• Audio-language instruction datasets", "• Integration with multimodal assistants"]),
    ("Applications Across Domains", ["• Medical imaging report generation", "• AR/VR scene understanding", "• Robotic perception-action loops", "• Educational tutoring with diagrams"]),
    ("Research Frontiers", ["• Unified representations for 3D + language", "• World models combining simulation and narration", "• Continual multimodal RL", "• Evaluation of reasoning over charts and math"]),
    ("Lab Roadmap", ["• Session 1: reproduce CLIP inference", "• Session 2: fine-tune LLaVA projector", "• Session 3: extend to audio-text alignment", "• Capstone: any-to-any mini assistant"]),
    ("Palmetto Workflow", ["• VS Code + SSH for development", "• Conda environments per modality", "• Slurm array jobs for hyperparameter sweeps", "• Weave in MLflow logging"]),
    ("Data Management", ["• Using Palmetto's /scratch for datasets", "• Versioning with DVC or LakeFS", "• Streaming from object storage", "• Cleaning scripts and audit trails"]),
    ("Monitoring & Debugging", ["• TensorBoard/Weights & Biases on head node", "• GPU utilization diagnostics (nsys, nvprof)", "• Alerting with Slurm email notifications", "• Checkpoint validation and rollback"]),
    ("Ethical Considerations", ["• Content filtering during pretraining", "• Bias amplification across modalities", "• Accessibility trade-offs", "• Human-in-the-loop review"]),
    ("Accessibility & Inclusion", ["• Designing models for assistive tech", "• Handling low-resource languages and modalities", "• Caption quality for visually impaired users", "• Audio descriptions and tactile outputs"]),
    ("Security & Privacy", ["• PII leakage across modalities", "• Watermarking and provenance", "• Model inversion risks", "• Secure data enclaves on Palmetto"]),
    ("Failure Modes", ["• Hallucinated visual details", "• Audio misalignment under noise", "• Temporal drift in video reasoning", "• Mitigation via confidence estimation"]),
    ("Discussion Prompts", ["• Where do multimodal assistants add the most value?", "• Trade-offs between frozen vs. trainable encoders", "• How to evaluate reasoning vs. perception?", "• What datasets should Clemson curate next?"]),
    ("Key Takeaways", ["• Multimodal LLMs unify perception and language", "• Architectural choices hinge on modality interplay", "• Training from scratch demands careful data + compute planning", "• Fine-tuning unlocks domain-specific assistants"]),
    ("Further Resources", ["• Papers: Flamingo, LLaVA, Kosmos-2, AudioLM", "• Libraries: Hugging Face Transformers, OpenCLIP, LAVIS", "• Tools: DeepSpeed-MII, Megatron-LM", "• Communities: LAION, MLCommons Multimodal"]),
    ("Next Steps", ["• Review lecture notes and suggested readings", "• Set up Palmetto environment before lab", "• Complete homework fine-tuning tasks", "• Prepare questions for lab Q&A"]),
]

assert len(slides) >= 60

SLIDE_WIDTH = 9_144_000
SLIDE_HEIGHT = 6_858_000


def _build_core_created(timestamp: datetime | None) -> str:
    moment = timestamp or datetime.now(UTC).replace(microsecond=0)
    return moment.isoformat().replace("+00:00", "Z")


def _make_paragraph(text: str) -> str:
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return dedent(
        f"""
              <a:p>
                <a:r>
                  <a:rPr lang=\"en-US\" smtClean=\"0\"/>
                  <a:t>{safe}</a:t>
                </a:r>
                <a:endParaRPr lang=\"en-US\"/>
              </a:p>
        """
    ).rstrip()


def _make_slide_xml(title: str, bullets: list[str]) -> str:
    title_safe = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body_xml = "\n".join(_make_paragraph(b) for b in bullets)
    return dedent(
        f"""
            <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
            <p:sld xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">
              <p:cSld>
                <p:spTree>
                  <p:nvGrpSpPr>
                    <p:cNvPr id=\"1\" name=\"\"/>
                    <p:cNvGrpSpPr/>
                    <p:nvPr/>
                  </p:nvGrpSpPr>
                  <p:grpSpPr>
                    <a:xfrm>
                      <a:off x=\"0\" y=\"0\"/>
                      <a:ext cx=\"0\" cy=\"0\"/>
                      <a:chOff x=\"0\" y=\"0\"/>
                      <a:chExt cx=\"0\" cy=\"0\"/>
                    </a:xfrm>
                  </p:grpSpPr>
                  <p:sp>
                    <p:nvSpPr>
                      <p:cNvPr id=\"2\" name=\"Title 1\"/>
                      <p:cNvSpPr/>
                      <p:nvPr>
                        <p:ph type=\"title\"/>
                      </p:nvPr>
                    </p:nvSpPr>
                    <p:spPr>
                      <a:xfrm>
                        <a:off x=\"685800\" y=\"457200\"/>
                        <a:ext cx=\"7772400\" cy=\"1143000\"/>
                      </a:xfrm>
                    </p:spPr>
                    <p:txBody>
                      <a:bodyPr/>
                      <a:lstStyle/>
                      <a:p>
                        <a:r>
                          <a:rPr lang=\"en-US\" sz=\"4400\" b=\"1\"/>
                          <a:t>{title_safe}</a:t>
                        </a:r>
                        <a:endParaRPr lang=\"en-US\" sz=\"4400\"/>
                      </a:p>
                    </p:txBody>
                  </p:sp>
                  <p:sp>
                    <p:nvSpPr>
                      <p:cNvPr id=\"3\" name=\"Content Placeholder 2\"/>
                      <p:cNvSpPr/>
                      <p:nvPr>
                        <p:ph type=\"body\" idx=\"1\"/>
                      </p:nvPr>
                    </p:nvSpPr>
                    <p:spPr>
                      <a:xfrm>
                        <a:off x=\"685800\" y=\"1714500\"/>
                        <a:ext cx=\"7772400\" cy=\"4114800\"/>
                      </a:xfrm>
                    </p:spPr>
                    <p:txBody>
                      <a:bodyPr anchor=\"t\"/>
                      <a:lstStyle/>
    {body_xml}
                    </p:txBody>
                  </p:sp>
                </p:spTree>
              </p:cSld>
              <p:clrMapOvr>
                <a:masterClrMapping/>
              </p:clrMapOvr>
            </p:sld>
        """
    ).strip()


def _make_slide_rel_xml() -> str:
    return dedent(
        """
            <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
            <Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
              <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout\" Target=\"../slideLayouts/slideLayout1.xml\"/>
            </Relationships>
        """
    ).strip()


def build_pptx(output_path: Path | str, *, timestamp: datetime | None = None) -> None:
    """Construct the presentation and write it to ``output_path``."""

    output_path = Path(output_path)
    core_created = _build_core_created(timestamp)

    content_types_xml = dedent(
        """
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">
          <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>
          <Default Extension=\"xml\" ContentType=\"application/xml\"/>
          <Override PartName=\"/ppt/presentation.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml\"/>
          <Override PartName=\"/ppt/slideMasters/slideMaster1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml\"/>
          <Override PartName=\"/ppt/slideLayouts/slideLayout1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml\"/>
          <Override PartName=\"/ppt/theme/theme1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.theme+xml\"/>
          <Override PartName=\"/ppt/tableStyles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml\"/>
          <Override PartName=\"/ppt/presProps.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presProps+xml\"/>
          <Override PartName=\"/ppt/viewProps.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml\"/>
          <Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>
          <Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>
        """
    )
    for idx in range(len(slides)):
        content_types_xml += (
            f"      <Override PartName=\"/ppt/slides/slide{idx + 1}.xml\" "
            "ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slide+xml\"/>\n"
        )
    content_types_xml += "</Types>"

    rels_xml = dedent(
        """
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
          <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"ppt/presentation.xml\"/>
          <Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>
          <Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" Target=\"docProps/app.xml\"/>
        </Relationships>
        """
    )

    core_xml = dedent(
        f"""
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">
          <dc:title>Multimodal Large Language Models</dc:title>
          <dc:subject>Applied Data Science</dc:subject>
          <dc:creator>Course Staff</dc:creator>
          <cp:lastModifiedBy>Automated Generator</cp:lastModifiedBy>
          <dcterms:created xsi:type=\"dcterms:W3CDTF\">{core_created}</dcterms:created>
          <dcterms:modified xsi:type=\"dcterms:W3CDTF\">{core_created}</dcterms:modified>
        </cp:coreProperties>
        """
    )

    app_xml = dedent(
        f"""
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">
          <Application>PowerPoint</Application>
          <Slides>{len(slides)}</Slides>
          <Notes>0</Notes>
          <HiddenSlides>0</HiddenSlides>
          <MMClips>0</MMClips>
        </Properties>
        """
    )

    pres_rels_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n"
        "  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster\" Target=\"slideMasters/slideMaster1.xml\"/>\n"
    )
    for i in range(len(slides)):
        pres_rels_xml += (
            f"  <Relationship Id=\"rId{i + 2}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide\" Target=\"slides/slide{i + 1}.xml\"/>\n"
        )
    pres_rels_xml += "</Relationships>"

    sldId_entries = "".join(
        f"    <p:sldId id=\"{256 + i}\" r:id=\"rId{i + 2}\"/>\n" for i in range(len(slides))
    )
    presentation_xml = dedent(
        f"""
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <p:presentation xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">
          <p:sldMasterIdLst>
            <p:sldMasterId r:id=\"rId1\"/>
          </p:sldMasterIdLst>
          <p:sldIdLst>
{sldId_entries}          </p:sldIdLst>
          <p:sldSz cx=\"{SLIDE_WIDTH}\" cy=\"{SLIDE_HEIGHT}\"/>
          <p:notesSz cx=\"6858000\" cy=\"9144000\"/>
        </p:presentation>
        """
    )

    slideMaster_rel_xml = dedent(
        """
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
          <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme\" Target=\"../theme/theme1.xml\"/>
          <Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout\" Target=\"../slideLayouts/slideLayout1.xml\"/>
        </Relationships>
        """
    )

    slideLayout_rel_xml = dedent(
        """
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
          <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster\" Target=\"../slideMasters/slideMaster1.xml\"/>
        </Relationships>
        """
    )

    slideMaster_xml = dedent(
        """
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <p:sldMaster xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">
          <p:cSld name=\"Default Master\">
            <p:bg>
              <p:bgPr>
                <a:solidFill>
                  <a:schemeClr val=\"bg1\"/>
                </a:solidFill>
              </p:bgPr>
            </p:bg>
            <p:spTree>
              <p:nvGrpSpPr>
                <p:cNvPr id=\"1\" name=\"\"/>
                <p:cNvGrpSpPr/>
                <p:nvPr/>
              </p:nvGrpSpPr>
              <p:grpSpPr>
                <a:xfrm>
                  <a:off x=\"0\" y=\"0\"/>
                  <a:ext cx=\"0\" cy=\"0\"/>
                  <a:chOff x=\"0\" y=\"0\"/>
                  <a:chExt cx=\"0\" cy=\"0\"/>
                </a:xfrm>
              </p:grpSpPr>
              <p:sp>
                <p:nvSpPr>
                  <p:cNvPr id=\"2\" name=\"Title Placeholder 1\"/>
                  <p:cNvSpPr>
                    <a:spLocks noGrp=\"1\"/>
                  </p:cNvSpPr>
                  <p:nvPr>
                    <p:ph type=\"title\"/>
                  </p:nvPr>
                </p:nvSpPr>
                <p:spPr>
                  <a:xfrm>
                    <a:off x=\"685800\" y=\"457200\"/>
                    <a:ext cx=\"7772400\" cy=\"1143000\"/>
                  </a:xfrm>
                </p:spPr>
                <p:txBody>
                  <a:bodyPr/>
                  <a:lstStyle/>
                  <a:p/>
                </p:txBody>
              </p:sp>
              <p:sp>
                <p:nvSpPr>
                  <p:cNvPr id=\"3\" name=\"Content Placeholder 2\"/>
                  <p:cNvSpPr>
                    <a:spLocks noGrp=\"1\"/>
                  </p:cNvSpPr>
                  <p:nvPr>
                    <p:ph type=\"body\" idx=\"1\"/>
                  </p:nvPr>
                </p:nvSpPr>
                <p:spPr>
                  <a:xfrm>
                    <a:off x=\"685800\" y=\"1714500\"/>
                    <a:ext cx=\"7772400\" cy=\"4114800\"/>
                  </a:xfrm>
                </p:spPr>
                <p:txBody>
                  <a:bodyPr anchor=\"t\"/>
                  <a:lstStyle/>
                  <a:p/>
                </p:txBody>
              </p:sp>
            </p:spTree>
          </p:cSld>
          <p:sldLayoutIdLst>
            <p:sldLayoutId id=\"1\" r:id=\"rId2\"/>
          </p:sldLayoutIdLst>
          <p:txStyles>
            <p:titleStyle>
              <a:lvl1pPr algn=\"l\">
                <a:defRPr sz=\"4400\" b=\"1\"/>
              </a:lvl1pPr>
            </p:titleStyle>
            <p:bodyStyle>
              <a:lvl1pPr marL=\"457200\" algn=\"l\">
                <a:defRPr sz=\"3200\"/>
              </a:lvl1pPr>
            </p:bodyStyle>
            <p:otherStyle>
              <a:lvl1pPr/>
            </p:otherStyle>
          </p:txStyles>
          <p:clrMap accent1=\"accent1\" accent2=\"accent2\" accent3=\"accent3\" accent4=\"accent4\" accent5=\"accent5\" accent6=\"accent6\" bg1=\"lt1\" bg2=\"lt2\" tx1=\"dk1\" tx2=\"dk2\" hlink=\"hlink\" folHlink=\"folHlink\"/>
        </p:sldMaster>
        """
    )

    slideLayout_xml = dedent(
        """
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <p:sldLayout xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\" type=\"title\" preserve=\"1\">
          <p:cSld name=\"Title and Content\">
            <p:spTree>
              <p:nvGrpSpPr>
                <p:cNvPr id=\"1\" name=\"\"/>
                <p:cNvGrpSpPr/>
                <p:nvPr/>
              </p:nvGrpSpPr>
              <p:grpSpPr>
                <a:xfrm>
                  <a:off x=\"0\" y=\"0\"/>
                  <a:ext cx=\"0\" cy=\"0\"/>
                  <a:chOff x=\"0\" y=\"0\"/>
                  <a:chExt cx=\"0\" cy=\"0\"/>
                </a:xfrm>
              </p:grpSpPr>
              <p:sp>
                <p:nvSpPr>
                  <p:cNvPr id=\"2\" name=\"Title Placeholder 1\"/>
                  <p:cNvSpPr>
                    <a:spLocks noGrp=\"1\"/>
                  </p:cNvSpPr>
                  <p:nvPr>
                    <p:ph type=\"title\"/>
                  </p:nvPr>
                </p:nvSpPr>
                <p:spPr>
                  <a:xfrm>
                    <a:off x=\"685800\" y=\"457200\"/>
                    <a:ext cx=\"7772400\" cy=\"1143000\"/>
                  </a:xfrm>
                </p:spPr>
                <p:txBody>
                  <a:bodyPr/>
                  <a:lstStyle/>
                  <a:p/>
                </p:txBody>
              </p:sp>
              <p:sp>
                <p:nvSpPr>
                  <p:cNvPr id=\"3\" name=\"Content Placeholder 2\"/>
                  <p:cNvSpPr>
                    <a:spLocks noGrp=\"1\"/>
                  </p:cNvSpPr>
                  <p:nvPr>
                    <p:ph type=\"body\" idx=\"1\"/>
                  </p:nvPr>
                </p:nvSpPr>
                <p:spPr>
                  <a:xfrm>
                    <a:off x=\"685800\" y=\"1714500\"/>
                    <a:ext cx=\"7772400\" cy=\"4114800\"/>
                  </a:xfrm>
                </p:spPr>
                <p:txBody>
                  <a:bodyPr anchor=\"t\"/>
                  <a:lstStyle/>
                  <a:p/>
                </p:txBody>
              </p:sp>
            </p:spTree>
          </p:cSld>
          <p:clrMapOvr>
            <a:masterClrMapping/>
          </p:clrMapOvr>
        </p:sldLayout>
        """
    )

    presProps_xml = dedent(
        """
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <p:presPr xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">
          <p:showPr/>
        </p:presPr>
        """
    )

    viewProps_xml = dedent(
        """
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <p:viewPr xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">
          <p:normalViewPr>
            <p:restoredLeft sz=\"25935\"/>
            <p:restoredTop sz=\"15565\"/>
          </p:normalViewPr>
        </p:viewPr>
        """
    )

    tableStyles_xml = dedent(
        """
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <a:tblStyleLst xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" def=\"tableStyleMedium2\"/>
        """
    )

    theme_xml = dedent(
        """
        <?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
        <a:theme xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" name=\"Default Theme\">
          <a:themeElements>
            <a:clrScheme name=\"Office\">
              <a:dk1><a:srgbClr val=\"000000\"/></a:dk1>
              <a:lt1><a:srgbClr val=\"FFFFFF\"/></a:lt1>
              <a:dk2><a:srgbClr val=\"44546A\"/></a:dk2>
              <a:lt2><a:srgbClr val=\"E7E6E6\"/></a:lt2>
              <a:accent1><a:srgbClr val=\"4472C4\"/></a:accent1>
              <a:accent2><a:srgbClr val=\"ED7D31\"/></a:accent2>
              <a:accent3><a:srgbClr val=\"A5A5A5\"/></a:accent3>
              <a:accent4><a:srgbClr val=\"FFC000\"/></a:accent4>
              <a:accent5><a:srgbClr val=\"5B9BD5\"/></a:accent5>
              <a:accent6><a:srgbClr val=\"70AD47\"/></a:accent6>
              <a:hlink><a:srgbClr val=\"0563C1\"/></a:hlink>
              <a:folHlink><a:srgbClr val=\"954F72\"/></a:folHlink>
            </a:clrScheme>
            <a:fontScheme name=\"Office\">
              <a:majorFont>
                <a:latin typeface=\"Calibri Light\"/>
                <a:ea typeface=\"\"/>
                <a:cs typeface=\"\"/>
              </a:majorFont>
              <a:minorFont>
                <a:latin typeface=\"Calibri\"/>
                <a:ea typeface=\"\"/>
                <a:cs typeface=\"\"/>
              </a:minorFont>
            </a:fontScheme>
            <a:fmtScheme name=\"Office\">
              <a:fillStyleLst>
                <a:solidFill><a:schemeClr val=\"accent1\"/></a:solidFill>
                <a:gradFill rotWithShape=\"1\"><a:gsLst><a:gs pos=\"0\"><a:schemeClr val=\"accent1\"/></a:gs><a:gs pos=\"100000\"><a:schemeClr val=\"accent1\"/></a:gs></a:gsLst><a:lin ang=\"5400000\" scaled=\"1\"/></a:gradFill>
                <a:gradFill rotWithShape=\"1\"><a:gsLst><a:gs pos=\"0\"><a:schemeClr val=\"accent1\"/></a:gs><a:gs pos=\"100000\"><a:schemeClr val=\"accent1\"/></a:gs></a:gsLst><a:lin ang=\"5400000\" scaled=\"1\"/></a:gradFill>
              </a:fillStyleLst>
              <a:lnStyleLst>
                <a:ln w=\"9525\"><a:solidFill><a:schemeClr val=\"accent1\"/></a:solidFill></a:ln>
                <a:ln w=\"9525\"><a:solidFill><a:schemeClr val=\"accent1\"/></a:solidFill></a:ln>
                <a:ln w=\"9525\"><a:solidFill><a:schemeClr val=\"accent1\"/></a:solidFill></a:ln>
              </a:lnStyleLst>
              <a:effectStyleLst>
                <a:effectStyle><a:effectLst/></a:effectStyle>
                <a:effectStyle><a:effectLst/></a:effectStyle>
                <a:effectStyle><a:effectLst/></a:effectStyle>
              </a:effectStyleLst>
              <a:bgFillStyleLst>
                <a:solidFill><a:schemeClr val=\"lt1\"/></a:solidFill>
                <a:solidFill><a:schemeClr val=\"lt1\"/></a:solidFill>
                <a:solidFill><a:schemeClr val=\"lt1\"/></a:solidFill>
              </a:bgFillStyleLst>
            </a:fmtScheme>
          </a:themeElements>
        </a:theme>
        """
    )

    with ZipFile(output_path, "w") as z:
        z.writestr("[Content_Types].xml", content_types_xml)
        z.writestr("_rels/.rels", rels_xml)
        z.writestr("docProps/core.xml", core_xml)
        z.writestr("docProps/app.xml", app_xml)
        z.writestr("ppt/_rels/presentation.xml.rels", pres_rels_xml)
        z.writestr("ppt/presentation.xml", presentation_xml)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slideMaster_rel_xml)
        z.writestr("ppt/slideMasters/slideMaster1.xml", slideMaster_xml)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slideLayout_rel_xml)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slideLayout_xml)
        z.writestr("ppt/theme/theme1.xml", theme_xml)
        z.writestr("ppt/presProps.xml", presProps_xml)
        z.writestr("ppt/viewProps.xml", viewProps_xml)
        z.writestr("ppt/tableStyles.xml", tableStyles_xml)
        for idx, (title, bullets) in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{idx}.xml", _make_slide_xml(title, bullets))
            z.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", _make_slide_rel_xml())


def _parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Multimodal_LLMs_Lecture.pptx"),
        help="Destination for the generated PPTX (default: %(default)s)",
    )
    parser.add_argument(
        "--timestamp",
        type=datetime.fromisoformat,
        default=None,
        help="Optional ISO-8601 timestamp for deterministic metadata.",
    )
    return parser


def main() -> None:
    args = _parse_args().parse_args()
    build_pptx(args.output, timestamp=args.timestamp)


if __name__ == "__main__":
    main()
