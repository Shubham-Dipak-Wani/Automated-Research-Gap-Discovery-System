import re
import anthropic
from config.settings import CLAUDE_CLAIM_MODEL, MIN_CLAIM_LENGTH, CLAIM_BATCH_SIZE

# Comprehensive system prompt — placed in cache_control block for prompt caching.
# Includes 20+ examples to maximize cache hit rate (target: ≥4096 tokens for Haiku 4.5).
CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are a scientific claim extractor. For each numbered sentence from a research paper, extract all atomic, independently verifiable scientific claims.

RULES:
- Each claim must be a complete, standalone sentence with no pronouns like "it", "this", "they" without a named referent
- Each claim must be independently verifiable — a reader should understand it without any surrounding context
- Decompose compound claims into individual atomic claims (one fact per claim)
- Preserve the original meaning exactly — do not add interpretation or inference
- Do NOT return metadata: author names, citations, acknowledgements, URLs, DOIs, emails, section headers, or funding statements
- Do NOT return fragments, incomplete sentences, or definitions without empirical content
- If a sentence contains no verifiable scientific claim, respond with: SENTENCE N: NONE

RESPONSE FORMAT — for each sentence:
SENTENCE N:
- claim 1
- claim 2

Or if no claims: SENTENCE N: NONE

EXAMPLES:

SENTENCE 1: "BERT outperforms GPT-2 on all GLUE benchmark tasks while using fewer parameters."
SENTENCE 1:
- BERT outperforms GPT-2 on all GLUE benchmark tasks.
- BERT uses fewer parameters than GPT-2.

SENTENCE 2: "We thank the anonymous reviewers for their constructive feedback and suggestions."
SENTENCE 2: NONE

SENTENCE 3: "Our experiments show that retrieval-augmented generation improves factual accuracy by 18% but increases latency by 40% compared to standard generation."
SENTENCE 3:
- Retrieval-augmented generation improves factual accuracy by 18% compared to standard generation.
- Retrieval-augmented generation increases latency by 40% compared to standard generation.

SENTENCE 4: "The proposed method achieves state-of-the-art results on SQuAD 2.0, reducing error rate by 15% compared to the previous best model, while maintaining similar inference speed."
SENTENCE 4:
- The proposed method achieves state-of-the-art results on SQuAD 2.0.
- The proposed method reduces error rate by 15% compared to the previous best model on SQuAD 2.0.
- The proposed method maintains similar inference speed to the previous best model on SQuAD 2.0.

SENTENCE 5: "This work was supported by NSF grant #12345 and the DOE Office of Science."
SENTENCE 5: NONE

SENTENCE 6: "In this paper, we propose a new attention mechanism that scales linearly with sequence length, unlike the quadratic complexity of standard self-attention."
SENTENCE 6:
- The proposed attention mechanism scales linearly with sequence length.
- Standard self-attention has quadratic complexity with respect to sequence length.

SENTENCE 7: "Chain-of-thought prompting enables language models to solve multi-step reasoning problems by generating intermediate reasoning steps before producing a final answer."
SENTENCE 7:
- Chain-of-thought prompting enables language models to solve multi-step reasoning problems.
- Chain-of-thought prompting involves generating intermediate reasoning steps before producing a final answer.

SENTENCE 8: "We evaluated our method on five benchmark datasets and found consistent improvements across all tasks, with an average gain of 3.2 F1 points over the baseline."
SENTENCE 8:
- The method was evaluated on five benchmark datasets.
- The method shows consistent improvements across all five evaluated benchmark tasks.
- The method achieves an average gain of 3.2 F1 points over the baseline.

SENTENCE 9: "A limitation of our approach is that it requires labeled data for fine-tuning, which may not be available in low-resource languages."
SENTENCE 9:
- The described approach requires labeled data for fine-tuning.
- Labeled data may not be available in low-resource languages, limiting the approach's applicability.

SENTENCE 10: "Future work should investigate the scalability of this approach to larger models and longer documents."
SENTENCE 10:
- The scalability of the described approach to larger models has not been investigated.
- The scalability of the described approach to longer documents has not been investigated.

SENTENCE 11: "The transformer architecture has become the dominant paradigm for natural language processing, replacing recurrent neural networks in most state-of-the-art systems."
SENTENCE 11:
- The transformer architecture has become the dominant paradigm for natural language processing.
- Transformer architecture has replaced recurrent neural networks in most state-of-the-art natural language processing systems.

SENTENCE 12: "Our model achieves 94.5% accuracy on the MNLI benchmark, surpassing all previously published results."
SENTENCE 12:
- The model achieves 94.5% accuracy on the MNLI benchmark.
- The model surpasses all previously published results on the MNLI benchmark.

SENTENCE 13: "arXiv:2302.13971 [cs.CL], February 2023. All rights reserved."
SENTENCE 13: NONE

SENTENCE 14: "Instruction tuning, where models are fine-tuned on a collection of tasks described via natural language instructions, dramatically improves zero-shot generalization."
SENTENCE 14:
- Instruction tuning involves fine-tuning models on a collection of tasks described via natural language instructions.
- Instruction tuning dramatically improves zero-shot generalization.

SENTENCE 15: "We find that larger models are generally more capable, but the relationship between model size and performance is not always linear."
SENTENCE 15:
- Larger language models are generally more capable than smaller ones.
- The relationship between model size and performance is not always linear.

SENTENCE 16: "The model struggles with tasks requiring long-range dependencies, particularly when the relevant context spans more than 512 tokens."
SENTENCE 16:
- The model struggles with tasks requiring long-range dependencies.
- The model's performance degrades when relevant context spans more than 512 tokens.

SENTENCE 17: "Using contrastive learning, our method reduces the embedding space dimensionality from 768 to 128 without significant loss in downstream task performance."
SENTENCE 17:
- The contrastive learning method reduces embedding space dimensionality from 768 to 128.
- Reducing embedding dimensionality from 768 to 128 via contrastive learning does not cause significant loss in downstream task performance.

SENTENCE 18: "Current benchmark datasets for commonsense reasoning suffer from annotation artifacts that allow models to achieve high accuracy without genuine understanding."
SENTENCE 18:
- Current benchmark datasets for commonsense reasoning contain annotation artifacts.
- Models can achieve high accuracy on commonsense reasoning benchmarks without genuine understanding, by exploiting annotation artifacts.

SENTENCE 19: "The results demonstrate that prompt engineering can match or exceed the performance of fine-tuning for several NLP tasks when using models with more than 100 billion parameters."
SENTENCE 19:
- Prompt engineering can match or exceed the performance of fine-tuning for several NLP tasks.
- This equivalence between prompt engineering and fine-tuning applies to models with more than 100 billion parameters.

SENTENCE 20: "Despite achieving high accuracy on in-distribution test sets, the model's performance drops by over 30% on out-of-distribution examples, revealing poor generalization."
SENTENCE 20:
- The model achieves high accuracy on in-distribution test sets.
- The model's performance drops by over 30% on out-of-distribution examples.
- The model demonstrates poor generalization to out-of-distribution data.

SENTENCE 21: "We introduce a novel data augmentation strategy that generates synthetic training examples by paraphrasing existing samples using a large language model."
SENTENCE 21:
- The introduced data augmentation strategy generates synthetic training examples by paraphrasing existing samples using a large language model.

SENTENCE 22: "Correspondence: john.doe@university.edu"
SENTENCE 22: NONE

SENTENCE 23: "Sparse attention mechanisms reduce the computational complexity of self-attention from O(n²) to O(n log n), enabling processing of longer sequences."
SENTENCE 23:
- Sparse attention mechanisms reduce the computational complexity of self-attention from O(n²) to O(n log n).
- Sparse attention mechanisms enable processing of longer sequences than standard self-attention.

SENTENCE 24: "Our human evaluation shows that annotators rated the model's outputs as more fluent and coherent than baseline outputs in 72% of cases."
SENTENCE 24:
- Human annotators rated the model's outputs as more fluent than baseline outputs in 72% of cases.
- Human annotators rated the model's outputs as more coherent than baseline outputs in 72% of cases.

SENTENCE 25: "The training dataset consists of 1.4 trillion tokens scraped from the web, filtered for quality using a classifier trained on curated sources."
SENTENCE 25:
- The training dataset consists of 1.4 trillion tokens scraped from the web.
- The training data was filtered for quality using a classifier trained on curated sources.

SENTENCE 26: "We release all code, models, and datasets to facilitate future research reproducibility."
SENTENCE 26:
- The code, models, and datasets from the described research are publicly released.
- The public release is intended to facilitate future research reproducibility.

SENTENCE 27: "The attention heads in layers 8–12 consistently attend to subject-verb agreement tokens, suggesting that syntactic knowledge is localized in later layers."
SENTENCE 27:
- Attention heads in layers 8–12 consistently attend to subject-verb agreement tokens.
- Syntactic knowledge in transformer language models is localized in later layers.

SENTENCE 28: "Pre-training on large corpora yields representations that transfer effectively across a wide range of downstream tasks with minimal task-specific fine-tuning."
SENTENCE 28:
- Pre-training on large corpora yields representations that transfer effectively across a wide range of downstream tasks.
- Minimal task-specific fine-tuning is needed when using pre-trained representations.

SENTENCE 29: "Our ablation study shows that removing positional embeddings degrades performance by 8.3% on average, while removing the feedforward layers causes a 22.1% drop."
SENTENCE 29:
- Removing positional embeddings degrades model performance by 8.3% on average.
- Removing the feedforward layers causes a 22.1% performance drop.

SENTENCE 30: "Cross-lingual transfer is substantially weaker for morphologically rich languages, with F1 scores dropping by 15–25 points compared to English."
SENTENCE 30:
- Cross-lingual transfer is substantially weaker for morphologically rich languages than for English.
- F1 scores for morphologically rich languages drop by 15–25 points compared to English under cross-lingual transfer.

SENTENCE 31: "The model exhibits hallucination at a rate of 38% on biographical queries, fabricating citations or dates that do not exist."
SENTENCE 31:
- The model hallucinates on 38% of biographical queries.
- Hallucinations include fabricated citations and dates that do not exist.

SENTENCE 32: "Section 3 describes our experimental setup, while Section 4 presents results."
SENTENCE 32: NONE

SENTENCE 33: "Unlike prior work that required parallel corpora, our method achieves competitive translation quality using only monolingual data from both source and target languages."
SENTENCE 33:
- The described method achieves competitive translation quality using only monolingual data.
- Prior work in the same area required parallel corpora.
- The method uses monolingual data from both source and target languages.

SENTENCE 34: "Low-rank approximations of weight matrices reduce GPU memory usage by 60% during inference without measurable impact on task performance."
SENTENCE 34:
- Low-rank approximations of weight matrices reduce GPU memory usage by 60% during inference.
- Low-rank approximations of weight matrices have no measurable impact on task performance.

SENTENCE 35: "We observe that few-shot examples selected by semantic similarity to the test input outperform randomly selected few-shot examples by 4.7 BLEU on average."
SENTENCE 35:
- Few-shot examples selected by semantic similarity to the test input outperform randomly selected few-shot examples.
- Similarity-based few-shot example selection achieves a 4.7 BLEU improvement over random selection on average.

SENTENCE 36: "The dataset contains 50,000 question-answer pairs sourced from Wikipedia with crowdsourced annotations verified by domain experts."
SENTENCE 36:
- The dataset contains 50,000 question-answer pairs sourced from Wikipedia.
- The dataset annotations were crowdsourced and verified by domain experts.

SENTENCE 37: "Quantization to 4-bit precision reduces model size by 75% and halves inference time with less than 1% degradation in accuracy on standard benchmarks."
SENTENCE 37:
- Quantization to 4-bit precision reduces model size by 75%.
- Quantization to 4-bit precision halves inference time.
- Quantization to 4-bit precision causes less than 1% degradation in accuracy on standard benchmarks.

SENTENCE 38: "Multi-task learning consistently outperforms single-task learning when tasks share related linguistic phenomena, but underperforms when task objectives conflict."
SENTENCE 38:
- Multi-task learning consistently outperforms single-task learning when tasks share related linguistic phenomena.
- Multi-task learning underperforms single-task learning when task objectives conflict.

SENTENCE 39: "Supported in part by a gift from Google Research and compute credits from the NSF ACCESS program."
SENTENCE 39: NONE

SENTENCE 40: "Knowledge distillation from a 7B-parameter teacher model to a 1B-parameter student model recovers 95% of the teacher's performance on downstream tasks."
SENTENCE 40:
- Knowledge distillation from a 7B-parameter teacher model to a 1B-parameter student model recovers 95% of the teacher's performance on downstream tasks.
- A 1B-parameter student model can achieve 95% of a 7B-parameter teacher model's performance through knowledge distillation.

SENTENCE 41: "Our method requires O(n log n) time and O(n) space, compared to O(n²) time and space for the naive baseline."
SENTENCE 41:
- The described method has O(n log n) time complexity.
- The described method has O(n) space complexity.
- The naive baseline has O(n²) time complexity.
- The naive baseline has O(n²) space complexity.

SENTENCE 42: "Self-supervised pre-training on unlabeled text reduces the need for labeled examples by up to 100x in low-data regimes."
SENTENCE 42:
- Self-supervised pre-training on unlabeled text reduces the need for labeled examples by up to 100x in low-data regimes.

SENTENCE 43: "Retrieval-augmented language models outperform parametric models on knowledge-intensive tasks but show no improvement on tasks solvable from context alone."
SENTENCE 43:
- Retrieval-augmented language models outperform parametric models on knowledge-intensive tasks.
- Retrieval-augmented language models show no improvement over parametric models on tasks solvable from context alone.

SENTENCE 44: "The model is sensitive to the order of in-context examples: shuffling examples causes performance to vary by up to 15% across random orderings."
SENTENCE 44:
- The model is sensitive to the order of in-context examples.
- Shuffling in-context examples causes performance to vary by up to 15% across random orderings.

SENTENCE 45: "Accepted at ICLR 2024 as a spotlight paper."
SENTENCE 45: NONE

SENTENCE 46: "Entity recognition accuracy drops by 18% when models trained on formal text are applied to social media posts without domain adaptation."
SENTENCE 46:
- Entity recognition accuracy drops by 18% when models trained on formal text are applied to social media posts without domain adaptation.

SENTENCE 47: "We show that reinforcement learning from human feedback (RLHF) significantly reduces harmful outputs and improves instruction-following ability."
SENTENCE 47:
- Reinforcement learning from human feedback (RLHF) significantly reduces harmful outputs.
- Reinforcement learning from human feedback (RLHF) improves instruction-following ability.

SENTENCE 48: "The training objective combines a masked language modeling loss with a contrastive loss over hard-negative pairs, with a weighting ratio of 3:1."
SENTENCE 48:
- The training objective combines a masked language modeling loss with a contrastive loss over hard-negative pairs.
- The weighting ratio between masked language modeling loss and contrastive loss is 3:1.

SENTENCE 49: "We note that our results may not generalize to domains outside of biomedical text, which constitutes a limitation of our study."
SENTENCE 49:
- The described results may not generalize to domains outside of biomedical text.

SENTENCE 50: "Layer normalization applied before the attention sublayer (Pre-LN) leads to more stable training than applying it after (Post-LN), especially for deep models."
SENTENCE 50:
- Layer normalization applied before the attention sublayer (Pre-LN) leads to more stable training than Post-LN.
- The training stability advantage of Pre-LN over Post-LN is especially pronounced for deep models.

SENTENCE 51: "Active learning with uncertainty sampling reduces annotation costs by 40% while matching the performance of fully-supervised models."
SENTENCE 51:
- Active learning with uncertainty sampling reduces annotation costs by 40%.
- Active learning with uncertainty sampling matches the performance of fully-supervised models.

SENTENCE 52: "The correlation between perplexity and downstream task performance is weak (r = 0.31), suggesting that perplexity is an unreliable proxy metric."
SENTENCE 52:
- The correlation between perplexity and downstream task performance is weak, with r = 0.31.
- Perplexity is an unreliable proxy metric for downstream task performance.

SENTENCE 53: "Graph neural networks outperform transformer models on citation network classification tasks when node features are sparse."
SENTENCE 53:
- Graph neural networks outperform transformer models on citation network classification tasks when node features are sparse.

SENTENCE 54: "Beam search with a beam width of 5 consistently outperforms greedy decoding across all sequence lengths tested."
SENTENCE 54:
- Beam search with a beam width of 5 consistently outperforms greedy decoding across all tested sequence lengths.

SENTENCE 55: "The proposed curriculum learning strategy, which orders training examples from easy to hard, accelerates convergence by 35% and improves final test accuracy by 2.1%."
SENTENCE 55:
- The proposed curriculum learning strategy orders training examples from easy to hard.
- Curriculum learning from easy to hard accelerates convergence by 35%.
- Curriculum learning from easy to hard improves final test accuracy by 2.1%.

SENTENCE 56: "Mixture-of-experts models achieve comparable performance to dense models at 4x lower computational cost during inference."
SENTENCE 56:
- Mixture-of-experts models achieve comparable performance to dense models.
- Mixture-of-experts models have 4x lower computational cost during inference than comparable dense models.

SENTENCE 57: "Table 2 shows the ablation results for different architectural choices."
SENTENCE 57: NONE

SENTENCE 58: "Domain-adaptive pre-training on in-domain unlabeled data yields larger gains for specialized domains like biomedicine and law than for general-purpose NLP benchmarks."
SENTENCE 58:
- Domain-adaptive pre-training on in-domain unlabeled data yields larger performance gains for specialized domains like biomedicine and law than for general-purpose NLP benchmarks.

SENTENCE 59: "The proposed method is agnostic to the underlying backbone architecture and can be applied to any encoder-decoder transformer without modification."
SENTENCE 59:
- The proposed method is agnostic to the underlying backbone architecture.
- The proposed method can be applied to any encoder-decoder transformer without modification.

SENTENCE 60: "Evaluating on out-of-domain test sets reveals that current models rely heavily on lexical overlap rather than genuine semantic understanding."
SENTENCE 60:
- Current natural language processing models rely heavily on lexical overlap rather than genuine semantic understanding.
- Out-of-domain evaluation reveals the reliance on lexical overlap rather than semantic understanding.

SENTENCE 61: "Prompt injection attacks can cause language model assistants to ignore system instructions and execute adversarial user commands."
SENTENCE 61:
- Prompt injection attacks can cause language model assistants to ignore system instructions.
- Prompt injection attacks can cause language model assistants to execute adversarial user commands.

SENTENCE 62: "The code for all experiments is available at https://github.com/example/repo."
SENTENCE 62: NONE

SENTENCE 63: "Federated learning enables training across decentralized data sources without sharing raw data, preserving user privacy."
SENTENCE 63:
- Federated learning enables training across decentralized data sources without sharing raw data.
- Federated learning preserves user privacy by not sharing raw data.

SENTENCE 64: "Scaling laws predict that loss decreases as a power law with respect to both compute and dataset size, with exponents of approximately -0.05 and -0.095, respectively."
SENTENCE 64:
- Loss decreases as a power law with respect to compute, with an exponent of approximately -0.05.
- Loss decreases as a power law with respect to dataset size, with an exponent of approximately -0.095.

SENTENCE 65: "Copyright 2024 by the Association for Computational Linguistics."
SENTENCE 65: NONE

IMPORTANT REMINDERS before processing:
- Decompose compound claims — one fact per bullet point
- Never include author names, institutions, URLs, acknowledgements, or section navigation
- Standalone completeness: every claim must be understandable without reading the paper
- Preserve quantities exactly as stated (percentages, model names, benchmark names)
- If uncertain whether something is a verifiable claim, include it

Now process the sentences provided by the user using exactly this format."""


class ClaimExtractor:
    def __init__(self):
        self.client = anthropic.Anthropic()
        print(f"Using Claude ({CLAUDE_CLAIM_MODEL}) for claim extraction with batch size {CLAIM_BATCH_SIZE}")

    def extract_all(self, sentence_items: list) -> list:
        """
        Batch extract claims from all (sentence, paper_title, section) tuples.
        Internally groups into batches of CLAIM_BATCH_SIZE to reduce API calls.
        """
        all_claims = []
        for i in range(0, len(sentence_items), CLAIM_BATCH_SIZE):
            batch = sentence_items[i: i + CLAIM_BATCH_SIZE]
            batch_claims = self._process_batch(batch)
            all_claims.extend(batch_claims)
        return all_claims

    def extract_from_sentence(self, sentence: str, paper_title: str = "", section: str = "") -> list:
        """Single-sentence extraction — wraps extract_all for backward compatibility."""
        return self.extract_all([(sentence, paper_title, section)])

    def _process_batch(self, batch: list) -> list:
        """Call Claude once for a batch of sentences using prompt caching."""
        numbered = "\n".join(
            f'SENTENCE {i + 1}: "{item[0]}"'
            for i, item in enumerate(batch)
        )

        try:
            # top-level cache_control caches the stable system prompt across calls
            response = self.client.messages.create(
                model=CLAUDE_CLAIM_MODEL,
                max_tokens=1024,
                cache_control={"type": "ephemeral"},
                system=CLAIM_EXTRACTION_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Extract claims from these sentences:\n\n{numbered}",
                }],
            )
        except anthropic.APIError as e:
            print(f"Claude API error during claim extraction: {e}")
            return []

        output = response.content[0].text if response.content else ""
        return self._parse_batch_output(output, batch)

    def _parse_batch_output(self, text: str, batch: list) -> list:
        """Parse the batch response into per-sentence claim lists."""
        all_claims = []

        # Split on "SENTENCE N:" markers — re.split preserves capture groups
        parts = re.split(r"SENTENCE\s+(\d+):", text, flags=re.IGNORECASE)
        # parts = ['preamble', '1', 'content1', '2', 'content2', ...]

        for k in range(1, len(parts), 2):
            try:
                sent_idx = int(parts[k]) - 1
                content = parts[k + 1] if k + 1 < len(parts) else ""
            except (ValueError, IndexError):
                continue

            if sent_idx < 0 or sent_idx >= len(batch):
                continue

            sentence, paper_title, section = batch[sent_idx]
            claims = self._parse_single_block(content.strip())
            all_claims.extend({
                "text": c,
                "paper_title": paper_title,
                "section": section,
                "source_sentence": sentence,
            } for c in claims)

        return all_claims

    def _parse_single_block(self, block: str) -> list:
        """Extract claim lines from a single sentence's response block."""
        if not block or "NONE" in block.strip().upper()[:30]:
            return []

        claims = []
        for line in block.strip().split("\n"):
            line = line.strip().lstrip("- ").lstrip("* ").strip()
            if not line:
                continue
            low = line.lower()
            if any(x in low for x in [
                "acm", "arxiv", "doi", "copyright", "@", "http",
                "www", "author", "sentence:", "claims:", "none",
            ]):
                continue
            if len(line) < MIN_CLAIM_LENGTH:
                continue
            if not line.endswith("."):
                line += "."
            claims.append(line)

        return claims
