import re
import traceback

try:
    import torch
    from sentence_transformers import SentenceTransformer, util
    from transformers import AutoTokenizer, AutoModel, pipeline

    print("Loading SentenceBERT (all-MiniLM-L6-v2) for Text mode...")
    text_evaluator = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading CodeBERT (microsoft/codebert-base) for Code mode...")
    code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    code_model = AutoModel.from_pretrained("microsoft/codebert-base")

    print("Loading NLI Evaluator (facebook/bart-large-mnli)...")
    nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    MODELS_LOADED = True
except Exception as e:
    print(f"CRITICAL ERROR loading HuggingFace models: {e}")
    MODELS_LOADED = False


def get_codebert_embedding(text):
    inputs = code_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = code_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token embedding


def evaluate_content(text: str, context: str, prompt: str, mode: str = "text", domain_weight: float = 1.0):
    if not text or not context:
        return {
            "accuracy_score": 0, "accuracy_percentage": 0,
            "evaluation_score": "Missing Input", "metrics": {}
        }

    try:
        def normalize(x):
            return max(0.0, min(1.0, float(x)))

        # ══════════════════════════════════════════════════════════════════
        #  CODE MODE — 5-Metric Python-Only Evaluation
        # ══════════════════════════════════════════════════════════════════
        if mode == "code":

            # 1. Source Trust Score (30%) — from domain_weight (Tier 1=1.0, Tier 2=0.8, Tier 3=0.5)
            source_trust = normalize(domain_weight)

            # 2. Functional Accuracy (25%) — Python compile() syntax test
            try:
                # Strip Markdown fences (e.g. ```python ... ```) if they exist
                raw_code = text
                if "```" in text:
                    match = re.search(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
                    if match:
                        raw_code = match.group(1)
                
                compile(raw_code, "<string>", "exec")
                functional_accuracy = 1.0
            except SyntaxError:
                functional_accuracy = 0.0

            # 3. Semantic Similarity (20%) — CodeBERT cosine similarity
            try:
                if not MODELS_LOADED: raise Exception("Model missing")
                emb_ctx  = get_codebert_embedding(context[:1000])
                emb_text = get_codebert_embedding(text[:1000])
                sem_sim  = normalize(torch.nn.functional.cosine_similarity(emb_ctx, emb_text).item())
            except:
                sem_sim = 0.5

            # 4. Coverage Score (15%) — matched logic token overlap
            ref_tokens = set(re.findall(r'\b\w+\b', context.lower()))
            gen_tokens = set(re.findall(r'\b\w+\b', text.lower()))
            coverage   = normalize(len(ref_tokens & gen_tokens) / len(ref_tokens)) if ref_tokens else 0.5

            # 5. Structural Similarity (10%) — BLEU-style unigram + bigram overlap
            ref_words = context.lower().split()
            gen_words = text.lower().split()
            unigram_match = len(set(ref_words) & set(gen_words))
            unigram_total = max(len(set(ref_words)), 1)
            bigrams_ref   = set(zip(ref_words, ref_words[1:]))
            bigrams_gen   = set(zip(gen_words, gen_words[1:]))
            bigram_match  = len(bigrams_ref & bigrams_gen)
            bigram_total  = max(len(bigrams_ref), 1)
            structural_sim = normalize(0.5 * unigram_match / unigram_total + 0.5 * bigram_match / bigram_total)

            # Final formula: 0.30×STS + 0.25×FA + 0.20×SS + 0.15×COV + 0.10×STRUCT
            base_score = (
                0.30 * source_trust +
                0.25 * functional_accuracy +
                0.20 * sem_sim +
                0.15 * coverage +
                0.10 * structural_sim
            )

            # Hard penalty: Python syntax failure
            if functional_accuracy == 0.0:
                base_score *= 0.5

            final_score = normalize(base_score)

            metrics_out = {
                "source_trust":           round(source_trust, 4),
                "functional_accuracy":    round(functional_accuracy, 4),
                "semantic_similarity":    round(sem_sim, 4),
                "coverage_score":         round(coverage, 4),
                "structural_similarity":  round(structural_sim, 4),
            }

        # ══════════════════════════════════════════════════════════════════
        #  TEXT MODE — Original 6-Metric Evaluation (unchanged)
        # ══════════════════════════════════════════════════════════════════
        else:
            try:
                if not MODELS_LOADED: raise Exception("Model offline")
                emb_context = text_evaluator.encode(context, convert_to_tensor=True)
                emb_text    = text_evaluator.encode(text, convert_to_tensor=True)
                sim_score   = normalize(util.cos_sim(emb_context, emb_text)[0][0].item())
            except:
                sim_score = 0.5

            try:
                result    = nli_model(text[:512], candidate_labels=["entailment", "neutral", "contradiction"],
                                      hypothesis_template="This text is {} with the source.")
                label     = result['labels'][0]
                nli_score = 1.0 if label == "entailment" else 0.5 if label == "neutral" else 0.2
            except:
                nli_score = 0.5
            nli_score = normalize(nli_score)

            source_words  = set(context.lower().split())
            summary_words = set(text.lower().split())
            overlap       = len(source_words & summary_words)
            tech_score    = normalize(overlap / len(source_words)) if source_words else 0.8

            source_len  = len(context.split())
            summary_len = len(text.split())
            ratio       = summary_len / source_len if source_len > 0 else 1.0
            comp_score  = 1.0 if 0.2 <= ratio <= 0.5 else 0.8 if 0.1 <= ratio <= 0.7 else 0.5
            comp_score  = normalize(comp_score)

            try:
                emb_prompt = text_evaluator.encode(prompt, convert_to_tensor=True)
                rel_score  = normalize(util.cos_sim(emb_prompt, emb_text)[0][0].item())
            except:
                rel_score = 0.8

            sentences  = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
            coh_scores = []
            if len(sentences) > 1:
                for s1, s2 in list(zip(sentences[:-1], sentences[1:]))[:3]:
                    try:
                        res = nli_model(s2, candidate_labels=["entailment", "neutral", "contradiction"],
                                        hypothesis_template="This text is {} with the source.")
                        lab = res['labels'][0]
                        coh_scores.append(1.0 if lab == "entailment" else 0.5 if lab == "neutral" else 0.2)
                    except:
                        pass
            coh_score = normalize(sum(coh_scores) / len(coh_scores) if coh_scores else 1.0)

            base_score = (
                0.30 * sim_score  +
                0.25 * nli_score  +
                0.15 * rel_score  +
                0.10 * coh_score  +
                0.10 * tech_score +
                0.10 * comp_score
            )
            # 60/40 Source Trust bias for text
            final_score = normalize((base_score * 0.4) + (domain_weight * 0.6))
            if nli_score  < 0.5: final_score *= 0.7
            if tech_score < 0.2: final_score *= 0.8
            final_score = normalize(final_score)

            metrics_out = {
                "similarity":  round(sim_score,  4),
                "nli":         round(nli_score,   4),
                "relevance":   round(rel_score,   4),
                "coherence":   round(coh_score,   4),
                "technical":   round(tech_score,  4),
                "compression": round(comp_score,  4)
            }

        # ── Shared evaluation label ────────────────────────────────────────
        if final_score >= 0.85:
            eval_score = "Highly Accurate"
        elif final_score >= 0.70:
            eval_score = "Moderately Accurate"
        elif final_score >= 0.55:
            eval_score = "Partially Accurate"
        else:
            eval_score = "Low Accuracy"

        return {
            "accuracy_score":      round(final_score, 2),
            "accuracy_percentage": round(final_score * 100, 2),
            "evaluation_score":    eval_score,
            "metrics":             metrics_out,
            "domain_weight":       domain_weight
        }

    except Exception as e:
        return {
            "accuracy_score": 0,
            "accuracy_percentage": 0,
            "evaluation_score": f"Error: {e}",
            "metrics": {},
            "domain_weight": domain_weight
        }

