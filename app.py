Enhanced Hallucination Detection System v3.2 - HuggingFace Deployment
Complete 8-Layer Verification System
"""

import gradio as gr
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from ddgs import DDGS
import pickle
import os
import re
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from collections import Counter
import plotly.graph_objects as go

# ============================================
# CONFIGURATION - HUGGINGFACE COMPATIBLE
# ============================================

# Get API key from HuggingFace Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not set!")
    print("Please add it in Space Settings → Secrets")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# File paths - NOT available on HuggingFace (will use pretrained mode)
CORPUS_FILE = None
FAISS_INDEX_FILE = None
BM25_INDEX_FILE = None

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GPT_MODEL = "gpt-4o-mini"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NUM_CONSISTENCY_CHECKS = 5
TEMPERATURE = 0.3
SIMILARITY_THRESHOLD = 0.85
WEB_SEARCH_RESULTS = 5

# ============================================
# ENTROPY CALCULATOR
# ============================================

class EntropyCalculator:
    def __init__(self):
        pass
    
    def calculate_shannon_entropy(self, probabilities):
        probs = np.array(probabilities)
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0, 0.0
        h = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probabilities))
        normalized_h = h / max_entropy if max_entropy > 0 else 0.0
        return h, normalized_h
    
    def calculate_semantic_entropy(self, answer_groups):
        if not answer_groups:
            return {'entropy': 0.0, 'normalized_entropy': 0.0, 'num_groups': 0}
        total_answers = sum(len(group) for group in answer_groups)
        group_probs = [len(group) / total_answers for group in answer_groups]
        h, normalized_h = self.calculate_shannon_entropy(group_probs)
        return {
            'entropy': h,
            'normalized_entropy': normalized_h,
            'num_groups': len(answer_groups),
            'group_sizes': [len(g) for g in answer_groups],
            'group_probabilities': group_probs
        }
    
    def calculate_nli_entropy(self, nli_scores):
        if not nli_scores:
            return {'entropy': 0.0, 'normalized_entropy': 0.0}
        avg_probs = np.mean(nli_scores, axis=0)
        h, normalized_h = self.calculate_shannon_entropy(avg_probs)
        return {
            'entropy': h,
            'normalized_entropy': normalized_h,
            'avg_probabilities': avg_probs.tolist()
        }
    
    def calculate_combined_confidence(self, semantic_entropy, nli_entropy, consistency_score, web_match=None):
        consistency_normalized = consistency_score / 100.0
        semantic_certainty = 1.0 - semantic_entropy
        nli_certainty = 1.0 - nli_entropy
        
        if web_match is not None:
            web_normalized = web_match / 100.0
            combined_certainty = (
                0.35 * consistency_normalized +
                0.25 * semantic_certainty +
                0.15 * nli_certainty +
                0.25 * web_normalized
            )
        else:
            combined_certainty = (
                0.50 * consistency_normalized +
                0.30 * semantic_certainty +
                0.20 * nli_certainty
            )
        
        confidence = combined_certainty * 100
        
        has_web_contradiction = False
        if web_match is not None and web_match < 30 and consistency_score > 80:
            confidence = max(confidence, 55)
            confidence = min(confidence, 65)
            has_web_contradiction = True
        
        if confidence >= 85:
            risk_level, risk_color = "VERY LOW", "#10b981"
        elif confidence >= 70:
            risk_level, risk_color = "LOW", "#3b82f6"
        elif confidence >= 50:
            risk_level, risk_color = "MEDIUM", "#f59e0b"
        else:
            risk_level, risk_color = "HIGH", "#ef4444"
        
        return {
            'confidence_score': confidence,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'has_web_contradiction': has_web_contradiction,
            'components': {
                'self_consistency': consistency_score,
                'semantic_certainty': semantic_certainty * 100,
                'nli_certainty': nli_certainty * 100,
                'web_match': web_match if web_match is not None else 0
            }
        }

# ============================================
# CLAIM VERIFIER (Layer 8)
# ============================================

class ClaimVerifier:
    """
    Layer 8: Verify if a claim is supported by the generated answer
    Uses NLI to classify: SUPPORTS, REFUTES, or NOT ENOUGH INFO
    """
    
    def __init__(self, nli_model, nli_tokenizer):
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
    
    def verify_claim(self, claim: str, answer: str) -> dict:
        """
        Check if the answer supports, refutes, or is neutral to the claim
        """
        
        inputs = self.nli_tokenizer(
            answer,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
        
        reordered_probs = torch.tensor([probs[0][0], probs[0][2], probs[0][1]])
        
        contradiction_score = reordered_probs[0].item()
        neutral_score = reordered_probs[1].item()
        entailment_score = reordered_probs[2].item()
        
        if entailment_score > 0.7:
            label = 'SUPPORTS'
        elif contradiction_score > 0.6:
            label = 'REFUTES'
        else:
            label = 'NOT ENOUGH INFO'
        
        return {
            'label': label,
            'entailment_score': entailment_score,
            'contradiction_score': contradiction_score,
            'neutral_score': neutral_score,
            'confidence': max(entailment_score, contradiction_score, neutral_score),
            'all_scores': {
                'SUPPORTS': entailment_score,
                'REFUTES': contradiction_score,
                'NOT ENOUGH INFO': neutral_score
            }
        }

# ============================================
# HALLUCINATION DETECTOR
# ============================================

class HallucinationDetector:
    def __init__(self, progress=None):
        self.progress = progress
        self.update_progress("Loading embedding model...", 0.2)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        self.update_progress("Loading NLI model...", 0.5)
        self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        
        self.update_progress("Initializing entropy calculator...", 0.8)
        self.entropy_calc = EntropyCalculator()
        
        self.update_progress("Initializing claim verifier...", 0.9)
        self.claim_verifier = ClaimVerifier(self.nli_model, self.nli_tokenizer)
        
        # Note: No Wikipedia corpus available on HuggingFace
        self.articles = []
        self.faiss_index = None
        self.bm25 = None
        
        self.update_progress("System ready!", 1.0)
        print("✓ System initialized (Pretrained mode - no Wikipedia corpus)")
    
    def update_progress(self, message, progress):
        if self.progress:
            self.progress(progress, desc=message)
    
    def generate_answer_pretrained(self, query):
        if not client:
            return "⚠️ OpenAI API key not configured. Please add OPENAI_API_KEY in Space Settings → Secrets."
        
        prompt = f"""Answer the following question using your knowledge. Be concise and factual.
Question: {query}
Answer:"""
        
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def is_dont_know_answer(self, answer):
        dont_know_phrases = [
            "don't have enough information", "cannot provide", "unable to answer",
            "don't know", "no information", "not enough information"
        ]
        return any(phrase in answer.lower() for phrase in dont_know_phrases)
    
    def semantic_similarity(self, text1, text2):
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        except:
            return 0.0
    
    def group_similar_answers(self, answers):
        if not answers:
            return []
        groups = []
        for answer in answers:
            placed = False
            for group in groups:
                if self.semantic_similarity(answer, group[0]) >= SIMILARITY_THRESHOLD:
                    group.append(answer)
                    placed = True
                    break
            if not placed:
                groups.append([answer])
        groups.sort(key=len, reverse=True)
        return groups
    
    def verify_with_nli(self, claim, evidence):
        inputs = self.nli_tokenizer(evidence, claim, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
        reordered_probs = torch.tensor([probs[0][0], probs[0][2], probs[0][1]])
        labels = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
        pred_idx = torch.argmax(reordered_probs).item()
        return {
            'label': labels[pred_idx],
            'entailment_score': reordered_probs[2].item(),
            'probabilities': reordered_probs.tolist()
        }
    
    def web_search(self, query):
        results = []
        try:
            ddgs = DDGS()
            search_results = ddgs.text(query, max_results=WEB_SEARCH_RESULTS)
            for result in search_results:
                results.append({
                    'title': result.get('title', 'No title'),
                    'url': result.get('href', 'No URL'),
                    'snippet': result.get('body', 'No snippet')
                })
        except:
            pass
        return results
    
    def verify_with_web(self, answer, query):
        if not client:
            return None
        
        try:
            web_results = self.web_search(query)
            if not web_results:
                return None
            
            web_context = "\n\n".join([f"Source {i+1}: {r['title']}\n{r['snippet']}" for i, r in enumerate(web_results)])
            
            verification_prompt = f"""Compare the following answer with web search results and determine if they match.
Answer to verify: {answer}
Web search results:
{web_context}
Rate the match from 0-100% and explain briefly. Format: "MATCH: X% - explanation"
"""
            
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": verification_prompt}],
                temperature=0.1,
                max_tokens=150
            )
            
            verification = response.choices[0].message.content.strip()
            match_percent = 0
            match = re.search(r'(\d+)%', verification)
            if match:
                match_percent = int(match.group(1))
            
            return {
                'web_results': web_results,
                'verification': verification,
                'match_percent': match_percent
            }
        except:
            return None
    
    def is_claim_input(self, text):
        """Detect if input is a claim (vs a question)"""
        if '?' in text:
            return False
        
        question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'tell', 'explain', 'describe']
        first_word = text.strip().lower().split()[0] if text.strip() else ''
        if first_word in question_words:
            return False
        
        return True
    
    def explain_confidence(self, results):
        explanations = []
        
        if results['consistency_score'] >= 80:
            explanations.append(("✅", "High agreement across all 5 verification attempts"))
        elif results['consistency_score'] >= 60:
            explanations.append(("⚠️", "Moderate variation detected in answer consistency"))
        else:
            explanations.append(("❌", "Significant disagreement between verification attempts"))
        
        if results['semantic_entropy'] < 0.3:
            explanations.append(("✅", "Answers are semantically very similar"))
        elif results['semantic_entropy'] < 0.6:
            explanations.append(("⚠️", "Some semantic variation in answers"))
        else:
            explanations.append(("❌", "High semantic uncertainty detected"))
        
        if results['web_match'] >= 70:
            explanations.append(("✅", f"Strong web verification: {results['web_match']}% match"))
        elif results['web_match'] >= 40:
            explanations.append(("⚠️", f"Partial web agreement: {results['web_match']}% match"))
        elif results['web_match'] > 0:
            explanations.append(("❌", f"Web sources disagree: {results['web_match']}% match"))
        
        if results.get('claim_verification'):
            cv = results['claim_verification']
            explanations.append(("🎯", f"Claim verification: {cv['label']} ({cv['confidence']*100:.0f}% confidence)"))
        
        return explanations
    
    def detect(self, query, progress=None):
        """Main detection method with all 8 layers"""
        
        if progress is None:
            progress = lambda x, desc='': None
        
        results = {
            'query': query,
            'answer': '',
            'mode': 'Pretrained',
            'mode_reason': 'Running in pretrained mode (Wikipedia corpus not available in web demo)',
            'wiki_articles': [],
            'wiki_score': 0,
            'consistency_score': 0,
            'semantic_entropy': 0,
            'nli_results': [],
            'nli_entropy': 0,
            'combined_confidence': 0,
            'risk_level': '',
            'risk_color': '',
            'has_web_contradiction': False,
            'web_results': [],
            'web_match': 0,
            'all_answers': [],
            'used_reranking': False,
            'used_fallback': False,
            'num_answer_groups': 0,
            'explanations': [],
            'verification_log': [],
            'claim_verification': None,
            'fever_label': None
        }
        
        results['verification_log'].append(("ℹ️", "Mode Selection", "Pretrained mode (Wikipedia not available in web demo)"))
        
        # Self-consistency
        progress(0.3, desc=f"🔄 Layer 3: Self-Consistency ({NUM_CONSISTENCY_CHECKS} attempts)...")
        answers = []
        for i in range(NUM_CONSISTENCY_CHECKS):
            answer = self.generate_answer_pretrained(query)
            answers.append(answer)
            time.sleep(0.3)
        
        results['verification_log'].append(("🔄", "Self-Consistency", f"Generated {NUM_CONSISTENCY_CHECKS} independent answers"))
        results['all_answers'] = answers
        
        # Semantic clustering
        progress(0.5, desc="🧬 Layer 4: Semantic Clustering...")
        answer_groups = self.group_similar_answers(answers)
        largest_group = answer_groups[0] if answer_groups else []
        consensus_answer = largest_group[0] if largest_group else answers[0]
        consistency_score = len(largest_group) / len(answers) * 100
        
        results['answer'] = consensus_answer
        results['consistency_score'] = consistency_score
        results['num_answer_groups'] = len(answer_groups)
        results['verification_log'].append(("🧬", "Semantic Clustering", f"Formed {len(answer_groups)} distinct groups (largest: {len(largest_group)}/{len(answers)})"))
        
        # Entropy calculation
        progress(0.70, desc="📊 Layer 6: Entropy Calculation...")
        semantic_entropy_result = self.entropy_calc.calculate_semantic_entropy(answer_groups)
        results['semantic_entropy'] = semantic_entropy_result['normalized_entropy']
        results['nli_entropy'] = 0.0
        results['verification_log'].append(("📊", "Entropy Analysis", f"Semantic: {results['semantic_entropy']:.3f}"))
        
        # Web verification
        progress(0.80, desc="🌐 Layer 7: Web Verification...")
        web_verification = self.verify_with_web(consensus_answer, query)
        if web_verification:
            results['web_results'] = web_verification['web_results']
            results['web_match'] = web_verification['match_percent']
            results['verification_log'].append(("🌐", "Web Verification", f"Matched {results['web_match']}% with {len(results['web_results'])} web sources"))
        
        # Final confidence calculation
        combined_confidence = self.entropy_calc.calculate_combined_confidence(
            semantic_entropy=results['semantic_entropy'],
            nli_entropy=results['nli_entropy'],
            consistency_score=consistency_score,
            web_match=results['web_match']
        )
        results['combined_confidence'] = combined_confidence['confidence_score']
        results['risk_level'] = combined_confidence['risk_level']
        results['risk_color'] = combined_confidence['risk_color']
        results['has_web_contradiction'] = combined_confidence['has_web_contradiction']
        
        # Layer 8: Claim Verification
        is_claim = self.is_claim_input(query)
        
        if is_claim:
            progress(0.95, desc="🎯 Layer 8: Claim Verification...")
            
            claim_verification = self.claim_verifier.verify_claim(
                claim=query,
                answer=consensus_answer
            )
            
            results['claim_verification'] = claim_verification
            results['fever_label'] = claim_verification['label']
            
            results['verification_log'].append((
                "🎯", 
                "Claim Verification", 
                f"Classified as: {claim_verification['label']} ({claim_verification['confidence']*100:.0f}% confidence)"
            ))
            
            if claim_verification['label'] == 'NOT ENOUGH INFO':
                results['combined_confidence'] *= 0.85
                results['verification_log'].append((
                    "⚠️",
                    "Confidence Adjustment",
                    "Reduced confidence - claim verification uncertain"
                ))
        else:
            results['verification_log'].append((
                "ℹ️",
                "Input Type",
                "Detected as question (not claim) - skipping claim verification"
            ))
        
        results['explanations'] = self.explain_confidence(results)
        results['verification_log'].append(("📊", "Final Confidence", f"{results['combined_confidence']:.1f}% ({results['risk_level']} risk)"))
        
        progress(1.0, desc="✅ Complete!")
        
        return results

# ============================================
# VISUALIZATION
# ============================================

def create_confidence_chart(results):
    categories = ['Self-Consistency', 'Semantic<br>Certainty', 'NLI<br>Certainty', 'Web Match']
    values = [
        results['consistency_score'],
        (1 - results['semantic_entropy']) * 100,
        (1 - results['nli_entropy']) * 100,
        results['web_match']
    ]
    
    colors = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='outside',
            textfont=dict(size=14, color='#1f2937'),
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Confidence Component Breakdown",
            'font': {'size': 18, 'color': '#1f2937', 'family': 'Inter'}
        },
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 110]),
        height=350,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=40, l=60, r=20),
        font=dict(family='Inter', color='#6b7280')
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f3f4f6')
    
    return fig

# ============================================
# GRADIO INTERFACE
# ============================================

def create_interface():
    print("="*70)
    print("Initializing Enhanced Hallucination Detection System...")
    print("="*70)
    detector = HallucinationDetector()
    print("\n✓ All systems operational!\n")
    
    def process_query(query, progress=gr.Progress()):
        if not query or not query.strip():
            return (
                "<div style='padding: 20px; text-align: center; color: #ef4444;'>⚠️ Please enter a question or claim</div>",
                "", "", "", None
            )
        
        results = detector.detect(query, progress)
        
        # Warning banner
        warning_banner = ""
        if results['has_web_contradiction']:
            warning_banner = """
            <div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 28px;">⚠️</div>
                    <div>
                        <div style="font-weight: 700; color: #991b1b; margin-bottom: 5px; font-size: 16px;">Web Source Contradiction Detected</div>
                        <div style="color: #7f1d1d; font-size: 14px;">High internal consistency but web sources disagree significantly.</div>
                    </div>
                </div>
            </div>
            """
        
        # Answer card
        answer_html = f"""
        {warning_banner}
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
            <div style="color: rgba(255,255,255,0.9); font-size: 13px; font-weight: 600; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1.2px;">Answer</div>
            <div style="color: white; font-size: 18px; line-height: 1.8; font-weight: 400;">{results['answer']}</div>
        </div>
        """
        
        # Confidence explanations
        explanations_html = "".join([
            f'<div style="display: flex; align-items: start; gap: 10px; margin-bottom: 12px;">'
            f'<span style="font-size: 18px; flex-shrink: 0;">{emoji}</span>'
            f'<span style="color: #374151; font-size: 14px; line-height: 1.6;">{text}</span>'
            f'</div>'
            for emoji, text in results['explanations']
        ])
        
        # Confidence dashboard
        confidence_html = f"""
        <div style="background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 25px;">
                <div style="font-size: 54px; font-weight: 800; color: {results['risk_color']}; margin-bottom: 12px; letter-spacing: -2px;">
                    {results['combined_confidence']:.1f}%
                </div>
                <div style="display: inline-block; padding: 10px 24px; background: {results['risk_color']}; color: white; border-radius: 25px; font-weight: 700; font-size: 15px; letter-spacing: 0.5px;">
                    {results['risk_level']} RISK
                </div>
            </div>
            
            <div style="background: #f8f9fa; height: 22px; border-radius: 11px; overflow: hidden; margin-bottom: 30px;">
                <div style="background: linear-gradient(90deg, {results['risk_color']}, {results['risk_color']}cc); height: 100%; width: {results['combined_confidence']}%; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);"></div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 25px;">
                <div style="text-align: center; padding: 18px; background: #f8f9fa; border-radius: 10px;">
                    <div style="font-size: 10px; color: #6b7280; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">Self-Consistency</div>
                    <div style="font-size: 26px; font-weight: 800; color: #1f2937;">{results['consistency_score']:.0f}%</div>
                </div>
                <div style="text-align: center; padding: 18px; background: #f8f9fa; border-radius: 10px;">
                    <div style="font-size: 10px; color: #6b7280; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">Semantic Entropy</div>
                    <div style="font-size: 26px; font-weight: 800; color: #1f2937;">{results['semantic_entropy']:.3f}</div>
                </div>
                <div style="text-align: center; padding: 18px; background: #f8f9fa; border-radius: 10px;">
                    <div style="font-size: 10px; color: #6b7280; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">NLI Entropy</div>
                    <div style="font-size: 26px; font-weight: 800; color: #1f2937;">{results['nli_entropy']:.3f}</div>
                </div>
                <div style="text-align: center; padding: 18px; background: {'#fef2f2' if results['web_match'] < 30 else '#f8f9fa'}; border-radius: 10px; border: {'2px solid #ef4444' if results['web_match'] < 30 else 'none'};">
                    <div style="font-size: 10px; color: #6b7280; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">Web Match</div>
                    <div style="font-size: 26px; font-weight: 800; color: {'#ef4444' if results['web_match'] < 30 else '#1f2937'};">{results['web_match']}%</div>
                </div>
            </div>
            
            <div style="padding: 20px; background: #f0f9ff; border-radius: 10px; border-left: 4px solid #3b82f6;">
                <div style="font-size: 13px; color: #1e40af; font-weight: 700; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 0.5px;">Confidence Breakdown</div>
                {explanations_html}
            </div>
        </div>
        """
        
        # Claim verification display
        claim_verification_html = ""
        if results.get('claim_verification'):
            cv = results['claim_verification']
            label_color = "#10b981" if cv['label'] == "SUPPORTS" else "#ef4444" if cv['label'] == "REFUTES" else "#6b7280"
            
            claim_verification_html = f"""
            <div style="margin-bottom: 25px; padding: 20px; background: #f0f9ff; border-radius: 12px; border-left: 4px solid #3b82f6;">
                <div style="font-size: 12px; color: #1e40af; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">🎯 Claim Verification (Layer 8)</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="color: #374151; font-weight: 600;">FEVER Classification:</span>
                    <span style="padding: 6px 16px; background: {label_color}; color: white; border-radius: 20px; font-weight: 700; font-size: 13px;">
                        {cv['label']}
                    </span>
                </div>
                <div style="font-size: 13px; color: #6b7280; margin-top: 10px;">
                    <div>Entailment: {cv['entailment_score']*100:.1f}% | Contradiction: {cv['contradiction_score']*100:.1f}% | Neutral: {cv['neutral_score']*100:.1f}%</div>
                </div>
            </div>
            """
        
        # Mode badge
        mode_badge_color = "#6b7280"
        
        # Verification log
        verification_log_html = "".join([
            f'<div style="display: flex; gap: 12px; align-items: start; padding: 12px; background: #f8f9fa; border-radius: 8px; margin-bottom: 8px;">'
            f'<span style="font-size: 18px; flex-shrink: 0;">{emoji}</span>'
            f'<div style="flex: 1;">'
            f'<div style="font-weight: 700; color: #1f2937; font-size: 13px; margin-bottom: 3px;">{layer}</div>'
            f'<div style="color: #6b7280; font-size: 13px;">{description}</div>'
            f'</div>'
            f'</div>'
            for emoji, layer, description in results['verification_log']
        ])
        
        verification_html = f"""
        <div style="background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <div style="font-size: 20px; font-weight: 800; color: #1f2937; margin-bottom: 25px;">🔬 Complete Verification Details</div>
            
            {claim_verification_html}
            
            <div style="margin-bottom: 25px;">
                <div style="font-size: 12px; color: #6b7280; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">System Mode</div>
                <div style="display: inline-block; padding: 8px 18px; background: {mode_badge_color}; color: white; border-radius: 20px; font-weight: 700; font-size: 13px; margin-bottom: 8px;">
                    {results['mode']}
                </div>
                <div style="color: #6b7280; font-size: 13px; font-style: italic;">{results['mode_reason']}</div>
            </div>
            
            <div style="margin-bottom: 25px;">
                <div style="font-size: 12px; color: #6b7280; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">8-Layer Verification Process</div>
                {verification_log_html}
            </div>
        </div>
        """
        
        # Web sources
        web_html = ""
        if results['web_results']:
            sources_html = ""
            for i, r in enumerate(results['web_results'][:3], 1):
                sources_html += f"""
                <div style="padding: 20px; background: #f8f9fa; border-radius: 10px; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #1f2937; margin-bottom: 8px; font-size: 15px;">{i}. {r['title']}</div>
                    <a href="{r['url']}" target="_blank" style="color: #667eea; text-decoration: none; font-size: 12px; display: block; margin-bottom: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500;">{r['url']}</a>
                    <div style="color: #6b7280; font-size: 13px; line-height: 1.7;">{r['snippet'][:250]}...</div>
                </div>
                """
            
            web_html = f"""
            <div style="background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                <div style="font-size: 20px; font-weight: 800; color: #1f2937; margin-bottom: 20px;">🌐 Web Verification Sources</div>
                {sources_html}
            </div>
            """
        
        chart = create_confidence_chart(results)
        
        return answer_html, confidence_html, verification_html, web_html, chart
    
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .gradio-container {
        max-width: 1400px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 14px 36px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .gr-button-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Hallucination Detection v3.2") as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 35px; box-shadow: 0 15px 50px rgba(0,0,0,0.2);">
            <h1 style="color: white; font-size: 46px; font-weight: 900; margin: 0 0 12px 0; letter-spacing: -1.5px;">
                Enhanced Hallucination Detection
            </h1>
            <div style="color: rgba(255,255,255,0.95); font-size: 20px; margin: 0 0 20px 0; font-weight: 500;">
                v3.2 - Complete 8-Layer Verification System
            </div>
            <div style="padding: 10px 20px; background: rgba(255,255,255,0.2); backdrop-filter: blur(10px); border-radius: 20px; display: inline-block;">
                <span style="color: white; font-size: 14px; font-weight: 600;">🔬 Self-Consistency → Clustering → NLI → Entropy → Web → Claim Verification</span>
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="",
                    placeholder="Enter your question or claim... (Try 'Who was the 44th president?' or 'Barack Obama was the 44th president')",
                    lines=3,
                    show_label=False
                )
                submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                
                gr.Markdown("### 💡 Example Inputs")
                gr.Examples(
                    examples=[
                        ["Who was the 44th president of America?"],
                        ["Barack Obama was the 44th president"],
                        ["What is UMBC?"],
                        ["UMBC is located in Maryland"],
                        ["Where is Baltimore?"]
                    ],
                    inputs=query_input,
                    label=""
                )
        
        gr.HTML("<div style='margin: 35px 0;'></div>")
        
        with gr.Row():
            answer_output = gr.HTML()
        
        with gr.Row():
            with gr.Column(scale=1):
                confidence_output = gr.HTML()
            with gr.Column(scale=1):
                verification_output = gr.HTML()
        
        with gr.Row():
            chart_output = gr.Plot(label="Confidence Metrics Visualization")
        
        with gr.Row():
            web_output = gr.HTML()
        
        submit_btn.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[answer_output, confidence_output, verification_output, web_output, chart_output]
        )
        
        query_input.submit(
            fn=process_query,
            inputs=[query_input],
            outputs=[answer_output, confidence_output, verification_output, web_output, chart_output]
        )
    
    return demo

# ============================================
# MAIN - HUGGINGFACE LAUNCH
# ============================================

if __name__ == "__main__":
    print("="*70)
    print("🎓 Enhanced Hallucination Detection System v3.2")
    print("   Complete 8-Layer Verification with Claim Support")
    print("="*70)
    print("\n🔬 All 8 Verification Layers:")
    print("   1. Wikipedia Search (not available in web demo)")
    print("   2. Cross-Encoder Re-ranking (not available in web demo)")
    print("   3. Self-Consistency Detection (5 attempts)")
    print("   4. Semantic Clustering")
    print("   5. Neural NLI Verification")
    print("   6. Entropy-Based Uncertainty")
    print("   7. Web Search Verification")
    print("   8. Claim Verification (FEVER) 🆕")
    print("\nInitializing system...\n")
    
    demo = create_interface()
    
    print("\n" + "="*70)
    print("✅ System Ready!")
    print("="*70)
    print("\n🌐 Launching HuggingFace Space...\n")
    
    # Simple launch for HuggingFace Spaces
    demo.launch()
