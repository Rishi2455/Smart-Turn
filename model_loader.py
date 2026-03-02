"""
Async Model Loader & Inference Engine for EOU Detection
Supports ONNX Runtime (fast) with PyTorch fallback.
"""

import os
import re
import json
import asyncio
import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger("eou_model")

# Try importing ONNX Runtime first, then PyTorch as fallback
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("ONNX Runtime available — will use fast inference path")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not installed — falling back to PyTorch")

try:
    import torch
    import torch.nn as nn
    from transformers import DebertaV2Model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from transformers import AutoTokenizer


# ============================================================
# Config & Feature Extraction
# ============================================================

@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 128  # Reduced from 256 — EOU utterances are short
    use_aux_features: bool = True
    dropout: float = 0.1
    label_smoothing: float = 0.05


class TextCleaner:
    """Clean text for ASR-trained model (no punctuation expected)"""

    # Compile regex once for performance
    _PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
    _MULTI_SPACE_RE = re.compile(r'\s+')

    @classmethod
    def clean(cls, text: str) -> str:
        """Strip punctuation, lowercase, and normalize whitespace."""
        text = text.strip()
        if not text:
            return text
        text = cls._PUNCT_RE.sub('', text)       # Remove all punctuation
        text = cls._MULTI_SPACE_RE.sub(' ', text) # Collapse multiple spaces
        text = text.strip().lower()               # Lowercase for ASR input
        return text


class SemanticFeatureExtractor:
    """Extract 15 semantic features for EOU detection (punctuation-free).

    Matches the feature_type='semantic_no_punctuation' training config.
    """

    CONJUNCTIONS = {'and', 'but', 'or', 'so', 'because', 'since', 'although',
                    'while', 'if', 'when', 'that', 'which', 'who', 'where',
                    'unless', 'until', 'whether', 'though', 'whereas'}

    PREPOSITIONS = {'to', 'for', 'with', 'at', 'in', 'on', 'of', 'from',
                    'by', 'about', 'into', 'through', 'during', 'before',
                    'after', 'above', 'below', 'between', 'under', 'over'}

    ARTICLES = {'a', 'an', 'the'}

    SUBJECT_PRONOUNS = {'i', 'we', 'they', 'he', 'she', 'it', 'you'}

    AUXILIARIES = {'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                   'have', 'has', 'had', 'do', 'does', 'did',
                   'will', 'would', 'shall', 'should',
                   'can', 'could', 'may', 'might', 'must'}

    COMMON_TRANSITIVE = {'get', 'got', 'take', 'took', 'make', 'made',
                         'give', 'gave', 'tell', 'told', 'find', 'found',
                         'know', 'knew', 'want', 'need', 'see', 'saw',
                         'put', 'keep', 'kept', 'let', 'say', 'said',
                         'think', 'thought', 'ask', 'asked', 'use', 'used',
                         'show', 'showed', 'try', 'tried', 'buy', 'bought'}

    # Common verbs for has_verb detection
    COMMON_VERBS = AUXILIARIES | COMMON_TRANSITIVE | {
        'go', 'went', 'come', 'came', 'run', 'ran', 'look', 'looked',
        'like', 'liked', 'play', 'played', 'work', 'worked', 'call',
        'called', 'move', 'moved', 'live', 'lived', 'believe', 'happen',
        'happened', 'include', 'included', 'turn', 'turned', 'follow',
        'followed', 'begin', 'began', 'seem', 'seemed', 'help', 'helped',
        'talk', 'talked', 'start', 'started', 'write', 'wrote', 'read',
        'feel', 'felt', 'provide', 'hold', 'held', 'stand', 'stood',
        'set', 'learn', 'learned', 'change', 'changed', 'lead', 'led',
        'understand', 'understood', 'watch', 'watched', 'pay', 'paid',
        'bring', 'brought', 'meet', 'met', 'send', 'sent', 'build',
        'built', 'stay', 'stayed', 'open', 'opened', 'create', 'created'
    }

    COMMON_NOUNS_SIMPLE = {
        'time', 'year', 'people', 'way', 'day', 'man', 'woman', 'child',
        'world', 'life', 'hand', 'part', 'place', 'case', 'week', 'company',
        'system', 'program', 'question', 'work', 'government', 'number',
        'night', 'point', 'home', 'water', 'room', 'mother', 'area',
        'money', 'story', 'fact', 'month', 'lot', 'right', 'study',
        'book', 'eye', 'job', 'word', 'business', 'issue', 'side', 'kind',
        'head', 'house', 'service', 'friend', 'father', 'power', 'hour',
        'game', 'line', 'end', 'members', 'city', 'community',
        'name', 'president', 'team', 'minute', 'idea', 'body', 'information',
        'back', 'parent', 'face', 'others', 'level', 'office', 'door',
        'health', 'person', 'art', 'car', 'food', 'phone', 'thing',
        'things', 'problem', 'answer', 'account', 'card', 'payment'
    }

    DISCOURSE_MARKERS = {'well', 'so', 'like', 'okay', 'ok', 'yeah',
                         'yes', 'no', 'right', 'sure', 'actually',
                         'basically', 'honestly', 'anyway', 'alright',
                         'exactly', 'absolutely', 'definitely', 'totally'}

    ADVERBS = {'very', 'really', 'also', 'just', 'now', 'then', 'still',
               'already', 'always', 'never', 'often', 'sometimes',
               'usually', 'quickly', 'slowly', 'well', 'too', 'quite',
               'almost', 'enough', 'only', 'even', 'probably', 'maybe',
               'certainly', 'finally', 'recently', 'actually', 'simply',
               'clearly', 'completely', 'especially', 'generally'}

    FUNCTION_WORDS = (
        CONJUNCTIONS | PREPOSITIONS | ARTICLES
        | SUBJECT_PRONOUNS | AUXILIARIES
        | {'the', 'a', 'an', 'this', 'that', 'these', 'those',
           'my', 'your', 'his', 'her', 'its', 'our', 'their',
           'not', 'no', 'very', 'just', 'also', 'too'}
    )

    @classmethod
    def extract(cls, text: str) -> List[float]:
        """Extract 15 semantic features (no punctuation features)."""
        text = text.strip()
        words = text.lower().split()
        num_words = len(words)
        last_word = words[-1] if words else ''

        # Check if text has a verb anywhere
        has_verb = float(any(w in cls.COMMON_VERBS for w in words))

        # Check if there's a subject followed by a verb (simple heuristic)
        has_subj_verb = 0.0
        for i in range(len(words) - 1):
            if words[i] in cls.SUBJECT_PRONOUNS and words[i + 1] in cls.COMMON_VERBS:
                has_subj_verb = 1.0
                break

        # Check if a verb appeared earlier and last word is a noun
        verb_seen = any(w in cls.COMMON_VERBS for w in words[:-1]) if num_words > 1 else False
        ends_noun_after_verb = float(
            verb_seen and last_word in cls.COMMON_NOUNS_SIMPLE
        )

        # Check if last word looks like a complete content word
        # (not a function word, and at least 3 chars)
        ends_complete_word = float(
            last_word not in cls.FUNCTION_WORDS
            and len(last_word) >= 3
        ) if last_word else 0.0

        # Adverb after verb check
        ends_adverb_after_verb = float(
            verb_seen and last_word in cls.ADVERBS
        )

        # Content word ratio
        content_words = [w for w in words if w not in cls.FUNCTION_WORDS]
        content_ratio = len(content_words) / max(num_words, 1)

        features = [
            float(last_word in cls.CONJUNCTIONS),       # ends_conjunction
            float(last_word in cls.PREPOSITIONS),       # ends_preposition
            float(last_word in cls.ARTICLES),            # ends_article
            float(last_word in cls.SUBJECT_PRONOUNS),   # ends_subject_pronoun
            float(last_word in cls.AUXILIARIES),         # ends_auxiliary
            float(last_word in cls.COMMON_TRANSITIVE),  # ends_transitive
            ends_complete_word,                          # ends_complete_word
            has_verb,                                    # has_verb
            ends_noun_after_verb,                        # ends_noun_after_verb
            float(last_word in cls.DISCOURSE_MARKERS),  # ends_discourse_marker
            min(num_words / 30.0, 1.0),                 # norm_word_count
            has_subj_verb,                               # has_subj_verb
            ends_adverb_after_verb,                      # ends_adverb_after_verb
            float(num_words <= 2),                       # is_very_short
            round(content_ratio, 4),                     # content_ratio
        ]
        return features

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            'ends_conjunction', 'ends_preposition', 'ends_article',
            'ends_subject_pronoun', 'ends_auxiliary', 'ends_transitive',
            'ends_complete_word', 'has_verb', 'ends_noun_after_verb',
            'ends_discourse_marker', 'norm_word_count', 'has_subj_verb',
            'ends_adverb_after_verb', 'is_very_short', 'content_ratio'
        ]


# ============================================================
# PyTorch Model (fallback only — kept for compatibility)
# ============================================================

if TORCH_AVAILABLE:
    class DeBERTaEOUClassifier(nn.Module):
        """DeBERTa with auxiliary features for End-of-Utterance detection"""

        def __init__(self, config: Config, num_aux_features: int = 15):  # 15 semantic features
            super().__init__()
            self.config = config
            self.use_aux = config.use_aux_features

            self.deberta = DebertaV2Model.from_pretrained(config.model_name)
            hidden_size = self.deberta.config.hidden_size

            self.pooler_dropout = nn.Dropout(config.dropout)

            if self.use_aux:
                self.aux_projection = nn.Sequential(
                    nn.Linear(num_aux_features, 32),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
                classifier_input_size = hidden_size + 32
            else:
                classifier_input_size = hidden_size

            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_size, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Dropout(config.dropout),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(64, 2),
            )

        def forward(self, input_ids, attention_mask, token_type_ids=None,
                    aux_features=None, labels=None):

            outputs = self.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            cls_output = outputs.last_hidden_state[:, 0, :]
            cls_output = self.pooler_dropout(cls_output)

            if self.use_aux and aux_features is not None:
                aux_projected = self.aux_projection(aux_features)
                combined = torch.cat([cls_output, aux_projected], dim=-1)
            else:
                combined = cls_output

            logits = self.classifier(combined)

            loss = None
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss(
                    label_smoothing=self.config.label_smoothing
                )
                loss = loss_fn(logits, labels)

            return {'loss': loss, 'logits': logits}


# ============================================================
# Async Inference Engine (ONNX primary, PyTorch fallback)
# ============================================================

class EOUModelEngine:
    """Async model engine — uses ONNX Runtime for fast inference"""

    def __init__(self):
        self.onnx_session = None          # ONNX Runtime session
        self.torch_model = None           # PyTorch model (fallback)
        self.tokenizer: Optional[Any] = None
        self.feature_extractor = SemanticFeatureExtractor()
        self.device = None
        self.threshold: float = 0.5
        self.eou_config: Dict = {}
        self.is_loaded: bool = False
        self.model_dir: str = ""
        self.backend: str = ""            # "onnx" or "pytorch"
        self.max_length: int = 128        # Reduced default

        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = asyncio.Lock()

    async def load_model(self, model_dir: str) -> Dict:
        """Load model — prefers ONNX, falls back to PyTorch"""
        async with self._lock:
            logger.info(f"Loading model from {model_dir}...")
            start_time = time.time()

            try:
                # Load config
                config_path = os.path.join(model_dir, 'eou_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        self.eou_config = json.load(f)
                    self.threshold = self.eou_config.get('best_threshold', 0.5)
                else:
                    self.eou_config = {}
                    self.threshold = 0.5

                # Use reduced max_length (128) unless config says otherwise
                self.max_length = min(
                    self.eou_config.get('max_length', 128), 128
                )

                # Load tokenizer (in thread to not block event loop)
                loop = asyncio.get_event_loop()
                self.tokenizer = await loop.run_in_executor(
                    self._executor,
                    lambda: AutoTokenizer.from_pretrained(model_dir)
                )
                # Try ONNX first
                onnx_path = os.path.join(model_dir, 'eou_model.onnx')
                if ONNX_AVAILABLE and os.path.exists(onnx_path):
                    self.backend = "onnx"
                    self.onnx_session = await loop.run_in_executor(
                        self._executor,
                        lambda: self._create_onnx_session(onnx_path)
                    )
                    logger.info("✅ Loaded ONNX model (fast path)")

                elif TORCH_AVAILABLE:
                    self.backend = "pytorch"
                    logger.info("⚠️ ONNX model not found, using PyTorch fallback")
                    self.device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )

                    model_config = Config()
                    model_config.model_name = self.eou_config.get(
                        'model_name', 'microsoft/deberta-v3-base'
                    )
                    model_config.use_aux_features = self.eou_config.get(
                        'use_aux_features', True
                    )
                    num_aux = self.eou_config.get('num_aux_features', 15)

                    def _load_pytorch():
                        model = DeBERTaEOUClassifier(
                            model_config, num_aux_features=num_aux
                        )
                        weights_path = os.path.join(
                            model_dir, 'pytorch_model_full.pt'
                        )
                        if os.path.exists(weights_path):
                            state_dict = torch.load(
                                weights_path,
                                map_location=self.device,
                                weights_only=True,
                            )
                        else:
                            for alt in ['model.safetensors', 'pytorch_model.bin']:
                                alt_path = os.path.join(model_dir, alt)
                                if os.path.exists(alt_path):
                                    state_dict = torch.load(
                                        alt_path,
                                        map_location=self.device,
                                        weights_only=True,
                                    )
                                    break
                            else:
                                raise FileNotFoundError(
                                    f"No model weights found in {model_dir}"
                                )

                        model.load_state_dict(state_dict, strict=False)
                        model.to(self.device)
                        model.eval()
                        return model

                    self.torch_model = await loop.run_in_executor(
                        self._executor, _load_pytorch
                    )
                else:
                    raise RuntimeError(
                        "Neither onnxruntime nor torch is available!"
                    )

                self.model_dir = model_dir
                self.is_loaded = True
                load_time = time.time() - start_time

                info = {
                    "status": "loaded",
                    "backend": self.backend,
                    "model_dir": model_dir,
                    "device": str(self.device) if self.device else "cpu",
                    "threshold": self.threshold,
                    "max_length": self.max_length,
                    "load_time_seconds": round(load_time, 2),
                    "model_name": self.eou_config.get(
                        'model_name', 'microsoft/deberta-v3-base'
                    ),
                    "use_aux_features": self.eou_config.get(
                        'use_aux_features', True
                    ),
                }
                logger.info(
                    f"Model loaded in {load_time:.2f}s "
                    f"[backend={self.backend}]"
                )
                return info

            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                self.is_loaded = False
                raise

    @staticmethod
    def _create_onnx_session(onnx_path: str):
        """Create an optimized ONNX Runtime session"""
        opts = ort.SessionOptions()
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.inter_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Use CPUExecutionProvider (add CUDAExecutionProvider if GPU)
        providers = ['CPUExecutionProvider']
        return ort.InferenceSession(
            onnx_path, sess_options=opts, providers=providers
        )

    # ----------------------------------------------------------
    # Prediction — ONNX path (fast)
    # ----------------------------------------------------------

    def _predict_onnx(self, text: str) -> Dict:
        """ONNX Runtime prediction — significantly faster on CPU"""
        start_time = time.time()

        # Clean text for ASR-trained model (strip punctuation)
        clean_text = TextCleaner.clean(text)

        # Tokenize with DYNAMIC padding (key optimization!)
        encoding = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=self.max_length,
            padding=True,               # Dynamic padding
            return_tensors='np',
        )

        # Build ONNX input feed
        feed = {
            'input_ids': encoding['input_ids'].astype(np.int64),
            'attention_mask': encoding['attention_mask'].astype(np.int64),
        }

        # Add token_type_ids if the model expects it
        onnx_input_names = [inp.name for inp in self.onnx_session.get_inputs()]
        if 'token_type_ids' in onnx_input_names:
            if 'token_type_ids' in encoding:
                feed['token_type_ids'] = (
                    encoding['token_type_ids'].astype(np.int64)
                )
            else:
                feed['token_type_ids'] = np.zeros_like(
                    encoding['input_ids'], dtype=np.int64
                )

        # Add auxiliary features if the model expects them
        if 'aux_features' in onnx_input_names:
            aux = np.array(
                [self.feature_extractor.extract(clean_text)], dtype=np.float32
            )
            feed['aux_features'] = aux

        # Run inference
        outputs = self.onnx_session.run(None, feed)
        logits = outputs[0]  # shape: [1, 2]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        probs = probs[0]

        complete_prob = float(probs[1])
        incomplete_prob = float(probs[0])
        is_complete = complete_prob >= self.threshold

        inference_time = time.time() - start_time

        # Feature analysis
        features = self.feature_extractor.extract(clean_text)
        feature_names = self.feature_extractor.feature_names()
        feature_analysis = {
            name: round(val, 3) for name, val in zip(feature_names, features)
        }

        return {
            "text": text,
            "is_complete": is_complete,
            "confidence": round(float(max(probs)), 4),
            "complete_probability": round(complete_prob, 4),
            "incomplete_probability": round(incomplete_prob, 4),
            "threshold": self.threshold,
            "inference_time_ms": round(inference_time * 1000, 2),
            "features": feature_analysis,
        }

    # ----------------------------------------------------------
    # Prediction — PyTorch path (fallback)
    # ----------------------------------------------------------

    def _predict_pytorch(self, text: str) -> Dict:
        """PyTorch prediction (fallback if ONNX not available)"""
        start_time = time.time()

        # Clean text for ASR-trained model (strip punctuation)
        clean_text = TextCleaner.clean(text)

        encoding = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=self.max_length,
            padding=True,               # Dynamic padding fix
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        aux_features = torch.tensor(
            [self.feature_extractor.extract(clean_text)], dtype=torch.float
        ).to(self.device)

        with torch.no_grad():
            outputs = self.torch_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                aux_features=aux_features,
            )

        probs = torch.softmax(outputs['logits'], dim=-1)[0].cpu().numpy()
        complete_prob = float(probs[1])
        incomplete_prob = float(probs[0])
        is_complete = complete_prob >= self.threshold

        inference_time = time.time() - start_time

        features = self.feature_extractor.extract(clean_text)
        feature_names = self.feature_extractor.feature_names()
        feature_analysis = {
            name: round(val, 3) for name, val in zip(feature_names, features)
        }

        return {
            "text": text,
            "is_complete": is_complete,
            "confidence": round(float(max(probs)), 4),
            "complete_probability": round(complete_prob, 4),
            "incomplete_probability": round(incomplete_prob, 4),
            "threshold": self.threshold,
            "inference_time_ms": round(inference_time * 1000, 2),
            "features": feature_analysis,
        }

    # ----------------------------------------------------------
    # Public async API
    # ----------------------------------------------------------

    async def predict(self, text: str) -> Dict:
        """Async prediction — dispatches to ONNX or PyTorch"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()
        predict_fn = (
            self._predict_onnx if self.backend == "onnx"
            else self._predict_pytorch
        )
        return await loop.run_in_executor(
            self._executor, predict_fn, text
        )

    async def predict_batch(
        self, texts: List[str]
    ) -> List[Dict]:
        """Async batch prediction"""
        tasks = [
            self.predict(text) for text in texts
        ]
        return await asyncio.gather(*tasks)

    async def update_threshold(self, new_threshold: float) -> Dict:
        """Update classification threshold"""
        old_threshold = self.threshold
        self.threshold = max(0.0, min(1.0, new_threshold))
        return {
            "old_threshold": old_threshold,
            "new_threshold": self.threshold,
        }

    def get_status(self) -> Dict:
        """Get model status"""
        return {
            "is_loaded": self.is_loaded,
            "backend": self.backend,
            "model_dir": self.model_dir,
            "device": str(self.device) if self.device else "cpu",
            "threshold": self.threshold,
            "max_length": self.max_length,
            "config": self.eou_config,
        }


# Singleton instance
engine = EOUModelEngine()