<<<<<<< HEAD
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import asyncio

class SemanticAnalyzer:
    def __init__(self, model_path=None):
        import os

        # Try a few common locations
        paths_to_try = [
            model_path,
            "./models/turn_detection_model",
            "../models/turn_detection_model",
            "./turn_detection_model",
        ]

        selected_path = None
        for p in paths_to_try:
            if p and os.path.exists(p):
                selected_path = os.path.abspath(p)
                break

        if not selected_path:
            print(f"Warning: No fine-tuned model found in {paths_to_try}")
            self.model = None
            self.tokenizer = None
            return

        print(f"Loading model from: {selected_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(selected_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(selected_path)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            print("Fine-tuned Turn Detection Model loaded.")
        except Exception as e:
            print(f"Failed to load fine-tuned model: {e}")
            self.model = None
            self.tokenizer = None

    def check_rules(self, text: str) -> bool | None:
        """
        Rule-based checks for specific entities (Phone, CC).
        Returns:
            - True: Definitely COMPLETE
            - False: Definitely INCOMPLETE
            - None: Unknown, let model decide
        """
        digits_seqs = re.findall(r'\d+', text)

        for seq in digits_seqs:
            length = len(seq)
            if 5 <= length <= 9:
                return False  # Incomplete mobile
            if 11 <= length <= 12:
                return False  # Incomplete CC

        return None

    async def is_sentence_complete(self, text: str) -> tuple[bool, float]:
        """
        Async determination of sentence completeness.
        Returns (is_complete, confidence_score)
        """
        rule_result = self.check_rules(text)
        if rule_result is not None:
            return rule_result, 1.0

        if self.model:
            return await asyncio.get_event_loop().run_in_executor(None, self._model_check, text)

        return False, 0.0

    def _model_check(self, text: str) -> tuple[bool, float]:
        """
        Inference using fine-tuned classifier
        """
        if not text.strip():
            return False, 0.0

        clean_text = text.replace(".", "").replace("?", "").replace("!", "").strip()

        inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128)

        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        confidence = probs[0][1].item()
        predicted_class_id = logits.argmax().item()

        return predicted_class_id == 1, confidence
=======
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import asyncio

class SemanticAnalyzer:
    def __init__(self, model_path=None):
        import os

        # Try a few common locations
        paths_to_try = [
            model_path,
            "./models/turn_detection_model",
            "../models/turn_detection_model",
            "./turn_detection_model",
        ]

        selected_path = None
        for p in paths_to_try:
            if p and os.path.exists(p):
                selected_path = os.path.abspath(p)
                break

        if not selected_path:
            print(f"Warning: No fine-tuned model found in {paths_to_try}")
            self.model = None
            self.tokenizer = None
            return

        print(f"Loading model from: {selected_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(selected_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(selected_path)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            print("Fine-tuned Turn Detection Model loaded.")
        except Exception as e:
            print(f"Failed to load fine-tuned model: {e}")
            self.model = None
            self.tokenizer = None

    def check_rules(self, text: str) -> bool | None:
        """
        Rule-based checks for specific entities (Phone, CC).
        Returns:
            - True: Definitely COMPLETE
            - False: Definitely INCOMPLETE
            - None: Unknown, let model decide
        """
        digits_seqs = re.findall(r'\d+', text)

        for seq in digits_seqs:
            length = len(seq)
            if 5 <= length <= 9:
                return False  # Incomplete mobile
            if 11 <= length <= 12:
                return False  # Incomplete CC

        return None

    async def is_sentence_complete(self, text: str) -> tuple[bool, float]:
        """
        Async determination of sentence completeness.
        Returns (is_complete, confidence_score)
        """
        rule_result = self.check_rules(text)
        if rule_result is not None:
            return rule_result, 1.0

        if self.model:
            return await asyncio.get_event_loop().run_in_executor(None, self._model_check, text)

        return False, 0.0

    def _model_check(self, text: str) -> tuple[bool, float]:
        """
        Inference using fine-tuned classifier
        """
        if not text.strip():
            return False, 0.0

        clean_text = text.replace(".", "").replace("?", "").replace("!", "").strip()

        inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128)

        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        confidence = probs[0][1].item()
        predicted_class_id = logits.argmax().item()

        return predicted_class_id == 1, confidence
>>>>>>> 73d9573 (safetensor uploaded)
