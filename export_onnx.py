import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# Legacy Model definition for backward compatibility
# ============================================================

class DeBERTaEOUClassifier(nn.Module):
    def __init__(self, model_name, num_aux_features=15,
                 use_aux_features=True, dropout=0.1, label_smoothing=0.05):
        super().__init__()
        self.use_aux = use_aux_features

        self.deberta = DebertaV2Model.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size

        self.pooler_dropout = nn.Dropout(dropout)

        if self.use_aux:
            self.aux_projection = nn.Sequential(
                nn.Linear(num_aux_features, 32),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            classifier_input_size = hidden_size + 32
        else:
            classifier_input_size = hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                aux_features=None):
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
        return logits


class HFWrapper(nn.Module):
    """Wrapper to just return logits from generic HF models during ONNX export"""
    def __init__(self, m):
        super().__init__()
        self.model = m
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    MODEL_DIR = "./model"

    print("=" * 60)
    print("EOU Model -> ONNX Export")
    print("=" * 60)

    config_path = os.path.join(MODEL_DIR, "eou_config.json")
    is_legacy = os.path.exists(config_path)

    if is_legacy:
        print("[legacy] eou_config.json found.")
        with open(config_path, "r") as f:
            eou_config = json.load(f)

        model_name = eou_config.get("model_name", "microsoft/deberta-v3-base")
        max_length = eou_config.get("max_length", 128)
        num_aux = eou_config.get("num_aux_features", 15)
        use_aux = eou_config.get("use_aux_features", True)
        
        print("[1/5] Loading PyTorch model...")
        start = time.time()

        model = DeBERTaEOUClassifier(
            model_name=model_name,
            num_aux_features=num_aux,
            use_aux_features=use_aux,
        )

        weights_path = os.path.join(MODEL_DIR, "pytorch_model_full.pt")
        if not os.path.exists(weights_path):
            for alt in ["model.safetensors", "pytorch_model.bin"]:
                alt_path = os.path.join(MODEL_DIR, alt)
                if os.path.exists(alt_path):
                    weights_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"No model weights found in {MODEL_DIR}")

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        print(f"   [OK] Loaded in {time.time() - start:.1f}s")

        print("[2/5] Creating dummy inputs...")
        tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)
        
    else:
        print("[standard] No eou_config.json, assuming standard HF model.")
        max_length = 128
        use_aux = False
        num_aux = 0
        
        print("[1/5] Loading PyTorch model...")
        start = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        
        print(f"   [OK] Loaded in {time.time() - start:.1f}s")
        
        print("[2/5] Creating dummy inputs...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    sample_text = "i want to check my account balance"
    encoding = tokenizer(
        sample_text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )

    dummy_input_ids = encoding["input_ids"]
    dummy_attention_mask = encoding["attention_mask"]

    dummy_token_type_ids = encoding.get("token_type_ids")
    if dummy_token_type_ids is None and is_legacy:
        dummy_token_type_ids = torch.zeros_like(dummy_input_ids)

    dummy_aux = torch.randn(1, num_aux) if use_aux else None
    
    print(f"   Input shape: {dummy_input_ids.shape}")

    print("[3/5] Exporting to ONNX...")
    start = time.time()

    onnx_path = os.path.join(MODEL_DIR, "eou_model.onnx")

    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch"},
    }
    
    if is_legacy:
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq_len"}
        dummy_inputs = (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
        if use_aux and dummy_aux is not None:
            input_names.append("aux_features")
            dynamic_axes["aux_features"] = {0: "batch"}
            dummy_inputs = dummy_inputs + (dummy_aux,)
        export_model = model
    else:
        dummy_inputs = (dummy_input_ids, dummy_attention_mask)
        export_model = HFWrapper(model)

    torch.onnx.export(
        export_model,
        dummy_inputs,
        onnx_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"   [OK] Exported in {time.time() - start:.1f}s")
    print(f"   File: {onnx_path} ({onnx_size:.1f} MB)")

    print("[4/5] Verifying ONNX model...")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        feed = {
            "input_ids": dummy_input_ids.numpy().astype(np.int64),
            "attention_mask": dummy_attention_mask.numpy().astype(np.int64),
        }
        if is_legacy:
            feed["token_type_ids"] = dummy_token_type_ids.numpy().astype(np.int64)
            if use_aux and dummy_aux is not None:
                feed["aux_features"] = dummy_aux.numpy().astype(np.float32)

        onnx_out = session.run(None, feed)[0]

        with torch.no_grad():
            if is_legacy:
                if use_aux:
                    pt_out = model(
                        dummy_input_ids, dummy_attention_mask,
                        dummy_token_type_ids, dummy_aux
                    ).numpy()
                else:
                    pt_out = model(
                        dummy_input_ids, dummy_attention_mask,
                        dummy_token_type_ids
                    ).numpy()
            else:
                pt_out = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask).logits.numpy()

        max_diff = np.max(np.abs(onnx_out - pt_out))
        print(f"   Max output difference: {max_diff:.8f}")

        if max_diff < 1e-4:
            print("   [OK] ONNX model verified - outputs match PyTorch!")
        else:
            print("   [WARN] Outputs differ slightly (may still be usable)")

    except ImportError:
        print("   [WARN] onnxruntime not installed - skipping verification")

    print()
    print("=" * 60)
    print("DONE! Restart the server to use the ONNX model")
    print("=" * 60)


if __name__ == "__main__":
    main()
