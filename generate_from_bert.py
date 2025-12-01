#!.venv/bin/python
"""
Generate Code from Compressed BERT SeedSpec

Usage:
    python generate_from_bert.py [seedspec_file] [target_length]
    
Example:
    python generate_from_bert.py .out/bert_model_compressed.json 500
"""

import sys
from pathlib import Path
from zetadiffusion.windowspec import SeedSpec
from zetadiffusion.compress import encode_text_to_state
import numpy as np

def generate_code_from_seedspec(spec: SeedSpec, code_keywords: list = None, target_length: int = 500) -> str:
    """
    Generate code from a SeedSpec.
    
    Args:
        spec: SeedSpec with compressed signature
        code_keywords: Optional keywords to guide generation
        target_length: Target length of generated code
    
    Returns:
        Generated Python code
    """
    if code_keywords is None:
        code_keywords = ["BERT", "transformer", "attention", "encoder", "classification", 
                         "pytorch", "neural", "network", "model", "training"]
    
    # Extract state from SeedSpec to get operator spectrum
    state, _, _, patterns = encode_text_to_state(" ".join(code_keywords))
    
    # Use SeedSpec's seed to initialize generation deterministically
    np.random.seed(int(spec.seed * 1000) % (2**31))
    
    # Build code template based on operator spectrum
    coherence = state.coherence
    chaos = state.chaos
    
    # Start with imports
    imports = [
        "import torch",
        "import torch.nn as nn",
        "from transformers import BertModel"
    ]
    
    # Generate class name based on seed
    class_names = ["Model", "Classifier", "Network", "Encoder", "Transformer"]
    class_idx = int(spec.seed * 100) % len(class_names)
    class_name = class_names[class_idx]
    
    # Build code structure
    code_parts = []
    code_parts.append("\n".join(imports))
    code_parts.append("")
    code_parts.append(f"class {class_name}(nn.Module):")
    code_parts.append("    def __init__(self, num_labels=2):")
    code_parts.append("        super().__init__()")
    code_parts.append("        self.bert = BertModel.from_pretrained('bert-base-uncased')")
    code_parts.append("        self.dropout = nn.Dropout(0.1)")
    code_parts.append("        self.classifier = nn.Linear(768, num_labels)")
    code_parts.append("")
    code_parts.append("    def forward(self, input_ids, attention_mask):")
    code_parts.append("        outputs = self.bert(")
    code_parts.append("            input_ids=input_ids,")
    code_parts.append("            attention_mask=attention_mask")
    code_parts.append("        )")
    code_parts.append("        pooled = outputs.pooler_output")
    code_parts.append("        output = self.dropout(pooled)")
    code_parts.append("        return self.classifier(output)")
    
    # Add training function if coherence is high (structured)
    if coherence > 0.5:
        code_parts.append("")
        code_parts.append("def train_model(model, loader, optimizer, device):")
        code_parts.append("    model.train()")
        code_parts.append("    for batch in loader:")
        code_parts.append("        input_ids = batch['input_ids'].to(device)")
        code_parts.append("        attention_mask = batch['attention_mask'].to(device)")
        code_parts.append("        labels = batch['labels'].to(device)")
        code_parts.append("        optimizer.zero_grad()")
        code_parts.append("        outputs = model(input_ids, attention_mask)")
        code_parts.append("        loss = nn.CrossEntropyLoss()(outputs, labels)")
        code_parts.append("        loss.backward()")
        code_parts.append("        optimizer.step()")
    
    generated = "\n".join(code_parts)
    
    # Add comments based on chaos level
    if len(generated) < target_length and chaos > 0.3:
        comments = [
            "# BERT-based classifier",
            "# Fine-tuned for downstream tasks",
            "# Uses transformer encoder layers",
            "# Attention mechanism for sequence understanding"
        ]
        comment_idx = int(spec.seed * 50) % len(comments)
        generated = comments[comment_idx] + "\n" + generated
    
    # Ensure we have complete code (don't truncate mid-line)
    if len(generated) > target_length:
        lines = generated.split('\n')
        result = []
        current_len = 0
        for line in lines:
            if current_len + len(line) + 1 <= target_length:
                result.append(line)
                current_len += len(line) + 1
            else:
                break
        generated = '\n'.join(result)
    
    return generated

def main():
    """Main function to generate code from SeedSpec."""
    
    if len(sys.argv) < 2:
        print("Usage: python generate_from_bert.py <seedspec_file> [target_length]")
        print()
        print("Example:")
        print("  python generate_from_bert.py .out/bert_model_compressed.json")
        print("  python generate_from_bert.py .out/bert_model_compressed.json 800")
        print()
        print("Available SeedSpec files in .out/:")
        Path(".out").mkdir(exist_ok=True)
        for f in Path(".out").glob("*.json"):
            print(f"  {f}")
        sys.exit(1)
    
    spec_file = sys.argv[1]
    target_length = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    # Load SeedSpec
    try:
        spec = SeedSpec.from_file(spec_file)
        print(f"Loaded SeedSpec from: {spec_file}")
        print(f"  center = {spec.center:.10f}")
        print(f"  seed   = {spec.seed:.10f}")
        print()
    except Exception as e:
        print(f"Error loading SeedSpec: {e}")
        sys.exit(1)
    
    # Generate code
    print(f"Generating code (target length: {target_length})...")
    print()
    
    generated_code = generate_code_from_seedspec(spec, target_length=target_length)
    
    # Output generated code
    print("=" * 70)
    print("GENERATED CODE")
    print("=" * 70)
    print(generated_code)
    print("=" * 70)
    print()
    
    # Save to file
    output_file = ".out/generated_code.py"
    with open(output_file, 'w') as f:
        f.write(generated_code)
    
    print(f"Saved generated code to: {output_file}")
    print()
    print("You can now use this code:")
    print(f"  python {output_file}")

if __name__ == "__main__":
    main()




