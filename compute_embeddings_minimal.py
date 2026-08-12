"""
Minimal embedding script - no sklearn, no scipy.
Just ONNX Runtime + tokenizers + numpy.
Downloads 20 Newsgroups via a simple HTTP fetch.
"""
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import tarfile
import urllib.request
import json

print("="*60)
print(" COMPUTING TRANSFORMER EMBEDDINGS")
print("="*60)

# Step 1: Load 20 Newsgroups - parse the .pkz directly
# The .pkz is joblib-compressed which triggers sklearn. 
# Instead, let's load from the original .venv where sklearn works with numpy 2.x
print("\nStep 1: Loading 20 Newsgroups via subprocess (using .venv with numpy 2.x)...")
import subprocess
import json

# Use the original venv that has sklearn + numpy 2.x working
extract_script = '''
import json, sys
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
# Write texts and targets as JSON lines (one doc per line)
output = {"targets": data.target.tolist(), "target_names": list(data.target_names), "n": len(data.data)}
with open("newsgroups_meta.json", "w") as f:
    json.dump(output, f)
with open("newsgroups_texts.txt", "w", encoding="utf-8") as f:
    for text in data.data:
        # Replace newlines with space, truncate
        clean = text.replace("\\n", " ").replace("\\r", " ")[:512]
        f.write(clean + "\\n")
print(f"Saved {len(data.data)} documents")
'''

# Write and run with the .venv that has sklearn working
with open("_extract_newsgroups.py", "w") as f:
    f.write(extract_script)

result = subprocess.run(
    [r".venv\Scripts\python", "_extract_newsgroups.py"],
    capture_output=True, text=True, cwd="."
)
print(f"  {result.stdout.strip()}")
if result.returncode != 0:
    print(f"  ERROR: {result.stderr[:200]}")
    raise RuntimeError("Failed to extract newsgroups")

# Load the extracted data
with open("newsgroups_meta.json") as f:
    meta = json.load(f)
with open("newsgroups_texts.txt", encoding="utf-8") as f:
    texts = [line.strip() for line in f.readlines()]

targets = np.array(meta["targets"])
target_names = meta["target_names"]
os.remove("_extract_newsgroups.py")
print(f"  Loaded {len(texts)} documents, {len(target_names)} categories")

# Step 2: Load model
print("\nStep 2: Loading MiniLM ONNX model...")
tok = Tokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tok.enable_padding(pad_id=0, pad_token='[PAD]')
tok.enable_truncation(max_length=128)
sess = ort.InferenceSession('models/model.onnx')
print("  Model loaded!")

# Step 3: Embed
print(f"\nStep 3: Embedding {len(texts)} documents (batch_size=64)...")
batch_size = 64
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    enc = tok.encode_batch(batch)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    tids = np.zeros_like(ids)
    
    out = sess.run(None, {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': tids})
    
    # Mean pooling
    emb = (out[0] * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    all_embeddings.append(emb.astype(np.float32))
    
    if (i // batch_size) % 50 == 0:
        print(f"  {min(i+batch_size, len(texts))}/{len(texts)}")

embeddings = np.vstack(all_embeddings)
print(f"\n  Final shape: {embeddings.shape}")

# Step 4: Save
print("\nStep 4: Saving...")
np.savez_compressed('embeddings.npz',
                    embeddings=embeddings,
                    targets=targets,
                    target_names=np.array(target_names))
fsize = os.path.getsize('embeddings.npz') / 1e6
print(f"  Saved embeddings.npz ({fsize:.1f} MB)")
print("\nDONE!")
