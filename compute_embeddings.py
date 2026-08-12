"""
Run this script on ANY machine with onnxruntime + tokenizers + numpy + sklearn.
(e.g., home computer, Google Colab, any PC with VC++ Redistributable)

It will:
1. Download MiniLM model from HuggingFace
2. Embed all 20 Newsgroups texts (384-dim vectors)
3. Save as embeddings.npz

Then copy embeddings.npz back to the sj project folder.
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
import onnxruntime as ort
from tokenizers import Tokenizer
import os

print("Step 1: Loading 20 Newsgroups...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
print(f"  {len(data.data)} documents, {len(data.target_names)} categories")

print("\nStep 2: Loading tokenizer and ONNX model...")
tok = Tokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tok.enable_padding(pad_id=0, pad_token='[PAD]')
tok.enable_truncation(max_length=128)

# Download ONNX model if not present
model_path = 'model.onnx'
if not os.path.exists(model_path):
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model.onnx')

sess = ort.InferenceSession(model_path)
print("  Model loaded successfully.")

print("\nStep 3: Embedding all documents (batch_size=64)...")
batch_size = 64
all_embeddings = []

for i in range(0, len(data.data), batch_size):
    batch_texts = data.data[i:i+batch_size]
    
    # Tokenize
    encoded = tok.encode_batch(batch_texts)
    ids = np.array([e.ids for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    tids = np.zeros_like(ids)
    
    # Run inference
    outputs = sess.run(None, {
        'input_ids': ids,
        'attention_mask': mask, 
        'token_type_ids': tids
    })
    
    # Mean pooling
    token_embeddings = outputs[0]  # (batch, seq_len, 384)
    mask_expanded = mask[:, :, np.newaxis].astype(np.float32)
    embeddings = (token_embeddings * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1)
    
    all_embeddings.append(embeddings)
    
    if (i // batch_size) % 50 == 0:
        print(f"  Embedded {min(i+batch_size, len(data.data))}/{len(data.data)} documents")

embeddings = np.vstack(all_embeddings)
print(f"\nFinal embeddings shape: {embeddings.shape}")  # Should be (18846, 384)

print("\nStep 4: Saving to embeddings.npz...")
np.savez_compressed('embeddings.npz',
                    embeddings=embeddings,
                    targets=data.target,
                    target_names=np.array(data.target_names))
print(f"  Saved! File size: {os.path.getsize('embeddings.npz') / 1e6:.1f} MB")
print("\nDone! Copy embeddings.npz to your sj project folder.")
