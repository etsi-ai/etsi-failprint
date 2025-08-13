# examples/test_cv_embeddings.py

import torch
try:
    from etsi.failprint.cv_embedder import generate_image_embeddings
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from etsi.failprint.cv_embedder import generate_image_embeddings

print("--- Testing CV Embedding Function ---")

# 1. Prepare sample data (make sure these file paths are correct)
sample_image_paths = [
    "examples/test_images/cat.png",
    "examples/test_images/dog.png"
]

print(f"Embedding {len(sample_image_paths)} images...")

# 2. Call the embedding function
# This will also print which device (CPU/GPU) it's using from the cv_embedder.py file
embeddings = generate_image_embeddings(sample_image_paths)

print("\n--- Verification Checks ---")

# 3. Check the output type and shape
print(f"Output type: {type(embeddings)}")
print(f"Output shape: {embeddings.shape}")

# The shape should be (number_of_images, embedding_dimension)
# For ResNet50, the dimension is 2048.

# 4. Look at a sample embedding to confirm it's not empty
print(f"First 10 values of the first embedding:\n{embeddings[0][:10]}")

# 5. Final verification
if embeddings.shape == (len(sample_image_paths), 2048):
    print("\n✅ Verification passed: Output shape is correct.")
else:
    print("\n❌ Verification failed: Output shape is incorrect.")