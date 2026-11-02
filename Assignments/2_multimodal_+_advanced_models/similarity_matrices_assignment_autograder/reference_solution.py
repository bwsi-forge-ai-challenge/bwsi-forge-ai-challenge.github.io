# Reference versions of the graded functions (copied from similarity_matrices_assignment_solution.ipynb).
import numpy as np
import torch
import torch.nn.functional as F


def cosine_similarity(u, v):
    """Cosine similarity between two 1-D NumPy arrays."""
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def l2_normalize(X):
    """Scale every row of a 2-D array to length 1."""
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def similarity_matrix(A, B):
    """S[i, j] = cosine similarity between A[i] and B[j]. One matrix multiplication, no loops."""
    return l2_normalize(A) @ l2_normalize(B).T


def retrieval_accuracy(S):
    """Fraction of rows i whose best-scoring column is i."""
    return float(np.mean(np.argmax(S, axis=1) == np.arange(S.shape[0])))


def info_nce_loss(img_emb, txt_emb, tau):
    """Symmetric InfoNCE loss for a batch of matching (image, text) embeddings."""
    img = F.normalize(img_emb, dim=1)
    txt = F.normalize(txt_emb, dim=1)
    logits = img @ txt.T / tau
    targets = torch.arange(img.shape[0])
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
