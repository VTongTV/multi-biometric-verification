
import torch
import torch.nn.functional as F
import hashlib
import numpy as np

class TemplateProtector:
    """
    Implements Template Protection via Feature Chaffing and Orthogonal BioHashing.
    
    Pipeline:
    1. Chaffing: Appends deterministic "dummy" features (chaff) to the raw embedding.
       - Hides true dimensionality.
       - Hides true feature distribution.
    
    2. BioHashing: Projects the chaffed vector onto a random orthogonal basis.
       - Non-invertible without the basis (defined by the key).
       - Preserves distance metrics (cosine similarity) if the same key is used.
       - Provides Unlinkability (different keys -> different templates).
    """
    
    def __init__(self, chaff_dim: int = 512):
        self.chaff_dim = chaff_dim

    def _str_to_seed(self, key: str) -> int:
        """Convert a string key into a deterministic integer seed."""
        hash_object = hashlib.sha256(key.encode())
        hex_dig = hash_object.hexdigest()
        return int(hex_dig, 16) % (2**32)

    def _generate_chaff(self, seed: int, device: str) -> torch.Tensor:
        """Generate deterministic chaff (noise) vector based on seed."""
        # Use a local generator to ensure thread safety and reproducibility
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.randn(1, self.chaff_dim, device=device, generator=g)

    def _get_projection_matrix(self, input_dim: int, output_dim: int, seed: int, device: str) -> torch.Tensor:
        """
        Generate a pseudo-random Orthogonal Projection Matrix.
        Uses Gram-Schmidt or QR decomposition to ensure orthogonality.
        """
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        
        # Generate random matrix
        # We generate a square matrix of the larger dimension to ensure we can orthogonalize properly
        # then slice it to the desired output_dim
        max_dim = max(input_dim, output_dim)
        R = torch.randn(max_dim, max_dim, device=device, generator=g)
        
        # QR Decomposition for orthogonality
        # Q is orthogonal, R is upper triangular
        Q, _ = torch.linalg.qr(R)
        
        # Slice to get our projection matrix (Input -> Output)
        # shape: [input_dim, output_dim]
        P = Q[:input_dim, :output_dim]
        
        return P

    def protect(self, embedding: torch.Tensor, key: str) -> torch.Tensor:
        """
        Apply protection pipeline to a batch of embeddings.
        
        Args:
            embedding: Raw features [Batch, Dim]
            key: Secret user key/PIN
            
        Returns:
            Protected embedding [Batch, Dim] (Same size as input for compatibility)
        """
        seed = self._str_to_seed(key)
        device = embedding.device
        batch_size, input_dim = embedding.shape
        
        # Ensure input is normalized for consistent chaff scaling
        embedding = F.normalize(embedding, p=2, dim=1)
        
        # 1. Chaffing
        # Generate chaff and normalize it to a small magnitude
        # If chaff is too large, it dominates the cosine similarity (Impostor Score -> 1.0)
        # We set chaff norm to 0.1 relative to feature norm (1.0)
        # Resulting Impostor Similarity ~= 0.1^2 / (1 + 0.1^2) ~= 0.01
        chaff = self._generate_chaff(seed, device)
        chaff = F.normalize(chaff, p=2, dim=1) * 0.1
        chaff = chaff.expand(batch_size, -1)
        
        # Concat: [Features || Chaff]
        chaffed_embedding = torch.cat([embedding, chaff], dim=1)
        
        # 2. Orthogonal BioHashing (Projection)
        # We project back to the original input_dim to maintain compatibility with the system
        # (or we could project to a different size, but keeping it same is easier for current pipeline)
        # Total input size = input_dim + chaff_dim
        total_dim = input_dim + self.chaff_dim
        
        projection_matrix = self._get_projection_matrix(total_dim, input_dim, seed, device)
        
        # Project: [Batch, Total] x [Total, Input] -> [Batch, Input]
        protected = torch.matmul(chaffed_embedding, projection_matrix)
        
        # 3. Normalization (Critical for Cosine Similarity)
        protected = F.normalize(protected, p=2, dim=1)
        
        return protected
