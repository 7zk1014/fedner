import torch

def svd_compress_tensor(tensor: torch.Tensor, topk: int) -> torch.Tensor:
    """
    Apply low-rank approximation to 2D tensor, keeping only topk singular values.
    For 1D tensors or when min(dim) < topk, return original tensor.
    """
    if tensor.ndim != 2 or min(tensor.shape) < topk:
        return tensor
    # Singular Value Decomposition
    U, S, V = torch.svd(tensor)
    U_k = U[:, :topk]           # (m, k)
    S_k = S[:topk]              # (k,)
    V_k = V[:, :topk]           # (n, k)
    return U_k @ torch.diag(S_k) @ V_k.t()


def compress_state_dict_diff(global_sd: dict, student_sd: dict,
                             topk: int, min_dim: int):
    """
    Calculate parameter differences, compress using SVD, and reconstruct compressed state dict:
    1) Calculate diff = student_sd - global_sd
    2) Apply svd_compress_tensor to each diff tensor
    3) Return compressed_student_sd = global_sd + compressed_diff
    """
    compressed_sd = {}
    for k in global_sd:
        g = global_sd[k].to(student_sd[k].device)
        s = student_sd[k]
        diff = s - g
        # Only apply SVD to 2D tensors that are large enough
        if diff.ndim == 2 and min(diff.shape) >= min_dim:
            diff = svd_compress_tensor(diff, topk)
        compressed_sd[k] = (g + diff).cpu()
    return compressed_sd
