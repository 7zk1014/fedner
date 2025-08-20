import torch

def svd_compress_tensor(tensor: torch.Tensor, topk: int) -> torch.Tensor:
    """
    对二维 tensor 做低秩近似，只保留 topk 个奇异值。
    对于一维或 min(dim) < topk 的 tensor，原样返回。
    """
    if tensor.ndim != 2 or min(tensor.shape) < topk:
        return tensor
    # SVD
    U, S, V = torch.svd(tensor)
    U_k = U[:, :topk]           # (m, k)
    S_k = S[:topk]              # (k,)
    V_k = V[:, :topk]           # (n, k)
    return U_k @ torch.diag(S_k) @ V_k.t()


def compress_state_dict_diff(global_sd: dict, student_sd: dict,
                             topk: int, min_dim: int):
    """
    1) 计算 diff = student_sd - global_sd
    2) 对每个 diff 张量调用 svd_compress_tensor
    3) 返回 compressed_student_sd = global_sd + compressed_diff
    """
    compressed_sd = {}
    for k in global_sd:
        g = global_sd[k].to(student_sd[k].device)
        s = student_sd[k]
        diff = s - g
        # 仅对 2D 且较大的做 SVD
        if diff.ndim == 2 and min(diff.shape) >= min_dim:
            diff = svd_compress_tensor(diff, topk)
        compressed_sd[k] = (g + diff).cpu()
    return compressed_sd
