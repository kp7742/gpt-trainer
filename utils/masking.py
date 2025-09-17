import torch

def create_causal_attention_mask(input_mask: torch.Tensor) -> torch.Tensor:
    """
    Creates a 4D causal attention mask from a 2D attention mask where 'True'
    indicates a position that should be attended to.

    Args:
        input_mask (torch.Tensor): A 2D tensor of shape [batch_size, seq_len]
                                  with 1s for valid tokens and 0s for padding.

    Returns:
        torch.Tensor: A 4D boolean tensor of shape [batch_size, 1, seq_len, seq_len]
                      where 'True' indicates a position to attend to.
    """
    _, seq_len = input_mask.shape
    device = input_mask.device

    # Create the lower-triangular causal mask.
    causal_mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    ) # [seq_len, seq_len]

    # We expand its dimensions to mask out columns corresponding to padding tokens.
    padding_mask = input_mask.bool().unsqueeze(1) # [batch_size, 1, seq_len]

    combined_mask = causal_mask & padding_mask # [batch_size, seq_len, seq_len]

    return combined_mask.unsqueeze(1)
