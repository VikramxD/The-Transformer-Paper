import torch
from transformer import build_model

def main():
    # Define model parameters (can be customized)
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    src_seq_len = 100  # Max length of source sequences
    tgt_seq_len = 120  # Max length of target sequences
    
    embedding_dim = 512
    num_layers = 6
    num_heads = 8
    dropout_rate = 0.1
    feed_forward_dim = 2048

    print("Building Transformer model...")
    # Build the model using the utility function
    model = build_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=src_seq_len,
        tgt_sequence_len=tgt_seq_len,
        embedding_dimension=embedding_dim,
        n_layer=num_layers,
        num_heads=num_heads,
        dropout=dropout_rate,
        feed_forward_dim=feed_forward_dim
    )
    print("Transformer model built successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 30)

    # --- Example: Dummy Forward Pass ---
    print("Performing a dummy forward pass...")
    batch_size = 2
    current_src_seq_len = 10 # Actual length of source sequences in this batch
    current_tgt_seq_len = 12 # Actual length of target sequences in this batch

    # Create dummy input tensors (random integers representing token IDs)
    dummy_src = torch.randint(0, src_vocab_size, (batch_size, current_src_seq_len))
    dummy_tgt = torch.randint(0, tgt_vocab_size, (batch_size, current_tgt_seq_len))

    # Create dummy masks (simplified for this example)
    # In a real scenario, src_mask would mask padding tokens,
    # and tgt_mask would include a causal mask for self-attention and padding mask.
    # For this example, we'll use None, which means no masking is applied by default
    # in the underlying attention mechanism if not handled explicitly by the model's
    # encode/decode methods or if the attention mechanism itself defaults to no mask.
    # However, the Transformer's encode/decode methods expect masks.
    # A simple source mask (all valid, no padding)
    dummy_src_mask = torch.ones(batch_size, 1, 1, current_src_seq_len, dtype=torch.bool) 
    # A simple target mask (causal + all valid, no padding)
    # This is a simplified causal mask. A proper one would be more complex.
    dummy_tgt_mask = torch.tril(torch.ones(batch_size, 1, current_tgt_seq_len, current_tgt_seq_len, dtype=torch.bool))

    print(f"Dummy source shape: {dummy_src.shape}")
    print(f"Dummy target shape: {dummy_tgt.shape}")
    
    # 1. Encode the source sequence
    # The model.src_embedding and model.src_pos are called within model.encode
    encoder_output = model.encode(src=dummy_src, src_mask=dummy_src_mask)
    print(f"Encoder output shape: {encoder_output.shape}") # Expected: (batch_size, src_seq_len, embedding_dim)

    # 2. Decode the target sequence
    # The model.tgt_embedding and model.tgt_pos are called within model.decode
    decoder_output = model.decode(
        encoder_output=encoder_output,
        src_mask=dummy_src_mask, # Source mask is used in cross-attention
        tgt=dummy_tgt,
        tgt_mask=dummy_tgt_mask  # Target mask for self-attention in decoder
    )
    print(f"Decoder output shape: {decoder_output.shape}") # Expected: (batch_size, tgt_seq_len, embedding_dim)

    # 3. Project to vocabulary
    # The model.linear_layer is a nn.Linear followed by log_softmax
    logits = model.linear_step(decoder_output)
    print(f"Logits shape: {logits.shape}") # Expected: (batch_size, tgt_seq_len, tgt_vocab_size)
    print("-" * 30)
    print("Dummy forward pass completed.")
    print("Note: This example uses random data and basic masks. For actual use,")
    print("you'll need proper data tokenization, embedding, and mask creation.")

if __name__ == "__main__":
    main()
