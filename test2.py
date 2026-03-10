print("Tok A type:", type(tok_a).__name__)
print("Tok B type:", type(scorer.tokenizer).__name__)

print("\nTok A vocab size:", tok_a.vocab_size)
print("Tok B vocab size:", scorer.tokenizer.vocab_size)

# Check the actual token IDs for one text
ids_a = enc_a['input_ids'][0][:15].tolist()
ids_b = enc_b['input_ids'][0][:15].tolist()
print(f"\nFirst 15 token IDs A: {ids_a}")
print(f"First 15 token IDs B: {ids_b}")

# Decode back to see what each tokenizer produced
print(f"\nDecoded A: {tok_a.convert_ids_to_tokens(ids_a)}")
print(f"Decoded B: {scorer.tokenizer.convert_ids_to_tokens(ids_b)}")

# Compare tokenizer configs
for attr in ['do_lower_case', 'model_max_length', 'padding_side', 
             'vocab_file', 'sp_model_kwargs', 'split_special_tokens']:
    val_a = getattr(tok_a, attr, 'N/A')
    val_b = getattr(scorer.tokenizer, attr, 'N/A')
    match = "✓" if val_a == val_b else "✗ DIFFERENT"
    print(f"  {attr}: A={val_a}, B={val_b} {match}")
