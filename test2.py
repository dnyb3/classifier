import sentencepiece as spm

# Check if the sp_model is actually functional in each
print(f"A sp_model type: {type(tok_a.sp_model)}")
print(f"B sp_model type: {type(scorer.tokenizer.sp_model)}")

print(f"\nA sp_model piece size: {tok_a.sp_model.get_piece_size()}")
print(f"B sp_model piece size: {scorer.tokenizer.sp_model.get_piece_size()}")

# Direct SentencePiece test
test = "i recommend rolling over"
print(f"\nA sp_model.encode: {tok_a.sp_model.encode(test, out_type=str)}")
print(f"B sp_model.encode: {scorer.tokenizer.sp_model.encode(test, out_type=str)}")
