from transformers import PegasusTokenizer, PegasusModel
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
model = PegasusModel.from_pretrained("google/pegasus-large")
input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
last_hidden_states = outputs.last_hidden_state