from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Llama2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggingface/llama2")
model = AutoModelForCausalLM.from_pretrained("huggingface/llama2")

# Tokenize the input text
inputs = tokenizer(text['text'], return_tensors="pt")

# Generate response
response = model.generate(inputs.input_ids, max_length=50)
output_text = tokenizer.decode(response[0], skip_special_tokens=True).split('.')[0:2] # Restrict to 2 sentences
