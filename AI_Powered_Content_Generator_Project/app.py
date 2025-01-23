# Sample Flask app code
from flask import Flask, jsonify, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

@app.route('/generate_text', methods=['POST'])
def generate_text():
    user_input = request.json['prompt']
    inputs = gpt2_tokenizer(user_input, return_tensors="pt")
    outputs = gpt2_model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)