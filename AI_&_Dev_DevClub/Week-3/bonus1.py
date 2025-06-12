# bonus_ai_generator.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class AITweetGenerator:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_ai_tweet(self, prompt, max_length=60):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=50,
                temperature=0.1,      
                do_sample=False, 
                top_k=50,   
                repetition_penalty = 1.2,
                top_p=0.9, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        tweet = generated_text[len(prompt):].strip()
        return tweet[:280]  # Twitter limit

