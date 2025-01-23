from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Model:
    def __init__(self,
                 model_name
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            trust_remote_code=True
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def get_model_answer(self, batch_prompts):
        inputs = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)