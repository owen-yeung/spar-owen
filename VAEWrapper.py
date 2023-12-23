from HiddenProcessing import *
# import LDLM
from LDLM import *
import torch

class WrappedVAE:
    """
    Wrapper for VAE model that handles tokenization and decoding
    Helper methods go here
    Allows changing methods without reloading model
    """
    def __init__(self, ldlm_model, tokenizer, device, dtype=torch.bfloat16):
        self.ldlm_model = ldlm_model.to(device)
        self.tokenizer = tokenizer
        self.device = device # Only model goes to device
        self.dtype = dtype
        
    def reconstruct(self, input_text, generation_length=48):
        with torch.cuda.amp.autocast(dtype=self.dtype):
            tokenizer_output = self.tokenizer(input_text, return_tensors="pt")
            input_ids = tokenizer_output.input_ids.to(self.device)
            attention_mask = tokenizer_output.attention_mask.to(self.device)
            input_latent = self.ldlm_model.encode(input_ids, attention_mask)
            output_ids = self.ldlm_model.generate(input_latent, input_ids, attention_mask, generation_length, output_hidden_states=False)
 
            text = self.ids_to_text(output_ids, generation_length)
            return {'output_ids': output_ids, 'generation_length': generation_length, 'text': text}
        
    def text_to_latent(self, text,):
        tokenizer_out = self.tokenizer(text, return_tensors="pt")
        tokens = tokenizer_out.input_ids
        mask = tokenizer_out.attention_mask
        # tokens, mask = accelerator.prepare(tokens, mask)
        tokens, mask = tokens.to(self.device), mask.to(self.device)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            return self.ldlm_model.encode(tokens, mask)
        
    
    def get_hidden_states(self, prompt, layer_index=-1, last_token_only=True,):
        """_summary_

        Args:
            prompt (_type_): _description_
            model (_type_): _description_
            tokenizer (_type_): _description_
            last_token_only (bool, optional): _description_. Defaults to True.
            no_embedding (bool, optional): _description_. Defaults to True.
            as_tuple (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor of size (n_layers, batch_size, hidden_size)
        """    
        get_hidden_states(self.ldlm_model, self.tokenizer, self.device, prompt, dtype=self.dtype, layer_index=layer_index, last_token_only=last_token_only, )

    def ids_to_text(self, ids, generation_length=48,):
        out_texts = []
        output_ids = ids[0][-generation_length:].unsqueeze(0)
        out_texts += [self.tokenizer.decode(toks, skip_special_tokens=False) for toks in output_ids]
        return ' | '.join(out_texts)
# initialize model
        

# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs")
#     model_vae = nn.DataParallel(model_vae)

# model_vae = accelerator.prepare(model_vae)


