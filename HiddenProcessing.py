import torch

# My own helper functions

def process_raw_hidden(hidden_states_tuple,
                       layer_index=-1,
                        last_token_only=True, 
                        # 
                        ):
    """Takes model_output.hidden_states and processes it as desired

    Args:
        hidden_states_tuple (_type_): _description_
        layer_index (int, optional): _description_. Defaults to -1. Layer list includes embedding layer (i.e. has length n_layers + 1)
        - Can also take a slice for range of layers
        last_token_only (bool, optional): _description_. Defaults to True.
        no_embedding (bool, optional): _description_. Defaults to True.
        as_tuple (bool, optional): _description_. Defaults to False.

    Returns:
        torch.Tensor of size (n_layers + include_embedding, batch_size, hidden_size)
    """    
    # if no_embedding:
    
    # if len(hidden_states_tuple) > 1:
    #     hidden_states_tuple = Tuple(hidden_states_tuple,)
    # if as_tuple:
    #     return hidden_states_tuple
    if isinstance(layer_index, slice):
        hidden_states_tuple = hidden_states_tuple[layer_index]
        hidden_states_tensor = torch.stack(hidden_states_tuple)
    elif isinstance(layer_index, int):
        hidden_states_tensor = hidden_states_tuple[layer_index].unsqueeze(0)
        # hidden_states_tensor = hidden_states_tuple[0]
    # hidden_states_tensor = torch.stack(hidden_states_tuple)
    if last_token_only:
        hidden_states_tensor = hidden_states_tensor[:, :, -1, :]
        # hidden_states_tensor.unsqueeze(2)
    return hidden_states_tensor

def get_hidden_states(model, tokenizer, device, prompt, dtype=torch.bfloat16, layer_index=-1, last_token_only=True,):
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
        #TODO check if support multiple prompts
        # prompt = 'Q: What is the largest animal?\nA:'
        tokenizer_output = tokenizer(prompt, return_tensors="pt").to(device)
        # tokenizer_output = accelerator.prepare(tokenizer_output)
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        # print(tokenizer_output)
        with torch.cuda.amp.autocast(dtype=dtype):
            model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states_tuple = model_outputs.hidden_states
        #TODO check if I have to remove any added hidden layers from the adapter
        #TODO check if just disabling adapter works
        return process_raw_hidden(hidden_states_tuple, layer_index=layer_index, last_token_only=last_token_only, )