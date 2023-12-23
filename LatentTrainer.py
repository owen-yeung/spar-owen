from VAEWrapper import WrappedVAE
import torch
from HiddenProcessing import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from transformers import Trainer, TrainingArguments


class LatentTrainer(Trainer):
    def __init__(
        self, wrapped_vae: WrappedVAE, training_args, latent_module, logging_steps=1e2,
        # target_text=None,
        target_dir=None,
        generation_length=10,
        # tokenizer=wrapped_vae.tokenizer,
        # length_reg=0.0,
    ):
        """_summary_
        Example use case: target_dir is hidden of last token of shaq
        loss is 1 - cosine_similarity(hidden - target)
        Loss transposed to be min at 0, max at 2

        Args:
            optimus (Optimus): _description_
            target_dir (_type_): _description_
            training_args (_type_): _description_
            latent (_type_): _description_
        """        
        '''
        decoder: decoder model from an Optimus VAE, must have output_hidden_states=True
        target_dir: of size (1 + num_layers, 1, 1, latent_size)
        '''
        # self.targeting_text = target_text != None
        # self.targeting_dir = target_dir != None

        # if self.targeting_text and self.targeting_dir:
        #     raise ValueError('Can only target text or activation direction, not both')

        # self.target_text = target_text
        self.target_dir = target_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent_module.to(self.device)
        super().__init__(model=latent_module, args=training_args)
        self.wrapped_vae = wrapped_vae # make sure this outputs hidden
        # if self.targeting_dir:
        self.target_dir = target_dir.to(self.device).reshape(-1) #flattened
        # assert target_dir.shape == 
        # context = optimus.model_vae.tokenizer_decoder.encode('<BOS>')
        # context = torch.tensor(context, dtype=torch.long, device=self.device)
        # self.tokenizer = tokenizer

        context = ""
        tokenizer_output = self.wrapped_vae.tokenizer(context, return_tensors="pt")
        self.context_ids = tokenizer_output.input_ids.to(self.device)
        self.context_mask = tokenizer_output.attention_mask.to(self.device)
        
        self.loss_values = []
        self.logging_steps = logging_steps
        # self.length_reg = length_reg
        self.generation_length = generation_length

    def compute_loss(self, latent_module, return_cosim=False, return_text=True
                     #TODO maybe change cosim to compute layerwise so we know where the cosim loss is coming from
        # return_dir=False,
        ): 
        '''
        latent is a trainable parameter/model with shape [1, latent_size]
        '''
        #1. Extract params from latent model
        latent = latent_module.get_parameter('latent').clone().requires_grad_(True)
        # inputs = {'input_ids': self.context, 'past': past}
        
        #2. Put latent through vae decoder and Get hidden state
        # hidden_states = self.decoder(**inputs)[2] 
        
        # if self.targeting_dir:
        out = self.wrapped_vae.ldlm_model.generate(latent, self.context_ids, self.context_mask, 48, output_hidden_states=True)

        output_ids = out['output_ids']
        hidden_states_tuple = out['hidden_states']
        current_dir = process_raw_hidden(hidden_states_tuple, layer_index=-1, last_token_only=True, ).reshape(-1)
        
        # hidden_states_last_token = Optimus.extract_last_token(hidden_states_tuple)
        # hidden_states_last_token = torch.stack(hidden_states_last_token) # hidden_states reformated from tuple of tensors to single tensor
        # print(f'hidden_states_last_token shape: {hidden_states_last_token.shape}')
        # hidden_states_last_token = hidden_states_all_tokens[..., -1, :]
        # current_dir = hidden_states_last_token.view(-1) #flatten hidden layers into 1d vector
        if current_dir.shape != self.target_dir.shape:
            print(f'Target shape is {self.target_dir.shape} but current_dir shape is {current_dir.shape}')
            raise ValueError
        #3. Compute loss with cosine similarity between hidden states
        similarity = F.cosine_similarity(current_dir, self.target_dir, dim=0)
        loss = 1 - similarity # + self.length_reg * output_length ** 2
        # assert loss.numel() == 1, "Loss must be a scalar"
        # assert loss <= 2 and loss >= 0
        return_dict = {'loss': loss}
        if return_cosim:
            return_dict['cos_similarity'] = similarity
        if return_text:
            return_dict['output_ids'] = output_ids
        return return_dict
    
    

    def train(self, optimizer_state=None
            #   resume_checkpoint=False,
              ):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if optimizer_state != None:
            optimizer.load_state_dict(optimizer_state)
        num_epochs = int(self.args.num_train_epochs)
        latent_module = self.model
        

        for epoch in range(num_epochs):
            # for step, batch in enumerate(self.get_train_dataloader()):
            optimizer.zero_grad()
            
            step_outputs = self.compute_loss(latent_module)
            loss = step_outputs['loss']
            loss.backward(
                retain_graph=True
                )
            
            # latent = latent_module.get_parameter('latent')
            # if torch.all(latent.grad == 0):
            #     print('Latent grad is 0')

            # for name, param in self.model.named_parameters():
            #     print(f'Before step: {name}, {param.data}')

            optimizer.step()

            # After optimizer.step()
            # for name, param in self.model.named_parameters():
            #     print(f'After step: {name}, {param.data}')
            loss_scalar = loss.item()
            self.loss_values.append(loss_scalar)
            #TODO save logs somewhere
            if epoch % self.logging_steps == 0:
                print(f"Epoch {epoch}")
                # self.optimus.print_greedy(latent)
                text = self.wrapped_vae.ids_to_text(step_outputs['output_ids'], self.generation_length, self.tokenizer)
                # print(text)
                # print(f'Loss = 1 - cosine_similarity = {loss_scalar}')
                wandb.log({"loss": loss_scalar, "text": text, "cos_similarity": 1 - loss_scalar})
                # print(text)
            # print(f"Epoch {epoch}, Loss: {loss.item()}")
        plt.plot(self.loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss = 1 - Cos Similarity')
        plt.show()
        training_state = {
            'epoch': num_epochs,
            'model_state_dict': latent_module.state_dict(), # save the state of the latent
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_scalar
            }
        return training_state
