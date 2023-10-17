import torch
import torch.nn as nn

import utils


class EMModule(nn.Module):
    """
    An Epsiodic Memory module that can be used as a sub-component of other models.

    The EM module is a key-value memory that stores a set of keys and values.
    When queried with a key, it returns a weighted sum of the values, where the weights
    are determined by the similarity between the query key and the stored keys.
    """
    def __init__(self, temperature, normalize_keys=True, weighted_retrieval=False) -> None:
        super().__init__()
        self.state_keys = None
        self.context_keys = None
        self.values = None
        self.encode_context = True
        self.temperature = temperature
        self.normalize_keys = normalize_keys
        self.state_weight = nn.Parameter(torch.zeros(1))
        self.state_weight.requires_grad = weighted_retrieval
        
        # Store most recently calculated match weights for logging
        self.state_match_weights_ = None
        self.context_match_weights_ = None

    
    def norm_key(self, key: torch.tensor) -> torch.tensor:
        """
        Normalize the provided key to unit length.

        Args:
            key: the key to normalize.
        """
        if self.normalize_keys:
            return key/key.norm(dim=-1,keepdim=True)
        else:
            return key

    
    def get_match_weights(self, state: torch.tensor, context: torch.tensor) -> torch.tensor:
        """
        Get the weights indicating the similarity between the provided state and context and the stored keys.

        Args:
            state: the state to compare to the stored keys.
            context: the context to compare to the stored keys.
        
        Returns:
            The weights indicating the similarity between the provided state and context and the stored keys,
            with one weight per key stored in memory.
        """
        if not self.encode_context:
            state = torch.cat([state,context],axis=-1)
        state = self.norm_key(state)
        if len(state.shape)==1:
            state = state.unsqueeze(0)
        self.state_match_weights_ = torch.einsum('b a, c a -> c b',self.state_keys,state)/self.temperature
        if not self.encode_context:
            return self.state_match_weights_
        
        context = self.norm_key(context)
        if len(context.shape)==1:
            context = context.unsqueeze(0)
        self.context_match_weights_ = torch.einsum('b a, c a -> c b',self.context_keys,context)/self.temperature
        return torch.sigmoid(self.state_weight)*self.state_match_weights_ + torch.sigmoid(-self.state_weight)*self.context_match_weights_


    def forward(self, state: torch.tensor, context: torch.tensor) -> torch.tensor:
        self.match_weights_ = self.get_match_weights(state, context)
        return torch.einsum('a b, c a -> c b',self.values,utils.safe_softmax(self.match_weights_,dim=-1))
    

    def write(self, state_key, context_key, value):
        state_key, context_key = self.norm_key(state_key), self.norm_key(context_key)
        if self.state_keys is None:
            self.state_keys = state_key
        else:
            self.state_keys = torch.cat((self.state_keys,state_key),dim=0)
        if self.context_keys is None:
            self.context_keys = context_key
        else:
            self.context_keys = torch.cat((self.context_keys,context_key),dim=0)
        if self.values is None:
            self.values = value
        else:
            self.values = torch.cat((self.values,value),dim=0)


class RecurrentContextModule(nn.Module):
    """
    An Recurrent Neural Network module based on an architecture similar to the minimally gated recurrent unit.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs) -> None:
        super().__init__()
        self.state_to_hidden = nn.Linear(n_inputs,n_hidden)
        self.hidden_to_hidden = nn.Linear(n_hidden,n_hidden)
        self.state_to_hidden_wt = nn.Linear(n_inputs,n_hidden)
        self.hidden_to_hidden_wt = nn.Linear(n_hidden,n_hidden)
        self.hidden_to_context = nn.Linear(n_hidden,n_outputs)

        self.n_hidden_units = n_hidden
        self.hidden_state = torch.zeros((self.n_hidden_units,),dtype=torch.float)
        self.update_hidden_state = True

    def forward(self, x: torch.tensor) -> torch.tensor:
        h_prev = self.hidden_state
        h_update = torch.tanh(self.state_to_hidden(x)+self.hidden_to_hidden(h_prev))
        h_weight = torch.sigmoid(self.state_to_hidden_wt(x)+self.hidden_to_hidden_wt(h_prev))
        h_new = h_weight*h_prev + (1-h_weight)*h_update
        if self.update_hidden_state:
            self.hidden_state = h_new.detach().clone()
        return self.hidden_to_context(h_new)


    def get_terminal_state(self, x: torch.tensor, max_iterations: int = 20) -> torch.tensor:
        last_hidden_state = self.hidden_state.detach().clone()
        for i in range(max_iterations):
            output = self.forward(x)
            if torch.allclose(self.hidden_state,last_hidden_state):
                break
            last_hidden_state = self.hidden_state.detach().clone()
        return output