from scripts.time_log import time_log_module as tlm
import json

class data():
    def __init__(self, logger, data_path, dataset_loading_size):
        self.logger = logger
        self.data_path = data_path
        self.dataset_loading_size = dataset_loading_size
        self.data = ""
    
    def load_data(self):
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.data = f.read(self.dataset_loading_size)
            self.logger.log(f"Data loaded successfully from {self.data_path}.", v=True, Wh=True, mention=False)
        except Exception as e:
            self.logger.log(f"Error loading data from {self.data_path}: {e}", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Error loading data from {self.data_path}: {e}")
        
        return self.data
    
    # Should work but is very unoptimized, will be optimized later, for now I just want to get it working
    # I forgot to add the attention heads -_-
    def ffn_data(self, tokenizer, embed): #error here, Fckd up function, bad optimization, weird data loading -> Imma use the old unoptimized loading data
        x = [] # List of pytorch tensors
        y = [] # List of token ids

        if self.data == "":
            self.load_data()
        
        tokenized_data = tokenizer.tokenize(self.data)
        tmp = "<UNK>"

        # Main unoptimized data loading loop, will be optimized later
        for i, token in enumerate(tokenized_data):
            x.append(embed.token_to_vector(token)) # Get the token vector from the embedding
            if i == 0:
                y.append(3)
            elif i == len(tokenized_data) - 1:
                y.append(4)
            else:
                try:
                    y.append(tmp) # y.append(tokenizer.tokenize(tmp)) # Get the token id from the tokenizer
                except Exception:
                    print(tmp) # If it prints an integer i'll have to remove the tokenizer.tokenize, optimization ig ¯\(0_0)/¯ | I am always right hehehe
                    exit(1)
            tmp = token

        #with open(embed.json_table_path, "r", encoding="utf-8") as f:
        #    tmp = json.load(f)

        #input_data = tmp

        ## On récupère les valeurs en retirant juste les doublons consécutifs
        #tokenized_data = []
        #prev = None
        #for v in input_data:
        #    if v != prev:
        #        tokenized_data.append(v)
        #        prev = v

        #for token in tokenized_data:
        #    x.append(token[1]) # Token[0] = Token ID Token[1] = Token vector 
        #
        #for token in tokenized_data[1:]:
        #    y.append(token[0])
        #
        ## Last token need to be <eos>
        #y.append(4)

        return (x, y)
