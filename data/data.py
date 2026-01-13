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
    
    def ffn_data(self, tokenizer, embed):
        x = [] # List of pytorch tensors
        y = [] # List of token ids

        if self.data == "":
            self.load_data()

        with open(embed.json_table_path, "r", encoding="utf-8") as f:
            tmp = json.load(f)

        print(tmp)
        input_data = tmp[0].get("input_data", {})

        # On récupère les valeurs en retirant juste les doublons consécutifs
        tokenized_data = []
        prev = None
        for v in input_data.values():
            if v != prev:
                tokenized_data.append(v)
                prev = v

        for token in tokenized_data:
            x.append(embed.token_to_vector(token))
        
        for token in tokenized_data[1:]:
            y.append(tokenizer.detokenize(token))

        return (x, y)
