import torch
import numpy as np
import traceback
from sys import argv
import json
from scripts.time_log import time_log_module as tlm
from scripts.logger import logger
from data.data import data
from model.model import tokenizer, embedding, SPE, attention_head, FFN # import LAM blocks
#import winsound

# Configuration
with open("config.json", "r") as f:
    config = json.load(f)
webhook_url = config.get("webhook_url", "") # leave empty to disable webhook logging
model_path = config.get("model_path", "model/model.pth")
data_path = config.get("data_path", "data/tiny_sheakespeare.txt") # Tiny Shakespeare Dataset by default
json_data_path = config.get("json_data_path", "data/dataset.json")
version = config.get("version", "None")
dataset_loading_size = config.get("dataset_loading_size", 10000)
tokenizer_config = config.get("tokenizer", {})
embedding_config = config.get("embedding", {})
attention_config = config.get("attention", {})
ffn_config = config.get("ffn", {})
del config

# Args
train = "--train" in argv or "-t" in argv
download = "--download" in argv or "-d" in argv
predict = "--predict" in argv
test_predict = "--test-predict" in argv # Only WITH the train arg
chat = "--chat" in argv 
tokenizer_test = "--tokenizer-test" in argv # Good
embedding_test = "--embedding-test" in argv # Good
force_cpu = "--cpu" in argv # Good
force_cuda = "--cuda" in argv # Good
load_from_file = "--path" in argv or "-p" in argv

if __name__ == "__main__":
    print(f"{tlm()} Start of program.")
    logger = logger(discord_webhook=webhook_url) # créer le logger

    # Logging system info
    logger.log(f"Micro Generative Pre-trained Lam test arch - V{version}.", v=True, Wh=True, mention=False)
    logger.log(f"To change any setting, go check config.json.", v=True, Wh=True, mention=False)
    logger.log(f"PyTorch version: {torch.__version__}", v=True, Wh=True, mention=False)
    logger.log(f"CUDA status : {str(torch.cuda.is_available())}", v=True, Wh=True, mention=False)
    logger.log(f"Script args : {argv}", v=False, Wh=True, mention=False)
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        msg = f"{count} GPU{'s' if count > 1 else ''} detected."
        logger.log(msg, v=True, Wh=True, mention=False)
        for i in range(count):
            logger.log(f" -> Device {i}: {torch.cuda.get_device_name(i)}", v=True, Wh=True, mention=False)
    
    # Select device
    if force_cuda:
        if not torch.cuda.is_available(): # Si pas de GPU
            logger.log("CUDA forced but no GPU detected. Exiting.", v=False, Wh=True, mention=True)
            raise EnvironmentError(f"{tlm()} CUDA forced but no GPU detected. Exiting.")
        if force_cpu: # Si les deux sont forcés
            logger.log("Both --force-cuda and --force-cpu flags detected. Please choose only one.", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Both --force-cuda and --force-cpu flags detected. Please choose only one.")
        logger.log("CUDA forced. Using GPU for computations.", v=True, Wh=True, mention=False)
        device = torch.device("cuda")
    elif force_cpu:
        logger.log("CPU forced. Using CPU for computations.", v=True, Wh=True, mention=False)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.log(f"Using device: {device}.", v=True, Wh=True, mention=False)

    # Var reset
    tk = None
    embed = None
    
    try:
        if train:
            # Load data
            logger.log(f"Loading data from {data_path}...", v=True, Wh=True, mention=False)
            data_loader = data(logger, data_path, dataset_loading_size)
            dataset = data_loader.load_data()
            del data_loader

            # Initialize tokenizer
            logger.log("Initializing tokenizer...", v=True, Wh=True, mention=False)
            tk = tokenizer(logger, tokenizer_config)
            if not tk.vocab_status():
                logger.log("No existing tokenizer found. Creating new vocabulary...", v=True, Wh=True, mention=False)
                tk.create_vocab(dataset)
                tk.save_vocab()
            else:
                logger.log("Existing tokenizer found. Loading vocabulary...", v=True, Wh=True, mention=False)
                tk.load_vocab()

            # Initialize embedding
            embed = embedding(logger, device, tk, embedding_config)
            if embed.check_saved_embedding_table():
                logger.log("Existing embedding table found. Loading embedding table...", v=True, Wh=True, mention=False)
                embed.load_embedding_table()
            else:
                logger.log("No existing embedding table found. Creating new embedding table...", v=True, Wh=True, mention=False)
                embed.create_embedding_model()
                try:
                    embed.train_embedding_model(dataset, json_data_path)
                    embed.save_embedding_table()
                except MemoryError:
                    logger.log("Error during embedding model training: Out of memory. Please try to make your dataset smaller or use a smaller model.", v=False, Wh=True, mention=True) # Getting memory error on TinySheakespeare with 16gb RAM - Crazy bad optimization but im too lazy
                    raise MemoryError(f"{tlm()} Error during embedding model training: Out of memory. Please try to make your dataset smaller or use a smaller model.")

            del dataset # Free memory

            logger.log("Building SPE class...", v=True, Wh=True, mention=False)
            spe = SPE(device)

            #attention_heads = []
            #for i in range(attention_config.get("num_heads", 1)):
            #    head = attention_head(logger, device, embed, attention_config)
            #    attention_heads.append(head)
            #del head, embed, tk

            # For the moment let's just test it with a single attention head
            head = attention_head(logger, embed, SPE, attention_config)
            if not head.check_matrices():
                logger.log("No existing attention matrices found. Trainning new matrices...", v=True, Wh=True, mention=False)
                head.train_matrices(epochs=attention_config.get("num_epochs", 10), lr=attention_config.get("learning_rate", 1e-3))
                head.save_matrices()
            else:
                logger.log("Existing attention matrices found. Loading matrices...", v=True, Wh=True, mention=False)
                head.get_wq()
                head.get_wk()
                head.get_wv()
            

            # ffn
            ffn = FFN(logger, embed, ffn_config)
            if not ffn.model_status():
                logger.log("No existing FFN model found. Training new model...", v=True, Wh=True, mention=False)
                data = data(logger, data_path, dataset_loading_size)
                x,y = data.ffn_data(tk, embed)
                # Prepare training data
                ffn.train_ffn(x, y)
                ffn.save_model()
            else:
                logger.log("Existing FFN model found. Loading model...", v=True, Wh=True, mention=False)
                ffn.load_model()

            if test_predict:
                with open("data/system_prompt.txt", "r") as f:
                    system_prompt = f.read()
                end_prompt = "<eos>\n<microgpl>"
                prompt = f"{system_prompt}Hey! I am Abgache, how are you?{end_prompt}"
                output = ""
                del system_prompt, end_prompt

                for _ in range(100): # Generate 100 tokens
                    token_ids = tk.encode(prompt) # BLOC 1
                    input_embeddings = embed.token_to_vector(token_ids) # BLOC 2
                    del token_ids
                    spe_embeddings = spe.vector_list2spe_vector_list(input_embeddings) # BLOC 3
                    del input_embeddings
                    # It only outputs the last token's attention output
                    attention_output = head.forward(spe_embeddings) # BLOC 4
                    del spe_embeddings
                    predicted_token_id = ffn.predict(attention_output) # BLOC 5
                    del attention_output
                    token = tk.detokenize([predicted_token_id])
                    output += token
                    prompt += token

                print("Generated Output:")
                print(output)

        if download:
            logger.log("Downloading pre-trained model...", v=True, Wh=True, mention=False)
            auto_choose = input("Do you want to choose the model size or let the program choose the best one for your hardware? (y = choose by yourself / n = let the program choose): ") == "n"
            print("No model is available for download yet. Please train your own model or wait for the main release.")

        if tokenizer_test:
            if tk is None:
                tk = tokenizer(logger, tokenizer_config)
                if not tk.vocab_status():
                    logger.log("No existing tokenizer found. Creating new vocabulary...", v=True, Wh=True, mention=False)
                    sample_data = "This is a sample dataset for tokenizer testing."
                    tk.create_vocab(sample_data)
                    tk.save_vocab()
                else:
                    logger.log("Existing tokenizer found. Loading vocabulary...", v=True, Wh=True, mention=False)
                    tk.load_vocab()

            sentence = "" # init
            while sentence != "exit":
                sentence = input("Enter a sentence to tokenize (type 'exit' to quit): ")
                if sentence == "exit":
                    break
                token_ids = tk.tokenize(sentence)
                print(f"Token IDs: {token_ids}")
                decoded_sentence = tk.detokenize(token_ids)
                print(f"Decoded Sentence: {decoded_sentence}")

        if embedding_test:
            if tk is None:
                tk = tokenizer(logger, tokenizer_config)
                if not tk.vocab_status():
                    logger.log("No existing tokenizer found. Creating new vocabulary...", v=True, Wh=True, mention=False)
                    sample_data = "This is a sample dataset for tokenizer testing."
                    tk.create_vocab(sample_data)
                    tk.save_vocab()
                else:
                    logger.log("Existing tokenizer found. Loading vocabulary...", v=True, Wh=True, mention=False)
                    tk.load_vocab()

            embed = embedding(logger, device, tk, embedding_config)
            if embed.check_saved_embedding_table():
                logger.log("Existing embedding table found. Loading embedding table...", v=True, Wh=True, mention=False)
                embed.load_embedding_table()
            else:
                logger.log("No existing embedding table found. Creating new embedding table...", v=True, Wh=True, mention=False)
                embed.create_embedding_table()
                embed.save_embedding_table()

            w1 = ""
            w2 = ""
            while w1 != "exit":
                w1 = input("Enter the first word (type 'exit' to quit): ")
                if w1 == "exit":
                    break
                w2 = input("Enter the second word (type 'exit' to quit): ")
                op = input("Enter the operation (+ or -): ")
                if op not in ["+", "-"]:
                    print("Invalid operation. Please enter + or -.")
                    continue
                else:
                    # Get each word embedding
                    a = embed.token_to_vector(tk.tokenize(w1))
                    print(type(a), a.shape)
                    b = embed.token_to_vector(tk.tokenize(w2))
                    if op == "+":
                        result_vector = a + b
                        print(f"Resulting vector (first 10 dimensions): {result_vector[:10]} | Length: {len(result_vector)} | New word approximation: {tk.detokenize(embed.vector_to_token(result_vector))}")
                    else:
                        result_vector = a - b
                        print(f"Resulting vector (first 10 dimensions): {result_vector[:10]} | Length: {len(result_vector)} | New word approximation: {tk.detokenize(embed.vector_to_token(result_vector))}")


        if predict:
            pass

        if chat:
            pass
    except:
        logger.log(f"Unknown error occurred : {Exception}. Please check the logs for more details.", v=True, Wh=True, mention=True)
        tb = traceback.format_exc()
        logger.log(tb, v=True, Wh=True, mention=True)
        #winsound.MessageBeep(winsound.MB_ICONASTERISK)
    logger.log(f"End Of program.", v=True, Wh=True, mention=True)