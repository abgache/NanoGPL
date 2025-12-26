# NanoGPL Beta 0.1
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red)](https://pytorch.org/)  
Small test generative pre-trainned LAM (Linear Attention Mechanism).
To change any setting, please go check ``config.json``.

## What is LAM?
**LAM** stands for Linear Attention Mechanism, it's a [Transformer](https://arxiv.org/pdf/1706.03762)-like neural network architecture made for NGPL, it's way easier to understand, cheaper in resources but less efficient than the real Transformer arch even thought they are both really similars.
### How does it works?
![shema](LAM - NanoGPL.png)
This shema show how it works from the input word list, to the tokens probabilities. Each bloc is explained down here.
<details>
<summary>1- Tokenizer</summary>
It's the first bloc, it's very usefull for going from a string (e.g. "My name is abgache and i love machine learning!") to numbers (e.g. [45, 58, 563]).
It basically just create a token list (in this case it's ``model/tokenizer.json``but you can change it in the ``config.json`` file) and asign to each token an ID without any real meaning.
It uses the [BPE](https://huggingface.co/learn/llm-course/chapter6/5) (byte pair-encoding) algorithm to divide the stings into tokens.

To test it, use this command: 

```batch
   python main.py --tokenizer-test
```
</details>
<details>
<summary>2- Embeddings</summary>
It's the second bloc, it's very usefull for giving a meaning to token IDs, it give a vector (In this case it has 256 dimensions, but you can change it in the ``config.json`` file) to each token to try to encode it's meaning.
The embeddings bloc outputs a list of vectors, each vector encodes the sense of a token.
At fist we generate them using a neural network, but then we save all tokens vectors in a list to save resources.

To test it, use this command: 

```batch
   python main.py --embedding-test
```
</details>
<details>
<summary>3- SPE</summary>
Soon...
</details>
<details>
<summary>4- Attention head(s)</summary>
Soon...
</details>
<details>
<summary>5- Feed forward neural network</summary>
It's the last bloc, it's a simple neural network that gets the input vector and try to predict the next token.
It's input layer is just 256 neurons (because in this case we have 256 dimensions in our vectors) and an output of the vocabulary size with a Softmax activation function.
</details>

---

## How to use it?
pass

---

> [!WARNING]
> I am __not__ responsible of any missusage for NanoGPL, any false information or anything he gave/told you. It is a test model, use it at your own risks.  
  
  
---

### TODO : 
- [X] Working tokenizer
- [ ] Working embedding
- [ ] Working SPE
- [ ] Working single attention head
- [ ] Working multiple attention heads
- [ ] Working and __trainned__ final feed forward neural network
- [ ] Special file format for GPL models (maybe something like .gpl)
- [ ] Local API
- [ ] WebUI for ngpl
- [ ] Mobile version