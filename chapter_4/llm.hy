; The core of a GPT model includes the transformer blocks
; in generative pretrained transformer (GPT) style architectures, parameters refers to the trainable weights of the model, which are adjusted and optimized during training to minimize a specific loss function -> which allows the model to learn from the training data


; Configuration for a small GPT-2 model 
(setv GPT-CONFIG-124M {
                       "vocab_size" 50257 ; words used by the BPE tokenizer 
                       "context_length" 1024 ; max number of input tokens the model can handle via the positional embeddings
                       "emb_dim" 768 ; each token is transformed into a 768 dimensional vector
                       "n_heads" 12 ; the count of attention heads in the multi-head attention mechanisms
                       "n_layers" 12 ; the number of transformer blocks in the model
                       "drop_rate" 0.1 ; intensity of the dropout mechanism (10%) - random drop out of hidden units to prevent overfitting
                       "qkv_bias" False ; include bias vector in linear layers of multihead attention for query, key, and value computations
                       })

(import tiktoken)
(import torch)
(import chapter_4.dummy_gpt_model :as DGPT)

; this results in an error?
; https://github.com/openai/tiktoken/issues/218
; (setv tokenizer (tiktoken.get_encoding "gpt-2"))
(setv tokenizer (tiktoken.get_encoding "o200k_base"))
(print tokenizer)

(setv batch [])
(setv txt1 "Every effort moves you")
(setv txt2 "Every day holds a")

(.append batch (torch.tensor (tokenizer.encode txt1)))
(.append batch (torch.tensor (tokenizer.encode txt2)))
(setv batch (torch.stack batch :dim 0))
(print batch)

(torch.manual_seed 123)

; initalize 124 million parameter DummyGPTModel and feed it the tokenized batch
(setv model (DGPT.DummyGPTModel GPT_CONFIG_124M))
(setv logits (model batch))
(print "output shape: " (.size logits)) ;; expect -> torch.Size ([2, 4, 50257])
(print logits)                                
                                        


; Noticed something interesting, if there are conflicting issues, first try deleting the generated CPython file or the entire __pycache__ directory because it can cause hard to debug issues (ex. a class that should have a specific property, keeps coming up as not found), the files will be re-generated as necessary so it should not cause issues deleting them and restarting the Hy REPL


