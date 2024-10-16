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

(:drop_rate GPT-CONFIG-124M)


