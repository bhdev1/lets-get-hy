(import torch.nn :as nn)

(defclass SelfAttentionV2 [nn.Module]
  (defn __init__ [self d-in d-out [qkv-bias False]]
    (.__init__ (super))
    (setv self.w-query (nn.Linear d-in d-out :bias qkv-bias))
    (setv self.w-key (nn.Linear d-in d-out :bias qkv-bias))
    (setv self.w-value (nn.Linear d-in d-out :bias qkv-bias)))

  (defn forward [self x]
    (setv keys (self.w-key x))
    (setv queries (self.w-query x))
    (setv values (self.w-value x))
    (setv attn-scores (@ queries keys.T))
    (setv attn-weights (torch.softmax (/ attn-scores (** d-k 0.5)) :dim -1))

    (setv context-vec (@ attn-weights values))

    context-vec))


(torch.manual_seed 789)
(setv sa-v2 (SelfAttentionV2 d-in d-out))
(print (sa-v2 inputs))
;; tensor([[-0.0739,  0.0713],
        ; [-0.0748,  0.0703],
        ; [-0.0749,  0.0702],
        ; [-0.0760,  0.0685],
        ; [-0.0763,  0.0679],
        ; [-0.0754,  0.0693]], grad_fn=<MmBackward0>)


;; Hiding future words with causal attention
;; for many llm tasks you will want the self attention mechanism to consider only the tokens that appear prior to the current position when predicting the next token in a sequence. Causal attention also known as masked attention is a specialized form of self attention. It restricts a model to only consider previous and current inputs in a sequence when processing any given token when computing attention scores. This is in contrast to the standard self-attention mechanism which allows access to the entire input sequenece at once.
;; to achieve this in GPT like llms for each token processed we mask out the future tokens which come after the current token in the input text. You mask out the attention weights above the diagonal and we normalize the nonmasked attention weights such that the attention weights sum to 1 in each row. 

;; in  the first step compute the attention weights using the softmax function:

(setv queries (.w-query sa-v2 inputs))
(setv keys (.w-key sa-v2 inputs))
(setv attn-scores (@ queries keys.T))
(setv attn-weights (torch.softmax (/ attn-scores (** (.size keys -1) 0.5)) :dim -1))
(print attn-weights)

;; implement the second step using PyTorch tril function to create a mask where the values above the diagonal are zero

(setv context-length (.size attn_scores 0))
(setv mask-simple (torch.tril (torch.ones context-length context-length)))
(print mask-simple)

;; multiply this mask with the attention weights to zero out the values above the diagonal

(setv masked-simple (* attn-weights mask-simple))
(print masked-simple)

;; next step, re-normalize the attention weigths to sum up to 1 in each row, do this by dividing each element in each row by the sum in each row

(setv row-sums (.sum masked-simple :dim -1  :keepdim True))
(print row-sums)
(setv masked-simple-norm (/ masked-simple row-sums))
(print masked-simple-norm)

 ;; to make it more efficient, we can mask the attention scores with negative infinity values before applying the softmax function
 ;; when negative infinity values are present in a row, the softmax function treats them as zero probability.
 ;; This can be implemented by creating a mask with 1s above the diagonal and then replacing these 1s with negative infinity values
 
(setv mask (torch.triu (torch.ones context-length context-length) :diagonal 1))
(print mask)

(setv masked (.masked_fill attn-scores (mask.bool) (* torch.inf -1)))
(print masked)

;; finally, just need to apply the softmax function to these masked results

(setv attn-weights (torch.softmax (/ masked (** (.size keys -1) 0.5)) :dim 1))
(print attn-weights)

;; now you can use the modified attention weights to compute the context vectors
(setv context-vec (@ attn-weights values))
(print context-vec)


;; Minor tweak to the causal attention mechanism that is useful for reducing overfitting when training LLMs
;; masking additional attention weights with "dropout"
;; dropout - deep learning technique where randomly selected hidden layer units are ignored during training basically dropping them out
;; this method helps prevent overfitting by ensuring that a model does not become overly reliant on any specific set of hidden layer units
;; it is ONLY used during draining and is disabled afterwards
;; in transformer architecture (as in GPT) dropout in the attention mechanism is typically applied at two specific times after calculating the attention weights or after applying the attention weights to the value vectors
;; in the following example the dropout rate is 50% which means masking out half the attention weights, when used in actual implementation use a lower dropout rate such as 0.1-0.2% this is just for illustrative purposes
(torch.manual_seed 123)
(setv dropout (torch.nn.Dropout 0.5))
(setv example (torch.ones 6 6))
(print (dropout example))

;; applying the dropout to the attention weight matrix
(torch.manual_seed 123)
(print (dropout attn-weights))

(print "to see all of the before techniques applied into the SelfAttention, go to CausalAttention.hy")