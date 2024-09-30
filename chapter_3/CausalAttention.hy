; (import torch.nn :as nn)

; (defclass CausalAttention [nn.Module]
;   (defn __init__ [self d-in d-out context-length dropout [qkv-bias False]]
;     (.__init__ (super))
;     (setv self.d-out d-out)
;     (setv self.w-query (nn.Linear d-in d-out :bias qkv-bias))
;     (setv self.w-key (nn.Linear d-in d-out :bias qkv-bias))
;     (setv self.w-value (nn.Linear d-in d-out :bias qkv-bias))
;     (setv self.dropout (nn.Dropout dropout))
;     (setv self.register-buffer ['mask'
;                                 (torch.triu (torch.ones context-length context-length :diagonal 1))]))

;   (defn forward [self x]
;     (setv b (.size x))
;     (setv num-tokens (.size x))
;     (setv d-in (.size x))
;     (setv keys (self.w-key x))
;     (setv queries (self.w-query x))
;     (setv values (self.w-value x))

;     (setv attn-scores (@ queries (.transpose keys 1 2)))
    
;     ;; might have to revisit this on page 81 -- this is supposed to happen in place, struggling to figure out how to do this 'in place'
;     ; (.masked_fill_ attn-scores mask [num-tokens num-tokens] (* torch.inf -1))
;     (setv mask (torch.triu (torch.ones context-length context-length) :diagonal 1))
;     (setv masked (.masked_fill attn-scores (mask.bool) (* torch.inf -1)))
    
;     (setv attn-weights (torch.softmax (/ attn-scores (** (.size keys -1) 0.5)) :dim 1))
;     (setv attn-weights (self.dropout attn-weights))
;     (setv context-vec (@ attn-weights values))
;     context-vec))


; ;; testing it out
; (setv batch (torch.stack #(inputs inputs) :dim 0))
; (print batch)

; (torch.manual_seed 123)
; (setv context-length (.size batch 1))
; (print context-length)
; (print (.size batch))

; (setv ca (CausalAttention d-in d-out context-length 0.0))
; (setv context-vecs (ca batch))
; (print (.size context-vecs))


; ;; Extending single head attention to multihead attention
; ;; now we will add multiple heads (multi-head attention)
; ;; multi-head means dividing the attention mechanism into multiple heads each operating independently
; ;; a single causal attention module can be considered single head attention where there is only one set of attention weigths processing the input sequentially
; ;; implementing multihead attention involves creating multiple instances of self attention mechanism each with its own weights and then combining their outputs using multiple instances of the self attention mechanism can be computationally intensive -- but this pattern is used in transformer based llms
; ;; they are typically ran in parallel with different learned linear projections results of multiplying the input data like the query key and value vectors in attention mechanisms by a weight matrix. This can be achieved by using a wrapper class that stacks multiple instances of the causualattention module
; ;; 



; Exercising the existing Python implementation of causal attention
(import torch)
(import chapter_3.causal_attention :as CA)

(setv inputs (torch.tensor
              [[0.43 0.15 0.89]
               [0.55 0.87 0.66]
               [0.57 0.85 0.64]
               [0.22 0.58 0.33]
               [0.77 0.25 0.10]
               [0.05 0.80 0.55]]))

(setv d-in (get (.size inputs) 1))
(print d-in)
(setv d-out 2)

(torch.manual_seed 123)
(setv batch (torch.stack #(inputs inputs) :dim 0))
(print batch)
(setv context-length (.size batch 1))
(print context-length)

(setv ca (CA.CausalAttention d-in d-out context-length 0.0))

(setv context-vecs (ca batch))
(print context-vecs)
(print "context-vecs .size ==" (.size context-vecs))