(import torch.nn :as nn)

(defclass SelfAttentionV1 [nn.Module]
  (defn __init__ [self d-in d-out] 
    (.__init__ (super))
    (setv self.w-query (nn.Parameter (torch.rand d-in d-out)))
    (setv self.w-key (nn.Parameter (torch.rand d-in d-out)))
    (setv self.w-value (nn.Parameter (torch.rand d-in d-out))))

  (defn forward [self x]
    (setv keys (@ x self.w-key))
    (setv queries (@ x self.w-query))
    (setv values (@ x self.w-value))
    (setv attn-scores (@ queries keys.T))
    (setv attn-weights (torch.softmax (/ attn-scores (** d-k 0.5)) :dim -1))
    
    (setv context-vec (@ attn-weights values))
    
    context-vec))


;; testing it out

(torch.manual_seed 123)
(setv sa-v1 (SelfAttentionV1 d-in d-out))
(print (sa-v1 inputs))

;; This implementation can be improved futher by utilizing PyTorch's nn.Linear layers, which effectively perform matrix multiplication when the bias units are disabled. Additionally, a significant advantage of using nn.Linear instead of manually implementing nn.Parameter is that nn.Linear has optimized weight initialization scheme contributing to more stable and effective model training
