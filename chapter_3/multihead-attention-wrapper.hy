(import torch)
(import chapter_3.causal_attention :as CA)
(import chapter_3.multihead_attention_wrapper :as MAW)

(torch.manual_seed 123)
(setv batch (torch.stack #(inputs inputs) :dim 0))
(print batch)

(setv context-length (.size batch 1))
(print context-length)

(setv d-in 3)
(setv d-out 2)

(setv mha (MAW.MultiHeadAttentionWrapper d-in d-out context-length 0.0 :num_heads 2))

(setv context-vecs (mha batch))
(print context-vecs)
(print "context-vecs .size ==" (.size context-vecs))



