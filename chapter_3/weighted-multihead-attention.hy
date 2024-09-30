(import torch)
(import chapter_3.weighted_multihead_attention :as MHA)

(setv inputs (torch.tensor
              [[0.43 0.15 0.89]
               [0.55 0.87 0.66]
               [0.57 0.85 0.64]
               [0.22 0.58 0.33]
               [0.77 0.25 0.10]
               [0.05 0.80 0.55]]))

(torch.manual_seed 123)

(setv batch (torch.stack #(inputs inputs) :dim 0))
(setv batch-size (.size batch 0))
(setv context-length (.size batch 1))
(setv d-in (.size batch 2))
(setv d-out 2)

(setv mha (MHA.MultiHeadAttention d-in d-out context-length 0.0 :num_heads 2))

(setv context-vecs (mha batch))
(print context-vecs)
(print "context-vecs .size" (.size context-vecs))
