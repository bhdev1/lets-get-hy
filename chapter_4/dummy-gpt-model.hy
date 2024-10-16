; (import torch)
; (import torch.nn :as nn)


; (defclass DummyTransformerBlock [nn.Module]
;   (defn __init__ [self cfg]
;     (.__init__ (super)))
;   (defn forward [self x] x))


; (defclass DummyLayerNorm [nn.Module]
;   (defn __init__ [self normalized_shape [eps 1e-5]]
;     (.__init__ (super)))
;   (defn forward [self x] x))


; (defclass DummyGPTModel [nn.Module]
;   (defn __init__ [self cfg]
;     (.__init__ (super))
;     (setv self.tok_emb (nn.Embedding (:vocab_size cfg) (:emb_dim cfg)))
;     (setv self.pos_emb (nn.Embedding (:context_length cfg) (:emb_dim cfg)))
;     (setv self.drop_emb (nn.Dropout (:drop_rate cfg)))
;     (setv self.trf_blocks (nn.Sequential [
    ; *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"]) ]
;     (DummyTransformerBlock cfg)]))
;     (setv self.final_norm (DummyLayerNorm (:emb_dim cfg)))
;     (setv self.out_head (nn.Linear (:emb_dim cfg) (:vocab_size cfg) (:bias False))))


;   (defn forward [self in_idx]
;     (setv batch_size (.size in_idx))
;     (setv seq_len (.size in_idx))
;     (setv tok_embeds (self.tok_emb in_idx))
;     (setv pos_embeds (self.pos_emb (torch.arange seq_len :device (:device in_idx))))

;     (setv x (+ tok_embeds pos_embeds))
;     (setv x (self.drop_emb x))
;     (setv x (self.trf_blocks x))
;     (setv x (self.final_norm x))
;     (setv logits (self.out_head x))
;     logits))
