(import torch)
(import torch.utils.data [Dataset DataLoader])
(import tiktoken)
(import io)

;; A dataset for batched inputs and targets

(defclass GPTDatasetV1 []
  (defn __init__ [self txt tokenizer max_length stride]
    (setv self.input_ids [])
    (setv self.target_ids [])
    (setv token_ids (tokenizer.encode txt))
    
    (for [i (range 0 (- (len token_ids) max_length) stride)]
      (setv input_chunk (cut token_ids i (+ i max_length)))
      (setv target_chunk (cut token_ids (+ i 1) (+ i 1 max_length)))
      (.append self.input_ids (torch.tensor input_chunk))
      (.append self.target_ids (torch.tensor target_chunk)))) 
  
(defn __len__ [self]
  (len self.input_ids))
  
(defn __getitem__ [self idx]
  [(get self.input_ids idx)
   (get self.target_ids idx)
   ]))


;; using GptDatasetV1 to load the inputs in batches via pytorch dataloader
(defn create-dataloader-v1 [txt
                            [batch_size 4]
                            [max_length 256]
                            [stride 128]
                            [shuffle True]
                            [drop_last True]
                            [num_workers 0]]
  (setv tokenizer (tiktoken.get_encoding "gpt2"))
  (setv dataset (GPTDatasetV1 txt tokenizer max_length stride))
  (setv dataloader (DataLoader 
                    dataset 
                    :batch_size batch_size 
                    :shuffle shuffle 
                    :drop_last drop_last 
                    :num_workers num_workers))
  dataloader)


;; testing the dataloader with batch size 1 for llm with context size of 4
(setv filepath "the-verdict.txt")
(with [f (io.open filepath :encoding "utf-8")]
  (setv raw-text (.read f)))

(setv dataloader (create-dataloader-v1 raw-text :batch_size 1 :max_length 4 :stride 1 :shuffle False))
(setv data-iter (iter dataloader))
(setv first-batch (next data-iter))

;; contains two tensors the first tensor stores the input token IDs and the second stores the target token IDs
;; each containing four token IDs because the max_length is set to 4
;; in the real world the input size would not be limited to 4 it would be at least 256
(print first-batch)

(setv second-batch (next data-iter))
(print second-batch)


; ex. 2.2 change settings to: max_length = 2, stride = 2 
; then max_length = 8, stride = 2

(setv dataloader-2 (create-dataloader-v1 raw-text :batch_size 1 :max_length 2 :stride 2 :shuffle False))
(setv data-iter-2 (iter dataloader-2))
(setv first-batch-2 (next data-iter-2))
(print first-batch-2)


(setv dataloader-3 (create-dataloader-v1 raw-text :batch_size 1 :max_length 8 :stride 2 :shuffle False))
(setv data-iter-3 (iter dataloader-3))
(setv first-batch-3 (next data-iter-3))
(print first-batch-3)



;; sample with batch size greater than 1
(setv dataloader-4 (create-dataloader-v1 raw-text :batch_size 8 :max_length 4 :stride 4 :shuffle False))
(setv data-iter-4 (iter dataloader-4))
(setv a (next data-iter-4))
(print "Inputs:\n" (get a 0))
(print "\nTargets:\n" (get a 1))


;; Creating token embeddings
;; the last step in preparing the input text for llm training is to convert the token IDs into embedding vectors
;; the embedding weights are random values -- this serves as the starting point for llm's learning process
;; continuous vecotr representation or embedding is necessary because llms are deep neural networks trained with backpropagation algorithms
;; 

(setv vocab-size 50257)
(setv output-dim 256)
(setv token-embedding-layer (torch.nn.Embedding vocab-size output-dim))

;; using the previous token-embedding-layer, if you sample data from the data loader, you embed each token in each batch into a 256 dimensional vector. If you have a batch size of 8 with 4 tokens each the result will be 8x4x256

(setv max-length 4)
(setv dataloader (create-dataloader-v1 raw-text :batch_size 8 :max_length max-length :stride max-length :shuffle False))
(setv data-iter (iter dataloader))
(setv inputs-and-targets (next data-iter))
(print inputs-and-targets)

; 8x4 dimensional
(print (.size (get inputs-and-targets 0)))

; 8x4x256
(setv token-embeddings (token-embedding-layer (get inputs-and-targets 0)))
(print (.size token-embeddings))


;; For a GPT model's absolute embedding approach you need to create another embedding layer that has the same embedding dimension as the token-embedding-layer

(setv context-length max-length)
(setv pos-embedding-layer (torch.nn.Embedding context-length output-dim))
(setv pos-embeddings (pos-embedding-layer (torch.arange context-length)))
(print (.size pos-embeddings))

;; add directly to the token embeddings
(setv input-embeddings (+ token-embeddings pos-embeddings))
(print (.size input-embeddings)) ;; can now be processed by the main LLM modules