(import torch)
(import hyrule)

;; self attention -- computes attention weights by relating different positions within a single input sequence. It assesses and learns the relationships and dependencies between various parts of the input itself such as words in a sentence or pixels in an image. 
;; calculating attention scores
(setv inputs (torch.tensor
              [[0.43 0.15 0.89]
               [0.55 0.87 0.66]
               [0.57 0.85 0.64]
               [0.22 0.58 0.33]
               [0.77 0.25 0.10]
               [0.05 0.80 0.55]]))

(setv query (get inputs 1))
(print query)


(setv attn-scores-2 (torch.empty (.size (get inputs 0))))
(print attn-scores-2)

(get inputs 0)
(get inputs 0 2)


(setv result [])
(for [p inputs]
  (.append result (torch.dot p query)))
(print result)
(setv computed-attention-scores (torch.stack result))
(print computed-attention-scores)

;; normalize the score to obtain attention weights that sum up to 1 -- this normalization is useful for interpretation and maintaining training stability in an LLM
;; below are the attention weights:
(setv attn-weights-2-tmp (/ computed-attention-scores (sum computed-attention-scores)))
(print attn-weights-2-tmp)
(print "Sum: " (sum attn-weights-2-tmp))


;; naive softmax function for normalizing attention scores
;; softmax ensures that the attention weights are always positive -- this makes the output interpretable as probabilities or relative importance wehre higher weights indicate greater importance
;; this naive implementation could be vulnerable to underflow and overflow when dealing with small/large inputs
;; use Pytorch implementation to prevent these issues
(defn softmax-naive [x] 
  (/ (torch.exp x) (sum (torch.exp x))))

(setv attn-scores-3 computed-attention-scores)
(setv attn-weights-2-naive (softmax-naive attn-scores-3))
(print "attention weights: " attn-weights-2-naive)
(print "sum: " (sum attn-weights-2-naive))

;; using pytorch's impl
(setv attn-weights-4 (torch.softmax attn-scores-3 :dim 0))
(print "attention weights: " attn-weights-4)
(print "sum: " (sum attn-weights-4))

;; Next -> calculating the context vector z^(2) by multiplying the embedded input tokens x^(i) with the corresponding attention weights and then summing the resulting vectors
;; Therefore, context vector z^(2) is the weighted sum of all input vecotrs obtained by multiplying each input vector by its corresponding attention weight

;; using for loops is slow for this, so use matrix multiplication:
(setv attn-scores (@ inputs inputs.T))
(print attn-scores)

;; normalize each row so that the values in each row sum to 1
(setv attn-weights (torch.softmax attn-scores :dim -1))
(print attn-weights)

(print "Row 2 sum: " (sum (get attn-weights 1)))
(print "All rows sums: " (.sum attn-weights :dim -1))


;; compute all context vectors via matrix multiplication
(setv all-context-vecs (@ attn-weights inputs))
(print all-context-vecs)


;; Implementing self-attention with trainable weights
;; this is called scaled dot-product attention
;; we want to compute context vectors as weighted sums over the input vectors specific to a certain input element
;; the most notable difference is the introduction of weight matrices that are updated during model training.


(setv x-2 (get inputs 1))
(setv d-in (.size inputs 1)) ;; input embedding size = 3
(setv d-out 2) ;; output embedding size = 2 

;; normally in GPT like models the input and output would be the same size


(torch.manual_seed 123)

(setv w-query (torch.nn.Parameter (torch.rand d-in d-out) :requires_grad False))
(print w-query)

(setv w-key (torch.nn.Parameter (torch.rand d-in d-out) :requires_grad False))
(print w-key)

(setv w-value (torch.nn.Parameter (torch.rand d-in d-out) :requires_grad False))
(print w-value)

(setv query-2 (@ x-2 w-query))
(setv key-2 (@ x-2 w-key))
(setv value-2 (@ x-2 w-value))
(print query-2) ;; two dimensional vectory since the number of columns of the corresponding weight matrix via d-out == 2


;; weight parameters vs. attention weights - in weight matrices 'W' the term weight is short for weight parameters the values of a neural network that are optimized during training. This is not to be confused with the attention weights. Attention weights determine the extent to which a context vector depends on the different parts of the input (to what extent the network focuses on different parts of the input)
;; weight parameters are the fundamental learned coefficients that define the network's connections, while attention weights are dynamic context specific values.


;; obtain all keys and values via matrix multiplication
(setv keys (@ inputs w-key))
(setv values (@ inputs w-value))

;; successfully projected the six input tokens from a three-dimensional onto a two-dimensional embedding space
(print "keys.size: " (keys.size))
(print "values.size: " (values.size))

;; Computing attention scores --- the attention score computation is a dot-product computation similar to what we used in the simplified self-attention mechanism. The new aspect here is that we are not directy computing the dot-product between the input elements but using the query and key obtained by transforming the inputs via respective weight matrices.

(setv keys-2 (get keys 1))
(print keys-2)

(setv attn-score-22 (.dot query-2 keys-2))
(print attn-score-22) ;; un-normalized attention score -> tensor (1.8524)

;; generalize this computation to all attention scores via matrix multiplication
(setv attn-scores-2 (@ query-2 keys.T))
(print attn-scores-2) ;; tensor ([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])
                      ;; as you can see the second value is the same as the above verifying it is correct 

;; now go from attention scores to attention weights
;; compute the attention weights by scaling the attention scores and using the softmax function
;; however, now we scale the attention scores by dividing them by the square root of the embedding dimension of the keys (taking the square root mathematically the same as exponentiating by 0.5)

(setv d-k (.size keys -1))
(setv attn-weights-2 (torch.softmax (/ attn-scores-2 (** d-k 0.5)) :dim -1))
(print attn-weights-2) ;; tensor ([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]) 
           
;; rationale behind scaled-dot product attention:
;; reason for normalization by embedding dimension size is to improve the training performance by avoiding small gradients, which can drastically slow down learning or cause training to stagnate. The scaling by the square root of the embedding dimension is the reason why this self-attention mechanism is also called scaled-dot product attention.

(setv context-vec-2 (@ attn-weights-2 values))
(print context-vec-2) ;; tensor ([0.3061, 0.8210])
             
;; next compute all context vectors in the input sequence from z^(1) to z^(T)

(print "see SelfAttentionV1.hy")
