(import io)
(import tiktoken)

;; 3
;; pg. 36

;; read in the file
(setv filepath "the-verdict.txt")
(with [f (io.open filepath :encoding "utf-8")]
  (setv raw-text (.read f)))

;; tokenize the text and encode it using byte pair encoding
(setv tokenizer (tiktoken.get_encoding "gpt2"))
(setv encoded-text (tokenizer.encode raw-text))
(print (len encoded-text))

; get a sample of the first 50 tokens
(setv encoded-sample (cut encoded-text 0 50))
 
;; input target pairs
(setv context-size 4)
(setv x (cut encoded-sample 0 context-size))
(setv y (cut encoded-sample 1 (+ 1 context-size)))
(print x)
(print y)

;; create the next word prediction tasks
;; this can be used for training the LLM
(for [i (range 1 (+ context-size 1))]
(setv context (cut encoded-sample 0 i))
(setv desired (get encoded-sample i))
  (print context "---->" desired))

;; decoded version makes a little more sense
(for [i (range 1 (+ context-size 1))]
  (setv context (cut encoded-sample 0 i))
  (setv desired (get encoded-sample i))
  (print (tokenizer.decode context) "---->" (tokenizer.decode [desired])))



