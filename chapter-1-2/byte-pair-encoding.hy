(import tiktoken)

;; 2

(setv special-tokens (set ["<|endoftext|>"]))

(setv tokenizer (tiktoken.get_encoding "gpt2"))
(setv tokenizer.allowed_special special_tokens)

(setv text ["Hello, do you like tea? <|endoftext|> In the sunlit terraces" "of someunknownPlace"])


;; encoding
(setv integers (list))
(for [sentence text]
  (print sentence)
   (setv ids (tokenizer.encode sentence :allowed_special special_tokens))
  (.append integers ids))
(print integers)

;; decoding
(setv strings (list))
(for [i integers]
  (setv s (tokenizer.decode i))
  (.append strings s))

(print strings)
