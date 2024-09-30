(import urllib.request)
(import io)
(import re)

;; 1

;; The text to be tokenized 
(setv url "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/appendix-D/01_main-chapter-code/the-verdict.txt")

(setv filepath "the-verdict.txt")

;; Do the request to get the text from the site
(urllib.request.urlretrieve url filepath)


;; Open the file and output all of its content to the variable 'text'
(with [f (io.open filepath :encoding "utf-8")]
  (setv raw-text (.read f)))



; figuring out how Im going to use regex to parse the text
; (setv text "Hello, world. This, is a test.")
; (setv result (re.split r"([,.:;?_!\"()\']|--|\s+)" text))
; (print result)


(setv preprocessed (re.split r"([,.:;?_!\"()\']|--|\s+)" raw-text))
(print preprocessed)


; a list of all unique tokens sorted alphabetically
(setv all-words (sorted (set preprocessed)))
(print all-words)

(setv vocab-size (len all-words))
(print vocab-size)

; A dictionary where the key is a string and the value is an integer (the ID)
(setv vocab (dict (zip all-words (list (range 0 vocab-size)))))

; key/value swapped
(setv opposite-vocab (dict (zip (list (range 0 vocab-size)) all-words)))


; print the first 50 elements
(for [[item -id] (vocab.items)]
  (when (< -id 50)
    (print item -id)))


