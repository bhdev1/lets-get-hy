(import sys)
(import pygame)

(defclass Invasion []
  (defn __init__ [self]
  (.init pygame)
    (setv self.screen (.set_mode pygame.display #(1200 800)))
    (.set_caption pygame.display "Invasion"))

  (defn run-game [self]
    (while True
      (for [event (.get pygame.event)]
        (when (= event.type pygame.QUIT)
          (.exit sys))
        
        (.flip pygame.display)))))

(setv invasion (Invasion))
(.run-game invasion)