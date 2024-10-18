(import sys)
(import pygame)

(import games.invasion.settings :as settings)

(defclass Invasion []
  (defn __init__ [self] 
    (.init pygame)
    (setv self.clock (.Clock pygame.time))
    (setv self.settings (settings.Settings))

    (setv self.screen (.set_mode pygame.display #(self.settings.screen_width self.settings.screen_height)))
    (.set_caption pygame.display "Invasion"))

  (defn run-game [self]
    (while True
      (for [event (.get pygame.event)]
        (when (= event.type pygame.QUIT)
          (.exit sys))

        (self.screen.fill self.settings.bg_color)

        (.flip pygame.display)
        (self.clock.tick 60)))))


(.run-game (Invasion))