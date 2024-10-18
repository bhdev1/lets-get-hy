(import pygame)

(import games.invasion.image_paths :as image_paths)

(defclass Ship []
  (defn __init__ [self invasion-game]
    (setv self.screen invasion-game.screen)
    (setv self.screen_rect (.get_rect invasion-game.screen))
    (setv self.image (pygame.image.load image_paths.SHIP-IMAGE-PATH))
    (setv self.rect (.get_rect self.image))
    (setv self.rect.midbottom self.screen_rect.midbottom))

  (defn draw-ship [self]
    (self.screen.blit self.image self.rect)))


