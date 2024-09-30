#!/usr/bin/env hy
(import flask [Flask render_template request])
(import jinja2 [Environment FileSystemLoader])

(setv environment 
  (Environment :loader (FileSystemLoader "/home/baron/Projects/hy_projects/diy-llm/tokenization/flask/app/templates")))

(setv -template1 (.get_template environment "template1.j2"))

(setv app (Flask "Flask test"))

(defn [(.route app "/")] 
  index [] 
  (render_template -template1))

(defn [(.route app "/response" :methods ["POST"])]
  response []
  (setv name (request.form.get "name"))
  (print name)
  (render_template -template1 :name name))


(app.run)