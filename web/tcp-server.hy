(import socket)
(import threading)

(setv IP "0.0.0.0")
(setv PORT 9998)


(defn handle-client [client-socket]
  (setv request (.recv client-socket 1024))
  (print f"[*] Received:" (.decode request) )
  (.send client-socket b"ACK"))


(defn handler [server] 
  (while True
    (setv [client address] (.accept server))
    (print f"[*] Accepted connection from: " address)
    (setv client-handler (threading.Thread :target handle-client :args client))
    (.start client-handler)))


(defn start-server []
  (setv server (socket.socket socket.AF_INET socket.SOCK_STREAM))
  (.bind server #(IP PORT)) 
  (.listen server 5)
  (print f"[*] Listening on {IP}:{PORT}") 
  (handler server))


(start-server)
