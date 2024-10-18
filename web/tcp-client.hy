(import socket)

(setv target-host "www.google.com")
(setv target-port 80)

(setv client (socket.socket socket.AF_INET socket.SOCK_STREAM))

(.connect client #(target-host target-port))

(.send client b"GET / HTTP/1.1\r\nHost: google.com\r\n\r\n")

(setv response (.recv client 4096))

(print (.decode response))

(.close client)
