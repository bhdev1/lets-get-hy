(import socket)

(setv target-host "127.0.0.1")
(setv target-port 9997)

(setv client (socket.socket socket.AF_INET socket.SOCK_DGRAM))

(.sendto client b"AAABBBCCC" #(target-host target-port))

(setv [data addr] (.recvfrom client 4096))

(.close client)

(print data)