#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: yolo
# @Date  : 2018/11/9
# @Contact : zrq
from socket import *
import threading
import xml.dom.minidom as xmldom
import os
import xml.sax
import xml.sax.handler
import json

class XMLHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        self.buffer = ""
        self.mapping = {}

    def startElement(self, name, attributes):
        self.buffer = ""

    def characters(self, data):
        self.buffer += data

    def endElement(self, name):
        self.mapping[name] = self.buffer

    def getDict(self):
        return self.mapping

BIND_IP = '127.0.0.1'  # 监听哪些网络  127.0.0.1是监听本机 0.0.0.0是监听整个网络
BIND_PORT = 8089  # 监听自己的哪个端口
BUFF_SIZE = 1024  # 接收从客户端发来的数据的缓存区大小

s = socket(AF_INET, SOCK_STREAM)
# 将套接字与指定的ip和端口相连
s.bind((BIND_IP, BIND_PORT))
# 启动监听，并将最大连接数设为5
s.listen(5)     # 最大连接数
print("[*] listening on {}:{}".format(BIND_IP, BIND_PORT))

def handle_client(client_sock, client_address):
    while True:
        receive_data = client_sock.recv(BUFF_SIZE).decode('utf-8')
        # print("接受到的XML为："+receive_data)
        # xh = XMLHandler()
        # xml.sax.parseString(receive_data, xh)
        # ret = xh.getDict()
        # ret_str=str(ret)
        # print('解析的XML')
        # print(ret)

        print("收到客户端的JSON："+receive_data)
        jsontext = json.dumps(receive_data)
        print("解析的JSON数据,返回给客户端：")
        print(jsontext)
        if not receive_data:
            break
        #send_data = "JIEXIXML::"+ret_str + ' from python sever'+'\n'
        send_data = "服务端JIEXIJSON::" + jsontext + ' from python sever' + '\n'
        # 返回一个数据包
        client_sock.send(send_data.encode())
    client_sock.close()

while True:
    client_sock, client_address = s.accept()
    print("[*] Accepted connection from: {}:{}".format(client_address[0], client_address[1]))
    # 传输数据都利用client_sock，和s 无关
    client_handler = threading.Thread(target=handle_client, args=(client_sock, client_address))  # t 为新创建的线程
    client_handler.start()

# s.close()
