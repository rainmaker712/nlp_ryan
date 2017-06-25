#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 20:02:25 2017

@author: ryan
"""

import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitter
import urllib.request
import os, re, json, random

# 문장 읽어 들이기 --- (※4)
dict_file = "/home/ryan/nlp_ryan/Chatbot/Slack_Bot/markov-toji.json"
dic = json.load(open(dict_file,"r"))

# 문장 만들기 --- (※3)
def make_sentence(dic):
    ret = []
    if not "@" in dic: return "no dic" 
    top = dic["@"]
    w1 = word_choice(top)
    w2 = word_choice(top[w1])
    ret.append(w1)
    ret.append(w2)
    while True:
        w3 = word_choice(dic[w1][w2])
        ret.append(w3)
        if w3 == ".": break
        w1, w2 = w2, w3
    ret = "".join(ret)
    # 띄어쓰기
    params = urllib.parse.urlencode({
        "_callback": "",
        "q": ret
    })
    # 네이버 맞춤법 검사기를 사용합니다.
    data = urllib.request.urlopen("https://m.search.naver.com/p/csearch/dcontent/spellchecker.nhn?" + params)
    data = data.read().decode("utf-8")[1:-2]
    data = json.loads(data)
    data = data["message"]["result"]["html"]
    #data = soup = BeautifulSoup(data, "html.parser").getText()
    data = BeautifulSoup(data, "html.parser").getText()
    
    # 리턴
    return data

def word_choice(sel):
    keys = sel.keys()
    return random.choice(list(keys))