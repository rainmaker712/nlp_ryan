#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:35:47 2017

modu-deepnlp
modubot

http://www.usefulparadigm.com/2016/04/06/creating-a-slack-bot-with-aws-lambda-and-api-gateway/
https://www.fullstackpython.com/blog/build-first-slack-bot-python.html

@author: ryan
https://hooks.slack.com/services/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

"""

import sys
sys.path.append('/home/ryan/nlp_ryan/Chatbot/Slack_Bot')
from mcbot_chat import make_sentence
import os, re, json, random

dict_file = "/home/ryan/nlp_ryan/Chatbot/Slack_Bot/markov-toji.json"
dic = json.load(open(dict_file,"r"))

import os
import time
from slackclient import SlackClient
import random

#Bot ID & Token
#slack_client.api_call("api.test")
BOT_NAME = 'modubot'
BOT_ID = 'U5Z492W0J'
slack_token = 'your token'

#export BOT_NAME='modubot'
#export slack_token='xoxb-203145098018-UFRw9AIzGDiZcuc4aSF1kFdl'

# instantiate Slack & Twilio clients
slack_client = SlackClient(slack_token)

#Check if everything is alright
is_ok = slack_client.api_call("users.list").get('ok')

# find the id of our slack bot
if(is_ok):
    for user in slack_client.api_call("users.list").get('members'):
        if user.get('name') == BOT_ID:
            print(user.get('id'))

# how the bot is mentioned on slack
def get_mention(user):
    return '<@{user}>'.format(user=user)

slack_mention = get_mention(BOT_ID)
           
#Start Chatbot
SOCKET_DELAY = 1

def is_private(event):
    """Checks if private slack channel"""
    return event.get('channel').startswith('D')

def is_for_me(event):
    #chekc if not my own event
    type = event.get('type')
    if type and type == 'message' and not(event.get('user')==BOT_ID):
            #in case it is a private message
            if is_private(event):
                return True
            #in case it is not a private
            text = event.get('text')
            channel = event.get('channel')
            if slack_mention in text.strip().split():
                return True
                
def post_message(message, channel):
    slack_client.api_call('chat.postMessage', channel=channel,
                          text=message, as_user=True)

import nltk
    
def is_hi(message):
    tokens = [word.lower() for word in message.strip().split()]
    return any(g in tokens
               for g in ['안녕', '안녕하세요', '테스트'])

def is_bye(message):
    tokens = [word.lower() for word in message.strip().split()]
    return any(g in tokens
               for g in ['bye', 'goodbye', 'revoir', 'adios', 'later', 'cya'])

def say_hi(user_mention):
    """Say Hi to a user by formatting their mention"""
    response_template = random.choice([make_sentence(dic)])
    return response_template.format(mention=user_mention)

def say_bye(user_mention):
    """Say Goodbye to a user"""
    response_template = random.choice(['see you later, alligator...',
                                       'adios amigo',
                                       'Bye {mention}!',
                                       'Au revoir!'])
    return response_template.format(mention=user_mention)

    
def handle_message(message, user, channel):
    if is_hi(message):
        user_mention = get_mention(user)
        post_message(message=say_hi(user_mention), channel=channel)
    elif is_bye(message):
        user_mention = get_mention(user)
        post_message(message=say_bye(user_mention), channel=channel)
    
def run():
    if slack_client.rtm_connect():
        print('[.] modubot is ON...')
        while True:
            event_list = slack_client.rtm_read()
            if len(event_list) > 0:
                for event in event_list:
                    print(event)
                    if is_for_me(event):
                        handle_message(message=event.get('text'), user=event.get('user'), channel=event.get('channel'))
            time.sleep(SOCKET_DELAY)
    else:
        print('[!] Connection to Slack failed.')
        
if __name__=='__main__':
    run()
