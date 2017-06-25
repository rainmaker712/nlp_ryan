#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:35:47 2017

modu-deepnlp
modubot
@author: ryan
https://hooks.slack.com/services/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

"""

import os
from slackclient import SlackClient

token = 'your token'
slack_client = SlackClient(token)
#slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
print(slack_client.api_call("api.test"))
print(slack_client.api_call("api.test"))

if __name__ == "__main__":
    api_call = slack_client.api_call("users.list")
    if api_call.get('ok'):
        # retrieve all users so we can find our bot
        users = api_call.get('members')
        for user in users:
            if 'name' in user and user.get('name') == BOT_NAME:
                print("Bot ID for '" + user['name'] + "' is " + user.get('id'))
    else:
        print("could not find bot user with the name " + BOT_NAME)

        
    
                