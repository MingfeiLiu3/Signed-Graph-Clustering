import json
import os
import stag.graphio
import stag.graph
#import d3py
import logging

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import numba as nb
from numba.typed import Dict
import numba.types as types

from textblob import TextBlob
import pickle

import py3langid as langid
import re
from random import sample
import stag.graphio
import stag.graph
import stag.cluster

import tarfile

tweets = []

#I choose one month dataset(2020-06) as an example and download it. 
for date_file in os.listdir('/Users/xiaomai/Desktop/2020-06'):
    if date_file != ".DS_Store":
        path = '/Users/xiaomai/Desktop/2020-06/' + date_file
        for file_name in os.listdir(path):
            json_file = open(path +'/'+ file_name,'r', encoding='utf-8')
            for line in json_file .readlines():
                dic = json.loads(line)
                tweets.append(dic)
             
"""
  -------------------------------------------------------------------------------------------------------------------------------
     Tweet Type  ||        How to identify       ||  (Original)User 1 field  ||  User 2 field  ||         Text field
  --------------------------------------------------------------------------------------------------------------------------------
        Reply        in_reply_to_user_id != None        in_reply_to_user_id       user/id                    text 
                                                                                                    or extended_tweet/full_text
       Retweet         retweeted_status != None    retweeted_status/user/id       user/id 
    Quoted Tweet     quoted_status != None &                                                               text 
                      retweeted_status == None        quoted_status/user/id        user/id          or extended_tweet/full_text
"""
user_list = []

#For each tweet object, I first selected reply, quoted, and retweeted tweets I want and stored them in txt files. 

for tweet in tweets:
    # Retweets attribute contains a representation of the original Tweet that was retweeted.
    retweeted_status = tweet.get('retweeted_status')
    quoted_status = tweet.get('quoted_status')
    
    if tweet.get('in_reply_to_user_id_str') != None:
            #If the represented Tweet is a reply, in_reply_to_user_id_str = original Tweet’s author ID
            original_reply_user = tweet.get('in_reply_to_user_id_str')
            #user = The user who posted this Tweet
            reply_user = tweet.get('user').get('id_str')
            
            if tweet.get('truncated') == False:
                reply_tweet = tweet.get('text')
            else:
                reply_tweet = tweet.get('extended_tweet').get('full_text')

            # unpack the result tuple in variables & identified language and probability
            lang, prob = langid.classify(reply_tweet)
            split_text = reply_tweet.split('https://')
            split_text_parts = split_text[0].split(' ')
            
            del_list = []
            for part in split_text_parts:
                if part != '' and part[0] == '@':
                    del_list.append(part)
            for p in del_list:
                split_text_parts.remove(p)
            final_reply_text = ' '.join(split_text_parts)
            reply_pol = TextBlob(final_reply_text).sentiment.polarity
            
            #remove the zero weights
            if lang == 'en' and abs(prob) > 100 and reply_pol != 0.0:
                #remove the self reply
                if reply_user != original_reply_user:
                    
                    #store all users
                    if original_reply_user not in user_list:
                        user_list.append(original_reply_user)
                        with open('user_list.txt','a') as file:
                            file.write(str(user_list.index(original_reply_user))+'   '+original_reply_user+'\n')
                    if reply_user not in user_list:
                        user_list.append(reply_user)
                        with open('user_list.txt','a') as file:
                            file.write(str(user_list.index(reply_user))+'   '+reply_user+'\n')
                    
                    with open("reply_tweet_construction.txt","a") as file:
                        file.write(original_reply_user + '     ')
                        file.write(reply_user + '     ')
                        file.write(str(reply_pol)+'     ')
                        file.write(final_reply_text+'\n')
                        
    if retweeted_status != None:
        original_retweet_user = retweeted_status.get('user').get('id_str')
        #user = The user who posted this Tweet
        retweet_user = tweet.get('user').get('id_str')
        
        #any retweet is regarded as postive 
        retweet_pol = 0.5
        
        #store all users
        if original_retweet_user not in user_list:
            user_list.append(original_retweet_user)
            with open('user_list.txt','a') as file:
                file.write(str(user_list.index(original_retweet_user))+'   '+original_retweet_user+'\n')
        if retweet_user not in user_list:
            user_list.append(retweet_user)
            with open('user_list.txt','a') as file:
                file.write(str(user_list.index(retweet_user))+'   '+retweet_user+'\n')
        
        with open("retweet_construction.txt","a") as file:
            file.write(original_retweet_user + '     ')
            file.write(retweet_user + '     ')
            file.write(str(retweet_pol)+'\n')
    
    #quoted_status field only surfaces when the Tweet is a quote Tweet.
    #This attribute contains the Tweet object of the original Tweet that was quoted.
    elif quoted_status != None:
        original_quoted_user = quoted_status.get('user').get('id_str')
        #user = The user who posted this Tweet
        quoted_user = tweet.get('user').get('id_str')
        
        if quoted_status.get('truncated') == True:
            quoted_tweet = quoted_status.get('extended_tweet').get('full_text')
        else:
            quoted_tweet = quoted_status.get('text')
            
        # unpack the result tuple in variables & identified language and probability
        lang, prob = langid.classify(quoted_tweet)
        split_text = quoted_tweet.split('https://')
        split_text_parts = split_text[0].split(' ')
            
        del_list = []
        for part in split_text_parts:
            if part != '' and part[0] == '@':
                del_list.append(part)
        for p in del_list:
            split_text_parts.remove(p)
        final_quoted_text = ' '.join(split_text_parts)
        quoted_pol = TextBlob(final_quoted_text).sentiment.polarity
            
        #remove the zero weights
        if lang == 'en' and abs(prob) > 100 and quoted_pol != 0.0:
            #remove the self reply
            if reply_user != original_reply_user:
                
                #store all users
                if original_quoted_user not in user_list:
                    user_list.append(original_quoted_user)
                    with open('user_list.txt','a') as file:
                        file.write(str(user_list.index(original_quoted_user))+'   '+original_quoted_user+'\n')
                if quoted_user not in user_list:
                    user_list.append(quoted_user)
                    with open('user_list.txt','a') as file:
                        file.write(str(user_list.index(quoted_user))+'   '+quoted_user+'\n')
                
                with open("quoted_tweet_construction.txt","a") as file:
                    file.write(original_quoted_user + '     ')
                    file.write(quoted_user + '     ')
                    file.write(str(quoted_pol) + '     ')
                    file.write(final_quoted_text+'\n')
                        

"""                      
Buid a signed undirected graph.
Firstly, read three txt files and selected the corresponding vertices and edges information to build the user graph. 
"""
G_pos = nx.Graph()

reply_file = open("reply_tweet_construction.txt",encoding='utf-8')
while True:
    reply_line = reply_file.readline()
    if reply_line:
        reply_tweet_info = reply_line.split('     ')
        if len(reply_tweet_info)>3 and reply_tweet_info[0] != '' and reply_tweet_info[1] != '':
            if reply_tweet_info[0][0] in ['0','1','2','3','4','5','6','7','8','9'] and reply_tweet_info[1][0] in ['0','1','2','3','4','5','6','7','8','9']:
                original_reply_user = reply_tweet_info[0]
                reply_user = reply_tweet_info[1]
                pol_reply = float(reply_tweet_info[2])
                
                if pol_reply > 0:
                    #use the index of users_list as the node name (shorter and safer)
                    G_pos.add_node(user_list.index(original_reply_user))
                    G_pos.add_node(user_list.index(reply_user))
                    G_pos.add_edge(user_list.index(original_reply_user), user_list.index(reply_user), weight=pol_reply)
                
    else:
        break
reply_file.close()


for retweet_line in open("retweet_construction.txt"):
    retweet_info = retweet_line.split('     ')
    original_retweet_user = retweet_info[0]
    retweet_user = retweet_info[1]
    pol_retweet = float(retweet_info[2])
    
    if pol_retweet > 0:
        #use the index of users_list as the node name (shorter and safer)
        G_pos.add_node(user_list.index(original_retweet_user))
        G_pos.add_node(user_list.index(retweet_user))
        G_pos.add_edge(user_list.index(original_retweet_user), user_list.index(retweet_user), weight=pol_retweet)

                
quoted_file = open("quoted_tweet_construction.txt",encoding='utf-8')
while True:
    quoted_line = quoted_file.readline()
    if quoted_line:
        quoted_tweet_info = quoted_line.split('     ')
        if len(quoted_tweet_info)>3 and quoted_tweet_info[0] != '' and quoted_tweet_info[1] != '':
            if quoted_tweet_info[0][0] in ['0','1','2','3','4','5','6','7','8','9'] and quoted_tweet_info[1][0] in ['0','1','2','3','4','5','6','7','8','9']:
                original_quoted_user = quoted_tweet_info[0]
                quoted_user = quoted_tweet_info[1]
                pol_quoted = float(quoted_tweet_info[2])
                
                if pol_quoted > 0:
                    #use the index of users_list as the node name (shorter and safer)
                    G_pos.add_node(user_list.index(original_quoted_user))
                    G_pos.add_node(user_list.index(quoted_user))
                    G_pos.add_edge(user_list.index(original_quoted_user), user_list.index(quoted_user), weight=pol_quoted)         
    else:
        break
quoted_file.close()
        
#store the edge information for above graph
g = stag.graph.from_networkx(G_pos)
filename = "pos_graph.edgelist"
stag.graphio.save_edgelist(g, filename) 
               
