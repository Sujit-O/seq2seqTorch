from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals, print_function, division

import os
import tensorflow as tf
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from threading import Thread
import sys
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
UNK_token = 2

MAX_LENGTH = 15
hidden_size = 512
num_layer=3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('tmp/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('**:**')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 
        
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    sys.stdout.write("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    sys.stdout.write("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    sys.stdout.write("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    sys.stdout.write("Counting words:")
    print(input_lang.name, input_lang.n_words)
    sys.stdout.write(str(input_lang.name)+' : '+ str(input_lang.n_words))
    print(output_lang.name, output_lang.n_words)
    sys.stdout.write(str(output_lang.name)+' : '+ str(output_lang.n_words))
    sys.stdout.flush()
    return input_lang, output_lang, pairs



#if  os.path.isfile('tmp/input_lang.en'):
#    lines = open('tmp/input_lang.en', encoding='utf-8').\
#        read().strip().split('\n')
#if  os.path.isfile('tmp/input_lang.en'):
#    with tf.gfile.GFile('tmp/input_lang.en', mode="ab") as file:
#      for i in range(len(qlines)):
#            if i in test_ids:
#               vocab_file2.write(alines[i].encode('utf-8') + b"\n")
#        file.write(str(iter).encode('utf-8') +' : '.encode('utf-8')+str(loss).encode('utf-8')+ b"\n")
#    
#if os.path.isfile('model/encoder.pkl'):
#    os.remove('model/encoder.pkl')    
#
#with tf.gfile.GFile('model/results.txt', mode="ab") as file:
#      file.write(str(iter).encode('utf-8') +' : '.encode('utf-8')+str(loss).encode('utf-8')+ b"\n")
    
input_lang, output_lang, pairs = prepareData('eng', 'eng', True)
#print(random.choice(pairs))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def indexesFromSentence(lang, sentence):
#    val=[]
#    try:
#        val=[lang.word2index[word] for word in sentence.split(' ')]
#    except KeyError:
#        val=2
    val=[]
    for word in sentence.split(' '):
        try:
            val.append(lang.word2index[word])
        except KeyError:
             val.append(2)
    return val

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


sys.stdout.write("\n Initializing the encoder and decoder")
sys.stdout.flush()
encoder_loaded=EncoderRNN(input_lang.n_words, hidden_size,num_layer)
decoder_loaded=AttnDecoderRNN(hidden_size, output_lang.n_words,
                               num_layer, dropout_p=0.1)

sys.stdout.write("\n Initializing cuda cores")
sys.stdout.flush()
if use_cuda:
    encoder_loaded = encoder_loaded.cuda()
    decoder_loaded = decoder_loaded.cuda()
 
sys.stdout.write("\n Loading encoder and decoder")
sys.stdout.flush()
encoder_loaded.load_state_dict(torch.load('model/encoder50000.pkl'))
decoder_loaded.load_state_dict(torch.load('model/decoder50000.pkl'))
sys.stdout.write("\n encoder and decoder loading successful")
sys.stdout.flush()       

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' 
EN_BLACKLIST = '"#$%&()*?!.\'+,-/:;<=>@[\\]^_`{|}~\''

def filter_sentence(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])
    
def start(bot, update):
   update.message.reply_text('Hi! I am Jiango Wango. How may I assist you today?')


def help(bot, update):
    update.message.reply_text('Help!')


def removeDuplicateWords(oldsentence):
    tempS=oldsentence.split()
    sentence = []
    [sentence.append(x) for x in tempS if x not in sentence]
    sentence=' '.join(sentence)
    return sentence



def echo(bot, update):
        global model
        global sess
        text=str.lower(update.message.text)

        if '@w' in text:
            text=text.split('@w')[-1]
            sys.stdout.write(text)
            sys.stdout.flush()
#            bot.send_message(chat_id=update.message.chat_id, text=text)
            text=filter_sentence(text, EN_WHITELIST)
            output_words, attentions = evaluate(encoder_loaded, decoder_loaded, text)
            sentence = ' '.join(output_words)
            sentence =  removeDuplicateWords(sentence)
            sentence=sentence.split('<EOS>')[0]
            sys.stdout.write(sentence)
            sys.stdout.flush()
            bot.send_message(chat_id=update.message.chat_id, text=sentence+'.')
            with tf.gfile.GFile(os.getcwd()+'/TelegramConversation.en', mode="ab") as vocab_file:
                vocab_file.write(str(update.message.chat_id).encode('utf-8')+b":--:"+str(update.message.from_user.id).encode('utf-8')+b":--:"+str(update.message.from_user.first_name).encode('utf-8')+b":--:"+str(update.message.message_id).encode('utf-8')+b":--:"+text.encode('utf-8') + b"\n")
                vocab_file.write(str(update.message.chat_id).encode('utf-8')+b":--:"+str(update.message.from_user.id).encode('utf-8')+b":--:"+str('Wango').encode('utf-8')+b":--:"+str(update.message.message_id).encode('utf-8')+b":--:"+sentence.encode('utf-8') + b"\n")  
        else:
    		#bot.send_message(chat_id=update.message.chat_id, text='Sorry I do not know how to respond to this one yet!')
            with tf.gfile.GFile(os.getcwd()+'/TelegramConversation.en', mode="ab") as vocab_file:
                vocab_file.write(str(update.message.chat_id).encode('utf-8')+b":--:"+str(update.message.from_user.id).encode('utf-8')+b":--:"+str(update.message.from_user.first_name).encode('utf-8')+b":--:"+str(update.message.message_id).encode('utf-8')+b":--:"+text.encode('utf-8') + b"\n")
    			#vocab_file.write(str(update.user.first_name).encode('utf-8')+b":"+text.encode('utf-8') + b"\n")
                
def decode():

#    updater = Updater("376658228:AAG4DxnElFJQetInxQtdLRoKQNDCEkYftFw")
    updater = Updater("393007383:AAEtMdzXRtKNbpDfoUZ8qC2u4jjEwyUNmzc")
    
    # Get the dispatcher to register handlers
    dp = updater.dispatcher
    	
    def stop():
        """Gracefully stop the Updater and exit"""
        updater.stop()
        os._exit(1)
    
    def stopcall(bot, update):
        update.message.reply_text('Bot is Stopping...')
        Thread(target=stop).start()
 
      	  
    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("stopp", stopcall))
    dp.add_handler(MessageHandler(Filters.text, echo))
    
    # Start the Bot
    updater.start_polling()
   
    updater.idle()

def main(_):
    decode()
 

if __name__ == "__main__":
  tf.app.run()
