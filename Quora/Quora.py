# -*- coding:utf-8 -*-
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from collections import Counter
def line_tokenize(line,low_freq_filter = True):
    english_stopwords = stopwords.words('english') #+ ['fail','property', 'dev', 'er', 'comments-', 'tim', 'act', 'describ', 'rework', 'value', 'fail', 'correct', 'ver', 'log',  'The', 'of', 'to', 'and', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'with', 'as', 'I', 'his', 'they', 'be', 'at', 'one', 'have', 'this', 'from', 'or', 'had', 'by', 'hot', 'word', 'but', 'what', 'some', 'we', 'can', 'out', 'other', 'were', 'all', 'there', 'when', 'up', 'use', 'your', 'how', 'said', 'an', 'each', 'she', 'which', 'do', 'their', 'if', 'will', 'way', 'about', 'many', 'then', 'them', 'would', 'like', 'so', 'these', 'her', 'long', 'make', 'thing', 'see', 'him', 'two', 'has', 'look', 'more', 'day', 'could', 'go', 'come', 'did', 'number',  'no', 'most', 'people', 'my', 'over', 'know', 'than', 'call', 'first', 'who', 'may', 'down', 'side', 'been', 'now', 'any', 'new', 'work', 'part', 'take', 'get', 'place', 'made', 'live', 'where', 'after', 'back', 'little', 'only', 'round', 'man', 'year', 'came', 'show', 'every', 'good', 'me', 'give', 'our', 'under', 'name', 'very', 'through', 'just', 'form', 'sentence', 'great', 'think', 'say', 'help', 'low', 'line',  'much', 'mean', 'before', 'move', 'right', 'boy', 'too', 'same', 'tell', 'does', 'set', 'three', 'want', 'air', 'well', 'also', 'play', 'small', 'end', 'put', 'home', 'hand',  'large', 'spell', 'add', 'even', 'land', 'here', 'must', 'big', 'high', 'such', 'follow', 'act', 'why', 'ask', 'men', 'change', 'went', 'light', 'kind', 'off', 'need', 'house', 'try', 'us', 'again', 'animal', 'point', 'mother', 'world', 'near', 'build', 'self', 'earth', 'father', 'head', 'stand', 'own', 'page', 'should',  'found', 'answer',  'grow', 'study', 'still', 'learn', 'plant', 'sun', 'four', 'between', 'keep', 'eye', 'never', 'last', 'let', 'thought', 'city', 'tree', 'cross', 'hard', 'start', 'might', 'story', 'saw', 'far', 'sea', 'draw', 'left', 'late', 'run', 'while', 'press', 'close', 'night', 'real', 'life', 'few', 'north', 'open', 'seem', 'together', 'next', 'white', 'children', 'begin', 'got', 'walk', 'example', 'ease', 'group', 'always', 'music', 'those', 'both', 'mark', 'often', 'letter', 'until', 'mile', 'river',  'feet', 'care', 'second', 'book', 'carry', 'took', 'eat', 'room', 'friend', 'began', 'idea', 'fish', 'mountain', 'stop', 'once', 'base', 'hear', 'horse', 'cut', 'sure', 'watch', 'color', 'face', 'wood', 'main', 'enough', 'plain', 'girl', 'usual', 'young', 'ready', 'above', 'ever', 'red', 'list', 'though', 'feel', 'talk', 'bird', 'soon', 'body', 'dog', 'family', 'pose', 'leave', 'song', 'measure', 'door', 'product', 'black', 'short',  'class', 'wind', 'question', 'happen', 'complete', 'ship', 'area', 'half', 'rock', 'order', 'fire', 'south', 'problem', 'piece', 'told', 'knew', 'pass', 'since', 'top', 'whole', 'king', 'space', 'heard', 'best', 'hour', 'better', 'true', 'during', 'hundred', 'five', 'remember', 'step', 'early', 'hold', 'west', 'ground', 'interest', 'reach', 'fast', 'verb', 'sing', 'listen', 'six', 'table', 'less', 'morning', 'ten', 'simple', 'several', 'vowel', 'toward', 'war', 'lay', 'against', 'slow', 'center', 'love', 'person', 'money', 'serve', 'appear', 'road', 'rain', 'rule', 'govern', 'pull', 'cold', 'notice', 'unit', 'town', 'fine', 'certain', 'fly', 'lead', 'cry',  'wait', 'plan', 'figure', 'star', 'box', 'noun', 'field', 'correct', 'able', 'pound', 'done', 'beauty', 'stood', 'contain', 'front', 'teach', 'week', 'final', 'gave', 'green', 'oh', 'quick', 'ocean', 'warm', 'free', 'minute', 'strong', 'special', 'mind', 'behind', 'clear', 'tail', 'produce', 'fact', 'street', 'inch', 'multiply', 'nothing', 'course', 'stay', 'wheel', 'full', 'force', 'blue', 'object', 'decide', 'deep', 'moon', 'island', 'foot', 'busy', 'test', 'boat', 'common', 'gold', 'possible', 'stead', 'dry', 'wonder', 'laugh', 'thousand', 'ago', 'ran', 'check', 'game', 'shape', 'equate', 'hot', 'miss', 'brought', 'heat', 'snow', 'tire', 'bring', 'yes', 'distant', 'fill', 'east', 'paint', 'language', 'among', 'grand', 'ball', 'yet', 'wave', 'drop', 'heart', 'am', 'present', 'heavy', 'dance', 'position', 'arm', 'wide', 'sail', 'size', 'vary', 'settle', 'speak', 'ice', 'matter', 'circle', 'pair', 'include', 'divide', 'felt', 'perhaps', 'pick', 'sudden', 'count', 'square', 'reason', 'length', 'represent', 'art', 'hunt', 'bed', 'brother', 'egg', 'ride',  'believe',  'sit', 'race', 'window', 'store', 'summer',  'sleep', 'prove', 'lone', 'leg', 'exercise', 'wall', 'catch', 'mount', 'wish', 'sky', 'board', 'joy', 'winter', 'sat', 'written', 'wild','kept', 'glass', 'grass', 'cow', 'job', 'edge', 'sign', 'visit', 'past', 'soft', 'fun', 'bright', 'gas', 'weather', 'month', 'million', 'bear', 'finish', 'happy', 'hope', 'flower', 'clothe', 'strange', 'gone', 'jump', 'baby', 'eight', 'village', 'meet', 'root', 'buy', 'raise', 'metal', 'whether', 'push', 'seven', 'paragraph', 'third', 'shall', 'held', 'hair', 'describe', 'cook', 'floor', 'either', 'result', 'burn', 'hill', 'safe', 'cat', 'century', 'consider', 'type', 'law', 'bit', 'coast', 'copy', 'phrase', 'silent', 'sand', 'soil', 'roll',  'finger','lie', 'beat', 'excite', 'natural', 'view', 'sense', 'ear', 'else', 'quite', 'broke', 'case', 'kill', 'son', 'lake',  'child', 'milk',  'dress', 'cloud', 'surprise', 'quiet',  'climb', 'cool', 'poor', 'lot','skin', 'smile', 'crease', 'melody', 'row', 'exact',  'die', 'least', 'shout', 'wrote', 'seed', 'tone', 'join', 'suggest', 'clean', 'break', 'lady', 'yard', 'rise', 'bad', 'blow', 'oil', 'blood', 'touch', 'grew', 'cent', 'mix', 'team', 'wire', 'cost', 'lost', 'brown', 'wear', 'garden', 'equal', 'sent', 'choose', 'fell', 'fit', 'flow',  'bank', 'collect', 'save', 'control']
    english_punctuations = ['.',',','.',':',';','?','(',')','[',']','&','!','*','@','#','$','%','__','...','-','=','==','..','--',"'", "''"]
    #line = unicode(line, errors='ignore')
    tokens = [word.lower() for word in nltk.tokenize.word_tokenize(line.decode('cp1252', 'ignore').encode('utf-8','ignore'))]
    tokens = [word for word in tokens if not word in english_stopwords and not word in english_punctuations and not word.isdigit()]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [LancasterStemmer().stem(t) for t in filtered_tokens]
    stems = [word for word in stems if not word in english_stopwords and not word in english_punctuations and not word.isdigit()]
    return stems
def topN_words(doc, i):
    """use topN words to draw word cloud, 'doc' is the text, 'i' represent N"""
    replace_reg = re.compile(r'(\]|\[|\,|\'|\"|\\)')
    doc1 = replace_reg.sub(' ', doc)
    wordcount = Counter(doc1.split()).most_common(i)
    #print wordcount
    topwords = []
    for j in range(len(wordcount)):
        for k in range(wordcount[j][1]):
            if len(wordcount[j][0]) >=2:
                topwords.append(wordcount[j][0])
    return topwords

from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts
topic_texts = ""
i=0
for i in range(212):
    quorafile = open("./output/quorafile" + str(i) +".txt", "r")
    topic_text = quorafile.readlines()
    topic_text = line_tokenize(str(topic_text))
    topic_texts += str(topic_text)

print "step0 done"
topN_words_texts = topN_words(topic_texts, 50)
tags = make_tags(get_tag_counts(str(topN_words_texts)), maxsize=100, minsize=8)
print "step1 done"
create_tag_image(tags, 'Quora.png', size=(900, 600), fontname='Lobster')
print "step2 done"
