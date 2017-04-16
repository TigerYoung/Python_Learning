import sklearn
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
def line_tokenize(line):
    english_stopwords = stopwords.words('english') + ['li', 'div', 'br']#+ ['fail','property', 'dev', 'er', 'comments-', 'tim', 'act', 'describ', 'rework', 'value', 'fail', 'correct', 'ver', 'log',  'The', 'of', 'to', 'and', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'with', 'as', 'I', 'his', 'they', 'be', 'at', 'one', 'have', 'this', 'from', 'or', 'had', 'by', 'hot', 'word', 'but', 'what', 'some', 'we', 'can', 'out', 'other', 'were', 'all', 'there', 'when', 'up', 'use', 'your', 'how', 'said', 'an', 'each', 'she', 'which', 'do', 'their', 'if', 'will', 'way', 'about', 'many', 'then', 'them', 'would', 'like', 'so', 'these', 'her', 'long', 'make', 'thing', 'see', 'him', 'two', 'has', 'look', 'more', 'day', 'could', 'go', 'come', 'did', 'number',  'no', 'most', 'people', 'my', 'over', 'know', 'than', 'call', 'first', 'who', 'may', 'down', 'side', 'been', 'now', 'any', 'new', 'work', 'part', 'take', 'get', 'place', 'made', 'live', 'where', 'after', 'back', 'little', 'only', 'round', 'man', 'year', 'came', 'show', 'every', 'good', 'me', 'give', 'our', 'under', 'name', 'very', 'through', 'just', 'form', 'sentence', 'great', 'think', 'say', 'help', 'low', 'line',  'much', 'mean', 'before', 'move', 'right', 'boy', 'too', 'same', 'tell', 'does', 'set', 'three', 'want', 'air', 'well', 'also', 'play', 'small', 'end', 'put', 'home', 'hand',  'large', 'spell', 'add', 'even', 'land', 'here', 'must', 'big', 'high', 'such', 'follow', 'act', 'why', 'ask', 'men', 'change', 'went', 'light', 'kind', 'off', 'need', 'house', 'try', 'us', 'again', 'animal', 'point', 'mother', 'world', 'near', 'build', 'self', 'earth', 'father', 'head', 'stand', 'own', 'page', 'should',  'found', 'answer',  'grow', 'study', 'still', 'learn', 'plant', 'sun', 'four', 'between', 'keep', 'eye', 'never', 'last', 'let', 'thought', 'city', 'tree', 'cross', 'hard', 'start', 'might', 'story', 'saw', 'far', 'sea', 'draw', 'left', 'late', 'run', 'while', 'press', 'close', 'night', 'real', 'life', 'few', 'north', 'open', 'seem', 'together', 'next', 'white', 'children', 'begin', 'got', 'walk', 'example', 'ease', 'group', 'always', 'music', 'those', 'both', 'mark', 'often', 'letter', 'until', 'mile', 'river',  'feet', 'care', 'second', 'book', 'carry', 'took', 'eat', 'room', 'friend', 'began', 'idea', 'fish', 'mountain', 'stop', 'once', 'base', 'hear', 'horse', 'cut', 'sure', 'watch', 'color', 'face', 'wood', 'main', 'enough', 'plain', 'girl', 'usual', 'young', 'ready', 'above', 'ever', 'red', 'list', 'though', 'feel', 'talk', 'bird', 'soon', 'body', 'dog', 'family', 'pose', 'leave', 'song', 'measure', 'door', 'product', 'black', 'short',  'class', 'wind', 'question', 'happen', 'complete', 'ship', 'area', 'half', 'rock', 'order', 'fire', 'south', 'problem', 'piece', 'told', 'knew', 'pass', 'since', 'top', 'whole', 'king', 'space', 'heard', 'best', 'hour', 'better', 'true', 'during', 'hundred', 'five', 'remember', 'step', 'early', 'hold', 'west', 'ground', 'interest', 'reach', 'fast', 'verb', 'sing', 'listen', 'six', 'table', 'less', 'morning', 'ten', 'simple', 'several', 'vowel', 'toward', 'war', 'lay', 'against', 'slow', 'center', 'love', 'person', 'money', 'serve', 'appear', 'road', 'rain', 'rule', 'govern', 'pull', 'cold', 'notice', 'unit', 'town', 'fine', 'certain', 'fly', 'lead', 'cry',  'wait', 'plan', 'figure', 'star', 'box', 'noun', 'field', 'correct', 'able', 'pound', 'done', 'beauty', 'stood', 'contain', 'front', 'teach', 'week', 'final', 'gave', 'green', 'oh', 'quick', 'ocean', 'warm', 'free', 'minute', 'strong', 'special', 'mind', 'behind', 'clear', 'tail', 'produce', 'fact', 'street', 'inch', 'multiply', 'nothing', 'course', 'stay', 'wheel', 'full', 'force', 'blue', 'object', 'decide', 'deep', 'moon', 'island', 'foot', 'busy', 'test', 'boat', 'common', 'gold', 'possible', 'stead', 'dry', 'wonder', 'laugh', 'thousand', 'ago', 'ran', 'check', 'game', 'shape', 'equate', 'hot', 'miss', 'brought', 'heat', 'snow', 'tire', 'bring', 'yes', 'distant', 'fill', 'east', 'paint', 'language', 'among', 'grand', 'ball', 'yet', 'wave', 'drop', 'heart', 'am', 'present', 'heavy', 'dance', 'position', 'arm', 'wide', 'sail', 'size', 'vary', 'settle', 'speak', 'ice', 'matter', 'circle', 'pair', 'include', 'divide', 'felt', 'perhaps', 'pick', 'sudden', 'count', 'square', 'reason', 'length', 'represent', 'art', 'hunt', 'bed', 'brother', 'egg', 'ride',  'believe',  'sit', 'race', 'window', 'store', 'summer',  'sleep', 'prove', 'lone', 'leg', 'exercise', 'wall', 'catch', 'mount', 'wish', 'sky', 'board', 'joy', 'winter', 'sat', 'written', 'wild','kept', 'glass', 'grass', 'cow', 'job', 'edge', 'sign', 'visit', 'past', 'soft', 'fun', 'bright', 'gas', 'weather', 'month', 'million', 'bear', 'finish', 'happy', 'hope', 'flower', 'clothe', 'strange', 'gone', 'jump', 'baby', 'eight', 'village', 'meet', 'root', 'buy', 'raise', 'metal', 'whether', 'push', 'seven', 'paragraph', 'third', 'shall', 'held', 'hair', 'describe', 'cook', 'floor', 'either', 'result', 'burn', 'hill', 'safe', 'cat', 'century', 'consider', 'type', 'law', 'bit', 'coast', 'copy', 'phrase', 'silent', 'sand', 'soil', 'roll',  'finger','lie', 'beat', 'excite', 'natural', 'view', 'sense', 'ear', 'else', 'quite', 'broke', 'case', 'kill', 'son', 'lake',  'child', 'milk',  'dress', 'cloud', 'surprise', 'quiet',  'climb', 'cool', 'poor', 'lot','skin', 'smile', 'crease', 'melody', 'row', 'exact',  'die', 'least', 'shout', 'wrote', 'seed', 'tone', 'join', 'suggest', 'clean', 'break', 'lady', 'yard', 'rise', 'bad', 'blow', 'oil', 'blood', 'touch', 'grew', 'cent', 'mix', 'team', 'wire', 'cost', 'lost', 'brown', 'wear', 'garden', 'equal', 'sent', 'choose', 'fell', 'fit', 'flow',  'bank', 'collect', 'save', 'control']
    english_punctuations = ['.',',','.',':',';','?','(',')','[',']','&','!','*','@','#','$','%','__','...','-','=','==','..','--',"'", "''"]
    #line = unicode(line, errors='ignore')
    tokens = [word.lower() for word in nltk.tokenize.word_tokenize(line.decode('cp1252', 'ignore').encode('utf-8','ignore'))]
    tokens = [word for word in tokens if not word in english_stopwords and not word in english_punctuations and not word.isdigit()]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    lemma = nltk.wordnet.WordNetLemmatizer()
    stems = [lemma.lemmatize(t) for t in filtered_tokens]
    #stems = [LancasterStemmer().stem(t) for t in filtered_tokens]
    stems = [word for word in stems if not word in english_stopwords and not word in english_punctuations and not word.isdigit()]
    return stems

def parse_all_files():
    topic_texts = []
    for i in range(990):
        try:
            quorafile = open("./glassdoorfile/glassdoorfile" + str(i) + ".txt", "r")
            topic_text = quorafile.readlines()
            topic_text = line_tokenize(str(topic_text))
            new_topic_text = []
            for j in range(len(topic_text)):
                if len(topic_text[j]) >= 2 and not re.search(r'[\\*]|[/*]|[*\\]|[*/]', topic_text[j]):
                    new_topic_text.append(topic_text[j])
            new_topic_text = str(new_topic_text).replace('\'', '').replace(',', ' ')
            # print new_topic_text[1:-1]
            topic_texts.append(str(topic_text))
            '''
            docs = nltk.pos_tag(topic_text)
            New_topic_text = []
            for item in docs:
                if item[1] == 'NN':
                    New_topic_text.append(item[0])
            topic_texts += str(New_topic_text)
            '''
        except:
            continue
    return topic_texts

from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts

def tfidf(topic_texts):
    ratefile = open('./glassdoorrate.txt', 'r')
    rates = []
    for item in ratefile:
        rate = re.findall(u'u\'\d.\d\'', item)
        for i in range(len(rate)):
            rates.append(rate[i][2:-1])

    vectorizer = CountVectorizer(stop_words=stopwords.words('english')+ ['li', 'div', 'br', 'u2022'])
    X = vectorizer.fit_transform(topic_texts)
    # print X.toarray()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    word_cloud = []
    for i in range(len(weight)):
        f = open('./top2_tfidf/' + string.zfill(i, 3) + '.txt', 'w+')
        # print weight[i]
        top2_weight = sorted(weight[i], reverse=True)[0:2]
        # print top2_weight
        bonding_word = ''
        flag = 0
        for j in range(len(word)):
            if weight[i][j] in top2_weight and flag == 0:
                f.write(word[j] + "	" + str(weight[i][j]) + "\n")
                bonding_word += word[j]
                first_word = word[j]
                flag = 1
            if weight[i][j] in top2_weight and flag == 1 and word[j] != first_word:
                f.write(word[j] + "	" + str(weight[i][j]) + "\n")
                bonding_word += '_' + word[j]
                #print bonding_word
        try:
            for k in range(int((float(rates[i])) * 10)):
                word_cloud.append(bonding_word)
        except:
            continue
    print "step2 done"
    tags = make_tags(get_tag_counts(str(word_cloud)), maxsize=50, minsize=5)
    create_tag_image(tags, './Glassdoor_TFIDF.png', size=(900, 600), fontname='Lobster')

def main():
    topic_texts = parse_all_files()
    print "step1 done"
    tfidf(topic_texts)
    print "step3 done"

if __name__ == "__main__":
    main()
