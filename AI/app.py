from underthesea import word_tokenize
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import keras

stop_word_arr = []
dict = []

with open("vietnamese-stopwords.txt", "r", encoding="UTF-8") as fi:
    stop_word = fi.readlines()
    for j in stop_word:
        j = j.replace('\n','').lower()
        stop_word_arr.append(j)

tf_vectorizer = TfidfVectorizer(ngram_range=(1,4),max_df=0.8,max_features=15000, encoding='utf-8')

bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)


def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
        else:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    return ''.join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def chuan_hoa_dau_cau_tieng_viet(sentence):
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\\p{P}*)([p{L}.]*\\p{L}+)(\\p{P}*$)', r'\1/\2/\3', word).split('/')
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    # print(' '.join(words))
    return ' '.join(words)

with open('label.txt', 'r') as f:
    # Read the contents of the file as a string
    text = f.read()
# Split the string into lines using the newline character as the delimiter
lines = text.splitlines()

# Convert each line into an element of the list
label1 = []
for line in lines:
    # Option 1: Keep the line as a string
    label1.append(line)

with open('text.txt', 'r') as f:
    # Read the contents of the file as a string
    text = f.read()
# Split the string into lines using the newline character as the delimiter
lines = text.splitlines()

# Convert each line into an element of the list
text1 = []
for line in lines:
    # Option 1: Keep the line as a string
    text1.append(line)

test_percent=0.15

X_train, X_test, y_train, y_test = train_test_split(text1, label1, test_size=test_percent, random_state=42)

# encode label
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

tf_vectorizer.fit_transform(X_train)
tf_vectorizer.transform(X_test)

MODEL_PATH='./engsent'

if __name__ == '__main__':
    inIn = "phục vụ không chuyên nghiệp, đồ ăn không ngon, quán chất lượng kém"
    inIn = inIn.lower()
    inIn = chuan_hoa_dau_cau_tieng_viet(inIn.strip())
    string_temp = ''
    tokenize_sent = word_tokenize(inIn, format="text")
    for word in tokenize_sent:
        string_temp += word
    string_temp = string_temp.strip()
    inIn_vectorizer = tf_vectorizer.transform([string_temp])

    # nb_model1 = pickle.load(open(os.path.join(MODEL_PATH,"linear_classifier.pkl"), 'rb'))
    # getthing1 = nb_model1.predict([string_temp])
    # res = label_encoder.inverse_transform(getthing1)[0],

    # nb_model2 = pickle.load(open(os.path.join(MODEL_PATH,"naive_bayes.pkl"), 'rb'))
    # getthing2 = nb_model2.predict(inIn_vectorizer)
    # res2 = label_encoder.inverse_transform(getthing2)[0],

    # nb_model3 = pickle.load(open(os.path.join(MODEL_PATH,"svm.pkl"), 'rb'))
    # getthing3 = nb_model3.predict([string_temp])
    # res3 = label_encoder.inverse_transform(getthing3)[0],

    model4 = keras.models.load_model('AI/lstm_model.h5')
    # load tokenizer object
    with open('AI/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
            # parameters
    result = model4.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inIn]),
                                                truncating='post', maxlen=20))
    lstm=label_encoder.inverse_transform([np.argmax(result)])[0],
    # em.append(get_label)

    # result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([i]),
    #                                                                   truncating='post', maxlen=20))
    # get_label=label_encoder.inverse_transform([np.argmax(result)])[0]
    # em.append(get_label)
    print('predict', lstm)
    # app.run(debug=True)