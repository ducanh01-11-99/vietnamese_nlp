import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle
from underthesea import word_tokenize
import re
from sklearn.metrics import classification_report
import time


def read_lines_to_list(filename):
    """Reads lines from a file and stores them as elements in a list.

    Args:
        filename (str): The path to the file containing the lines.

    Returns:
        list: The list of lines from the file.
    """

    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove trailing newline characters
    return [line.rstrip() for line in lines]


# Hàm tiền xử lý văn bản
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = word_tokenize(text, format="text")
    words = [word for sentence in text for word in sentence]  # Flatten list
    # Loại bỏ stopword
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)


stopwords = read_lines_to_list('vietnamese-stopwords.txt')
stopwords = set(stopwords)

label_file = 'label.txt'
text_file = 'text.txt'
texts = read_lines_to_list(text_file)
labels = read_lines_to_list(label_file)

data = {
    'text': texts,
    'label': labels,
}

# Tạo DataFrame từ từ điển dữ liệu
df = pd.DataFrame(data)

# Chuyển đổi nhãn thành số (pos: 0, neg: 1, neu: 2)
label_mapping = {'pos': 0, 'neg': 2, 'neu': 1}

df['label'] = df['label'].map(label_mapping)

# Tiền xử lý văn bản
df['text'] = df['text'].apply(clean_text)

# Chia tập dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Chuyển đổi nhãn thành dạng one-hot encoding
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Sử dụng tokenizer để biến văn bản thành các chỉ số số học
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Độ dài tối đa của mỗi sequence
maxlen = 100

# Padding các sequence để có cùng độ dài
X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen)
X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen)

# Tạo mô hình
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=maxlen))
model.add(LSTM(units=128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='softmax'))  # 3 đơn vị cho 3 lớp và activation là softmax

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Hiển thị kiến trúc mô hình
model.summary()
# Bắt đầu đo thời gian huấn luyện
start_train_time = time.time()

# Huấn luyện mô hình
history = model.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_split=0.2)
# Dự đoán nhãn của dữ liệu kiểm tra
start_predict_time = time.time()
y_pred = model.predict(X_test_padded)
end_predict_time = time.time()


# Kết thúc đo thời gian huấn luyện
end_train_time = time.time()

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')


y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# In báo cáo phân loại
print("\nClassification Report:\n", classification_report(y_test_classes, y_pred_classes, target_names=['pos', 'neg', 'neu']))

# Lưu mô hình
model.save('text_classification_lstm_3class.h5')

# Tính toán tổng thời gian huấn luyện và thời gian dự đoán một bản ghi
total_train_time = end_train_time - start_train_time
predict_time_per_sample = (end_predict_time - start_predict_time) / X_test.shape[0]

print("Total training time:", total_train_time, "seconds")
print("Prediction time for one sample:", predict_time_per_sample, "seconds")

# Lưu tokenizer để sử dụng sau này
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
