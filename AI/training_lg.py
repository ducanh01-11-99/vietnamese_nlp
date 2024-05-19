#Hồi quy Logistic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
from underthesea import word_tokenize
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

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tiền xử lý dữ liệu văn bản
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print('xasdasd', X_test)

# Khởi tạo mô hình Logistic Regression
model = LogisticRegression()

# Bắt đầu đo thời gian huấn luyện
start_train_time = time.time()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán nhãn của dữ liệu kiểm tra
start_predict_time = time.time()
y_pred = model.predict(X_test)
end_predict_time = time.time()

# Kết thúc đo thời gian huấn luyện
end_train_time = time.time()

# Đánh giá mô hình
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Tính toán tổng thời gian huấn luyện và thời gian dự đoán một bản ghi
total_train_time = end_train_time - start_train_time
predict_time_per_sample = (end_predict_time - start_predict_time) / X_test.shape[0]

print("Total training time:", total_train_time, "seconds")
print("Prediction time for one sample:", predict_time_per_sample, "seconds")