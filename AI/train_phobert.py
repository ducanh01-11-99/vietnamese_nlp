import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from underthesea import word_tokenize
from sklearn.metrics import classification_report
import time
import os


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
    return text


label_file = 'label.txt'
text_file = 'text.txt'
texts = read_lines_to_list(text_file)
labels = read_lines_to_list(label_file)

# Tiền xử lý văn bản
texts = [clean_text(text) for text in texts]

# Tải tokenizer và mô hình PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)

# Mã hóa dữ liệu văn bản
encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(encoded_inputs["input_ids"], labels, test_size=0.2, random_state=42)
train_masks, test_masks, _, _ = train_test_split(encoded_inputs["attention_mask"], labels, test_size=0.2,
                                                 random_state=42)

# Chuyển đổi nhãn thành các số nguyên
label_map = {"pos": 0, "neu": 1, "neg": 2}
train_labels = torch.tensor([label_map[label] for label in y_train])
test_labels = torch.tensor([label_map[label] for label in y_test])

# Tạo TensorDataset cho tập huấn luyện và kiểm tra
train_data = TensorDataset(X_train, train_masks, train_labels)
test_data = TensorDataset(X_test, test_masks, test_labels)

# Tạo DataLoader cho tập huấn luyện và kiểm tra
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=8)
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=8)

# Đặt mô hình vào chế độ huấn luyện
model.train()

# Đặt optimizer và scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * 3  # Số epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

start_time = time.time()
# Vòng lặp huấn luyện
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    total_loss = 0
    model.train()

    for batch in train_dataloader:
        print('...training')
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()

        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.2f}")

end_time = time.time()
print("Training complete.")

# Đặt mô hình vào chế độ đánh giá
model.eval()

predictions, true_labels = [], []

# Tính toán thời gian dự đoán mỗi bản ghi

start_time_predict = time.time()
with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_labels = batch

        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        predictions.append(logits)
        true_labels.append(b_labels)

end_time_predict = time.time()

# Chuyển đổi logits và labels thành các tensor
flat_predictions = torch.cat(predictions, dim=0).argmax(dim=1).cpu().numpy()
flat_true_labels = torch.cat(true_labels, dim=0).cpu().numpy()

# Tính độ chính xác
accuracy = accuracy_score(flat_true_labels, flat_predictions)
print(f"Accuracy: {accuracy:.2f}")

# In báo cáo phân loại
print("\nClassification Report:\n", classification_report(flat_true_labels, flat_predictions, target_names=['pos', 'neu', 'neg']))

# Tính toán tổng thời gian huấn luyện và thời gian trung bình dự đoán mỗi bản ghi
total_train_time = end_time - start_time
predict_time_per_sample = (end_time_predict - start_time_predict) / X_test.shape[0]

print("Total training time:", total_train_time, "seconds")
print("Prediction time for one sample:", predict_time_per_sample, "seconds")

# Đường dẫn đến thư mục lưu mô hình

# Tạo thư mục lưu trữ mô hình nếu nó không tồn tại
output_dir = './phoBERT'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lưu mô hình và tokenizer vào thư mục
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Model saved successfully.")