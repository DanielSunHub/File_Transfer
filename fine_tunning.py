import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的SQLCoder模型和分词器
model_name = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# 读取数据集
data = pd.read_csv("train_dataset.csv")
dataset = Dataset.from_pandas(data)

# 数据预处理
def preprocess_function(examples):
    inputs = [ex for ex in examples["input"]]
    targets = [ex for ex in examples["output"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # 根据模型大小调整batch size
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,  # 设置epoch为2
    fp16=True,  # 使用混合精度训练以提高性能
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,  # 添加tokenizer以确保生成评估的正确性
)

# 开始微调
trainer.train()

# 保存模型和分词器
trainer.save_model("./results")
tokenizer.save_pretrained("./results")

# 加载微调后的模型和分词器
model = AutoModelForCausalLM.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("./results")
