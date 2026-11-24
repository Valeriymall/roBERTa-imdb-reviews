from pathlib import Path
import numpy as np, evaluate, torch
from datasets import load_dataset
import matplotlib
matplotlib.use("Agg")                    # draw to a file, works without a display
import matplotlib.pyplot as plt
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

BASE = Path(__file__).resolve().parent   #the BERT folder, independent of where you launch from
OUT = BASE / "results" / "imdb_roberta"   #checkpoints, final model and plot go here
DATA = BASE / "data"                     #dataset cache goes here

print("GPU:", torch.cuda.get_device_name(0))

ds = load_dataset("stanfordnlp/imdb", cache_dir=str(DATA))   #loads imdb dataset, 25k + 25k labeled training/test reviews
split = ds["train"].train_test_split(test_size=0.1, seed=42)  #gives dict with random train/test split
train, val, test = split["train"], split["test"], ds["test"]

MODEL = "roberta-base"
tok = AutoTokenizer.from_pretrained(MODEL)  #tokenization based on roberta-base
enc = lambda b: tok(b["text"], truncation=True, max_length=512)  
train, val, test = (d.map(enc, batched=True) for d in (train, val, test)) #applies to chunk of many "reviews" at once

train_sub = train.select(range(2000))                 # train is already shuffled by the split
test_sub = test.shuffle(seed=0).select(range(5000))   # must shuffle: the IMDB test set is sorted by label


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2)  #builds roBERTa base and loads pretrained weights

acc = evaluate.load("accuracy")
def metrics(p):
    return acc.compute(predictions=np.argmax(p.predictions, -1),  #picks higher scoring class after evaluation pass
                       references=p.label_ids)

args = TrainingArguments(
    output_dir=str(OUT),
    learning_rate=2e-5,
    per_device_train_batch_size=16,   #"reviews" per gradient step
    gradient_accumulation_steps=1,   #if not enough vram: set per_device_train_batch_size = 8, this to 2.
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.01,
    warmup_steps=0.1,
    eval_strategy="steps",
    eval_steps=201,                  #7 evaluations per epoch
    save_strategy="steps",           #must match eval_strategy
    save_steps=201,
    save_only_model=True,            #skip optimizer state, smaller checkpoints
    save_total_limit=2,              #keep only the best and latest checkpoint
    load_best_model_at_end=True,
    train_sampling_strategy="group_by_length", #since imdb review lengths vary, this minimizes unnecessary padding
    metric_for_best_model="val_accuracy",  #pick the best checkpoint by validation accuracy only
    bf16=True,                             #consider fp16 if your hardware does not supprot bf16
    report_to="none",  
)

lengths = [len(tok(t)["input_ids"]) for t in ds["test"]["text"][:5000]]  #counting review length in database 
print("over 256:", np.mean(np.array(lengths) > 256))
print("over 512:", np.mean(np.array(lengths) > 512))

trainer = Trainer(model=model, args=args, train_dataset=train,   #pre written training loop!
                  eval_dataset={"train": train_sub, "val": val, "test": test_sub},  #three sets, each evaluated every 201 steps
                  data_collator=DataCollatorWithPadding(tok), #pad shorter "reviews" to length of longest in batch
                  compute_metrics=metrics)
trainer.train()
final = trainer.evaluate(test)           #full 25k test set, final number
print("TEST:", final)
trainer.save_model(str(OUT / "final"))
tok.save_pretrained(OUT / "final")       #so the model can be reloaded later with its tokenizer



### PLOTS ###

log = trainer.state.log_history

def curve(name):
    # each eval set is logged as its own entry, so collect them separately
    pts = [(h["epoch"], h[f"eval_{name}_accuracy"]) for h in log
           if f"eval_{name}_accuracy" in h]
    assert pts, f"no eval_{name}_accuracy in log"
    return zip(*pts)

plt.figure(figsize=(7, 4.5))
plt.plot(*curve("train"), marker="o", label="train (2k subset)")
plt.plot(*curve("val"), marker="o", label="validation")
plt.plot(*curve("test"), marker="o", label="test (5k subset)")
plt.axhline(final["eval_accuracy"], color="gray", linestyle="--",
            label=f"final test, full 25k ({final['eval_accuracy']:.4f})")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("RoBERTa on IMDB")
plt.ylim(0.80, 1.0)
plt.grid(alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "accuracy.png", dpi=150)
print("saved plot to", OUT / "accuracy.png")