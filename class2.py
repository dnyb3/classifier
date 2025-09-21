import re
import pandas as pd
from setfit import SetFitModel, SetFitTrainer, SetFitTrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# ==================================================
# 1. Prepare training data
# ==================================================
data = pd.read_csv("training_data/model_cases7.csv")
data = data.dropna(subset=["type"])
data["label_binary"] = (data["type"] > 0).astype(int)

train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df[["passage", "label_binary"]])
test_dataset  = Dataset.from_pandas(test_df[["passage", "label_binary"]])

# ==================================================
# 2. Train Binary Advice Detector with SetFit
# ==================================================
binary_model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

binary_args = SetFitTrainingArguments(
    batch_size=16,
    num_epochs=5,   # increase if underfitting
    learning_rate=2e-5,
    seed=42,
)

binary_trainer = SetFitTrainer(
    model=binary_model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_class="CosineSimilarityLoss",  # contrastive loss
    batch_size=16,
    num_epochs=5,
)

binary_trainer.train()
metrics = binary_trainer.evaluate()
print("Binary metrics:", metrics)

binary_model.save_pretrained("models/binary_setfit")

# ==================================================
# 3. Train Multi-Class Advice Type Model
# ==================================================
train_multi = Dataset.from_pandas(train_df[["passage", "type"]])
test_multi  = Dataset.from_pandas(test_df[["passage", "type"]])

multi_model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", num_labels=4)

multi_trainer = SetFitTrainer(
    model=multi_model,
    train_dataset=train_multi,
    eval_dataset=test_multi,
    loss_class="CosineSimilarityLoss",
    batch_size=16,
    num_epochs=5,
)

multi_trainer.train()
metrics = multi_trainer.evaluate()
print("Multi metrics:", metrics)

multi_model.save_pretrained("models/multi_setfit")

# ==================================================
# 4. Inference Pipeline on New Call CSV
# ==================================================
def extract_snippets(agent_text: str) -> list[str]:
    """Apply regex to grab candidate advice segments from full rep text."""
    pattern = re.compile(
        r"(?:\S+\s+){0,40}(?:recommend|recommends|recommended|recommendation|recommending|"
        r"suggest|suggestion|suggests|suggested|suggesting|propose|proposes|proposed|proposal|"
        r"advice|urge|you should|best bet|advise|guidance|best option|i would|we would|"
        r"should consider|instruct|my opinion|encourage|my input|our input|direction|"
        r"assessment|conclusion|judgement|my feeling|my reaction|our feeling|our reaction|"
        r"best interest)(?:\s+\S+){0,120}(?:\S+\s+){0,120}"
    )
    return pattern.findall(agent_text.lower())

def run_inference(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    binary_model = SetFitModel.from_pretrained("models/binary_setfit")
    multi_model = SetFitModel.from_pretrained("models/multi_setfit")

    results = []

    for _, row in df.iterrows():
        call_id = row["id"]
        agent_text = str(row["agent_text"])
        snippets = extract_snippets(agent_text)

        call_scores = []
        for snip in snippets:
            # binary stage
            binary_pred = binary_model.predict_proba([snip])[0][1]  # prob of "advice"
            if binary_pred > 0.5:
                # multi stage
                multi_probs = multi_model.predict_proba([snip])[0]
                pred_label = multi_probs.argmax()
                pred_score = multi_probs[pred_label]
                call_scores.append((binary_pred, pred_label, pred_score, snip))

        if call_scores:
            # take max scoring snippet
            top = max(call_scores, key=lambda x: x[2])
            results.append({
                "id": call_id,
                "binary_score": top[0],
                "pred_label": top[1],
                "pred_score": top[2],
                "snippet": top[3],
            })

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

# Example:
# run_inference("calls.csv", "scored_calls.csv")
