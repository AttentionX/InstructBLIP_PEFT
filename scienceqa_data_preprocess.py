import json
from tqdm import tqdm
    
with open("scienceqa_problems_path.json", 'r') as file:
    data = json.load(file)
    
with open("scienceqa_pid_splits.json") as file:
    pid_splits = json.load(file)

train_ids = pid_splits['train']
val_ids = pid_splits['val']
test_ids = pid_splits['test']

# make train annotation

train_annotation = []
for id in tqdm(train_ids):
    train_data = data[str(id)]
    if train_data['image'] is None:
        continue
    image_url = f"scienceqa/images/train/{id}/image.png"
    if train_data['answer'] == 0:
        answer = "(a) " + train_data['choices'][train_data['answer']]
    elif train_data['answer'] == 1:
       answer = "(b) " + train_data['choices'][train_data['answer']]
    elif train_data['answer'] == 2:
       answer = "(c) " + train_data['choices'][train_data['answer']]
    elif train_data['answer'] == 3:
        answer = "(d) " + train_data['choices'][train_data['answer']]
    else:
        answer = "(e) " + train_data['choices'][train_data['answer']]
    ann = {
        "image": image_url,
        "question": train_data['question'],
        "answer" : answer,
        "choices": train_data['choices'],
        "context" : train_data['hint'] + " " + train_data['lecture'],
        "question_id" : id
    }
    train_annotation.append(ann)

# make val annotation

val_annotation = []
for id in tqdm(val_ids):
    val_data = data[str(id)]
    if val_data['image'] is None:
        continue
    image_url = f"scienceqa/images/val/{id}/image.png"
    if val_data['answer'] == 0:
        answer = "(a) " + val_data['choices'][val_data['answer']]
    elif val_data['answer'] == 1:
       answer = "(b) " + val_data['choices'][val_data['answer']]
    elif val_data['answer'] == 2:
       answer = "(c) " + val_data['choices'][val_data['answer']]
    elif val_data['answer'] == 3:
       answer = "(d) " + val_data['choices'][val_data['answer']]
    else:
        answer = "(e) " + val_data['choices'][val_data['answer']]
    ann = {
        "image": image_url,
        "question": val_data['question'],
        "answer" : answer,
        "choices": val_data['choices'],
        "context" : val_data['hint']+ " " + val_data['lecture'],
        "question_id" : id
    }
    val_annotation.append(ann)
    
# make test annotation

test_annotation = []
for id in tqdm(test_ids):
    test_data = data[str(id)]
    if test_data['image'] is None:
        continue
    image_url = f"scienceqa/images/test/{id}/image.png"
    if test_data['answer'] == 0:
        answer = "(a) " + test_data['choices'][test_data['answer']]
    elif test_data['answer'] == 1:
       answer = "(b) " + test_data['choices'][test_data['answer']]
    elif test_data['answer'] == 2:
       answer = "(c) " + test_data['choices'][test_data['answer']]
    elif test_data['answer'] == 3:
       answer = "(d) " + test_data['choices'][test_data['answer']]
    else:
        answer = "(e) " + test_data['choices'][test_data['answer']]
    ann = {
        "image": image_url,
        "question": test_data['question'],
        "answer" : answer,
        "choices": test_data['choices'],
        "context" :test_data['hint']+ " " + test_data['lecture'],
        "question_id" : id
    }
    test_annotation.append(ann)

with open("/input/scienceqa/scienceqa_train.json", 'w') as file:
    json.dump(train_annotation, file)

with open("/input/scienceqa/scienceqa_test.json", 'w') as file:
    json.dump(test_annotation, file)

with open("/input/scienceqa/scienceqa_val.json", 'w') as file:
    json.dump(val_annotation, file)