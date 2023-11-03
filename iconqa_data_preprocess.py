import json

with open('problems.json', 'r') as f:
    data = json.load(f)

train, test, val = [], [], []

for key, value in data.items():
    split = value["split"]
    ques_type = value["ques_type"]
    if ques_type == "choose_txt":
        data = value
        data['id'] = key
        if split == "train" :
            train.append(data)
        elif split == "test":
            test.append(data)
        elif split == "val":
            val.append(data)

with open('/input/iconqa/annotations/train.json', 'w') as train_file:
    json.dump(train, train_file, ensure_ascii=False, indent=4)

with open('/input/iconqa/annotations/test.json', 'w') as test_file:
    json.dump(test, test_file, ensure_ascii=False, indent=4)

with open('/input/iconqa/annotations/val.json', 'w') as val_file:
    json.dump(val, val_file, ensure_ascii=False, indent=4)