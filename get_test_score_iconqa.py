import os
import json
import sys

def compute_accuracy(path):
    with open(path, 'r') as f:
        data = json.load(f)

    correct_answers = 0
    total_questions = len(data)

    for item in data:
        if item['pred_ans'] == item['gt_ans']:
            correct_answers += 1

    return correct_answers / total_questions

def find_latest_subdir(base_dir):
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir

def save_accuracy_to_json(path, accuracy):
    with open(path, 'w') as f:
        json.dump({"test_accuracy": accuracy}, f, indent=4)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <int_value>")
        sys.exit(1)

    int_value = int(sys.argv[1])
    # TODO: Fix this base_path to each local environment
    base_path = f"/input/results/iconqa/iconqa_{int_value}"
    latest_dir = find_latest_subdir(base_path)
    json_path = os.path.join(latest_dir, "result/test_iconqa_result.json")

    if os.path.exists(json_path):
        accuracy = compute_accuracy(json_path)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Save accuracy to a new JSON file in the same directory
        accuracy_json_path = os.path.join(latest_dir, "result/test_accuracy.json")
        save_accuracy_to_json(accuracy_json_path, accuracy)

    else:
        print(f"JSON file not found at {json_path}")
