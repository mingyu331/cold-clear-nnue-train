# serialization and deserialization of training files and nnue files
import json
def load_train(path):
    with open(path) as f:
        train_data = json.load(f)
    return train_data

def edit_data(path):
    with open(path, 'r') as f:
        n = f.read()
        with open(path + '.edited.txt', 'w') as f2:
            f2.write('['+n[:-1].replace('\n', ', \n')+']')

edit_data('./train_data/out_high.txt')
data = load_train('./train_data/out_high.txt.edited.txt')
print(data)