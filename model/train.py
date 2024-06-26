import fasttext
import json

def fit(train_set_path):
    return fasttext.train_supervised(input=train_set_path, wordNgrams=2, epoch=20, lr=0.1, dim=300) 

def predict(text, classifier, k=3):
    """
    Returns:
        list: [[id, author name, percentage], ...]
    """
    labels, probs = classifier.predict(text, k = k)
    label2ind = json.load(open('data/files/label2ind.json', 'r', encoding='utf-8'))
    result = []
    for i in range(k):
        id = labels[i][9:]
        name = next(key for key, val in label2ind.items() if val == labels[i])
        result.append([id,name, round(probs[i] * 100, 2)])
    return result

if __name__ == '__main__':
    # Chinese:
    classifier = fit('data/files/train_C.txt')
    result = classifier.test('data/files/train_C.txt')
    print("Chinese model results: ")
    print('P@1:', result[1])
    print('R@1:', result[2])
    print('Number of examples:', result[0])
    classifier.save_model('data/classifier_C.model')

    # Foreign:
    classifier = fit('data/files/train_F.txt')
    result = classifier.test('data/files/train_F.txt')
    print("Foreign model results: ")
    print('P@1:', result[1])
    print('R@1:', result[2])
    print('Number of examples:', result[0])
    classifier.save_model('data/classifier_F.model')