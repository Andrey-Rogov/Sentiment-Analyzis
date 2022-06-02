import numpy as np
from gensim.models import KeyedVectors
import re
import gensim
import random
from gensim.models.doc2vec import TaggedDocument

sw_nltk = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
           'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
           'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
           'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
           'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
           'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
           'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
           'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
           'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
           'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
           'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
           "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
           'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
           'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


def create_train_test_data():
    x_train_negative = ''
    # Train negative
    for i in range(12500):
        try:
            with open(f'reviews_data/movies/raw/train/neg/{i}_1.txt', encoding='utf-8') as f:
                x_train_negative += f.read() + ' '
        except FileNotFoundError:
            try:
                with open(f'reviews_data/movies/raw/train/neg/{i}_2.txt', encoding='utf-8') as f:
                    x_train_negative += f.read() + ' '
            except FileNotFoundError:
                try:
                    with open(f'reviews_data/movies/raw/train/neg/{i}_3.txt', encoding='utf-8') as f:
                        x_train_negative += f.read() + ' '
                except FileNotFoundError:
                    with open(f'reviews_data/movies/raw/train/neg/{i}_4.txt', encoding='utf-8') as f:
                        x_train_negative += f.read() + ' '
    # Train positive
    x_train_positive = ''
    for i in range(12500):
        try:
            with open(f'reviews_data/movies/raw/train/pos/{i}_7.txt', encoding='utf-8') as f:
                x_train_positive += f.read() + ' '
        except FileNotFoundError:
            try:
                with open(f'reviews_data/movies/raw/train/pos/{i}_8.txt', encoding='utf-8') as f:
                    x_train_positive += f.read() + ' '
            except FileNotFoundError:
                try:
                    with open(f'reviews_data/movies/raw/train/pos/{i}_9.txt', encoding='utf-8') as f:
                        x_train_positive += f.read() + ' '
                except FileNotFoundError:
                    with open(f'reviews_data/movies/raw/train/pos/{i}_10.txt', encoding='utf-8') as f:
                        x_train_positive += f.read() + ' '
    x_test_negative = ''
    # Test negative
    for i in range(12500):
        try:
            with open(f'reviews_data/movies/raw/test/neg/{i}_1.txt', encoding='utf-8') as f:
                x_test_negative += f.read() + ' '
        except FileNotFoundError:
            try:
                with open(f'reviews_data/movies/raw/test/neg/{i}_2.txt', encoding='utf-8') as f:
                    x_test_negative += f.read() + ' '
            except FileNotFoundError:
                try:
                    with open(f'reviews_data/movies/raw/test/neg/{i}_3.txt', encoding='utf-8') as f:
                        x_test_negative += f.read() + ' '
                except FileNotFoundError:
                    with open(f'reviews_data/movies/raw/test/neg/{i}_4.txt', encoding='utf-8') as f:
                        x_test_negative += f.read() + ' '
    # Test positive
    x_test_positive = ''
    for i in range(12500):
        try:
            with open(f'reviews_data/movies/raw/test/pos/{i}_7.txt', encoding='utf-8') as f:
                x_test_positive += f.read() + ' '
        except FileNotFoundError:
            try:
                with open(f'reviews_data/movies/raw/test/pos/{i}_8.txt', encoding='utf-8') as f:
                    x_test_positive += f.read() + ' '
            except FileNotFoundError:
                try:
                    with open(f'reviews_data/movies/raw/test/pos/{i}_9.txt', encoding='utf-8') as f:
                        x_test_positive += f.read() + ' '
                except FileNotFoundError:
                    with open(f'reviews_data/movies/raw/test/pos/{i}_10.txt', encoding='utf-8') as f:
                        x_test_positive += f.read() + ' '
    return x_train_negative, x_train_positive, x_test_negative, x_test_positive


def preprocessing(text):
    pattern = r'[!?:;\"0-9()<>\\/_\#,.]'  # ' -
    text = [[word.lower() for word in sentence if word not in sw_nltk] for sentence in text]
    for sentence in range(len(text)):
        for word in range(len(text[sentence])):
            if text[sentence][word].endswith(',') or text[sentence][word].endswith('.'):
                text[sentence][word] = text[sentence][word][:-1]
            if re.search(pattern, text[sentence][word]):
                text[sentence][word] = re.sub(pattern, '', text[sentence][word])
            # In this dataset there are many tags <br> can pop up, so delete them
            if text[sentence][word] == 'br':
                text[sentence][word] = ''
        while '' in text[sentence]:
            useless = text[sentence].index('')
            text[sentence] = text[sentence][:useless] + text[sentence][useless + 1:]
    return text


def read_data():
    X_train, X_test, y_train, y_test = [], [], [], []
    with open('reviews_data/amazon/raw/train_5000.txt', 'r') as file:
        data = file.read().split('\n')
        for review in data:
            if review[9] == '1':
                X_train.append(review[11:])
                y_train.append(0)
            else:
                X_train.append(review[11:])
                y_train.append(1)
    with open('reviews_data/amazon/raw/test_5000.txt', 'r') as file:
        data = file.read().split('\n')
        for review in data:
            if review[9] == '1':
                X_test.append(review[11:])
                y_test.append(0)
            else:
                X_test.append(review[11:])
                y_test.append(1)
    return X_train, X_test, y_train, y_test


def compute_movies_sentences():
    X_train_neg, X_train_pos, X_test_neg, X_test_pos = create_train_test_data()
    X_train_pos = [sent.split() for sent in X_train_pos.split('.')]
    X_train_neg = [sent.split() for sent in X_train_neg.split('.')]
    X_test_pos = [sent.split() for sent in X_test_pos.split('.')]
    X_test_neg = [sent.split() for sent in X_test_neg.split('.')]

    X_train_pos = [[sent, 1] for sent in preprocessing(X_train_pos) if sent]
    X_train_neg = [[sent, 0] for sent in preprocessing(X_train_neg) if sent]
    X_test_pos = [[sent, 1] for sent in preprocessing(X_test_pos) if sent]
    X_test_neg = [[sent, 0] for sent in preprocessing(X_test_neg) if sent]

    X_train = X_train_pos + X_train_neg
    random.shuffle(X_train)
    y_train = [class_ for class_ in np.array(X_train, dtype=list)[:, 1]]
    for sentence in range(len(X_train)):  # Delete marks 0/1 from X data
        X_train[sentence] = X_train[sentence][0]
    X_test = X_test_pos + X_test_neg
    random.shuffle(X_test)
    y_test = [class_ for class_ in np.array(X_test, dtype=list)[:, 1]]
    for sentence in range(len(X_test)):
        X_test[sentence] = X_test[sentence][0]
    with open('reviews_data/movies/words/X_train.txt', 'w', encoding='utf-8') as f:
        f.write(str(X_train))
    with open('reviews_data/movies/words/Y_train.txt', 'w', encoding='utf-8') as f:
        f.write(str(y_train))
    with open('reviews_data/movies/words/X_test.txt', 'w', encoding='utf-8') as f:
        f.write(str(X_test))
    with open('reviews_data/movies/words/Y_test.txt', 'w', encoding='utf-8') as f:
        f.write(str(y_test))
    return X_train, X_test, y_train, y_test


def compute_amazon_sentences():
    X_train, X_test, y_train, y_test = read_data()
    X_train = preprocessing([i.split() for i in X_train])
    X_test = preprocessing([i.split() for i in X_test])
    with open('reviews_data/amazon/words/X_train.txt', 'w', encoding='utf-8') as f:
        f.write(str(X_train))
    with open('reviews_data/amazon/words/Y_train.txt', 'w', encoding='utf-8') as f:
        f.write(str(y_train))
    with open('reviews_data/amazon/words/X_test.txt', 'w', encoding='utf-8') as f:
        f.write(str(X_test))
    with open('reviews_data/amazon/words/Y_test.txt', 'w', encoding='utf-8') as f:
        f.write(str(y_test))
    return X_train, X_test, y_train, y_test


def download_movies_sentences(return_vectors=True):
    if return_vectors:
        X_train = np.loadtxt('reviews_data/movies/vectors/X_train_vectors.txt', usecols=range(100), delimiter=',',
                             ndmin=2, dtype=np.float32)
        X_test = np.loadtxt('reviews_data/movies/vectors/X_test_vectors.txt', usecols=range(100), delimiter=',',
                            ndmin=2, dtype=np.float32)
    else:
        with open('reviews_data/movies/words/X_train.txt', 'r', encoding='utf-8') as f:
            X_train = eval(f.read())
        with open('reviews_data/movies/words/X_test.txt', 'r', encoding='utf-8') as f:
            X_test = eval(f.read())
    with open('reviews_data/movies/words/Y_train.txt', 'r', encoding='utf-8') as f:
        y_train = np.array(eval(f.read()))
    with open('reviews_data/movies/words/Y_test.txt', 'r', encoding='utf-8') as f:
        y_test = np.array(eval(f.read()))
    return X_train, X_test, y_train, y_test


def download_amazon_sentences(return_vectors=True):
    if return_vectors:
        X_train = np.loadtxt('reviews_data/amazon/vectors/X_train_vectors.txt', usecols=range(100), delimiter=',',
                             ndmin=2, dtype=np.float32)
        X_test = np.loadtxt('reviews_data/amazon/vectors/X_test_vectors.txt', usecols=range(100), delimiter=',',
                            ndmin=2, dtype=np.float32)
    else:
        with open('reviews_data/amazon/words/X_train.txt', 'r', encoding='utf-8') as f:
            X_train = eval(f.read())
        with open('reviews_data/amazon/words/X_test.txt', 'r', encoding='utf-8') as f:
            X_test = eval(f.read())

    with open('reviews_data/amazon/words/Y_train.txt', 'r', encoding='utf-8') as f:
        y_train = np.array(eval(f.read()))
    with open('reviews_data/amazon/words/Y_test.txt', 'r', encoding='utf-8') as f:
        y_test = np.array(eval(f.read()))
    return X_train, X_test, y_train, y_test


def movies_d2v_model(return_pretrained=True, **kwargs):
    """
    If return_pretrained=False, X_train must be passed as kw-argument
    """
    if return_pretrained:
        model = gensim.models.Doc2Vec.load("reviews_data/movies/d2v_model/doc2vec_model")
        return model
    else:
        tagged_train_data = [TaggedDocument(d, [i]) for i, d in enumerate(kwargs['X_train'])]

        model = gensim.models.Doc2Vec(vector_size=100, window=2, min_count=1, workers=4)
        model.build_vocab(tagged_train_data)
        model.train(tagged_train_data, total_examples=model.corpus_count, epochs=100)
        model.save("reviews_data/movies/d2v_model/doc2vec_model")
        return model


def amazon_d2v_model(return_pretrained=True, **kwargs):
    """
    If return_pretrained=False, X_train must be passed as kw-argument
    """
    if return_pretrained:
        model = gensim.models.Doc2Vec.load("reviews_data/amazon/d2v_model/doc2vec_model")
        return model
    else:
        tagged_train_data = [TaggedDocument(d, [i]) for i, d in enumerate(kwargs['X_train'])]

        model = gensim.models.Doc2Vec(vector_size=100, window=2, min_count=1, workers=4)
        model.build_vocab(tagged_train_data)
        model.train(tagged_train_data, total_examples=model.corpus_count, epochs=100)
        model.save("reviews_data/amazon/d2v_model/doc2vec_model")
        return model


def create_vectors(dataset, model, X_test):
    """
    dataset parameter should be 'movie' or 'amazon', depends on the dataset used
    """
    if dataset == 'movie':
        vectors = [model.dv[vector] for vector in range(len(model.dv))]
        new_array = np.array(vectors)
        train_vectors = np.savetxt('reviews_data/movie/vectors/X_train_vectors.txt', new_array, delimiter=',')

        for sent_num in range(
                len(X_test)):  # ~ 32 minutes (on 100 epochs) ~ 18 minutes (on 50 epochs) ~ 4 minutes (on 10 epochs)
            X_test[sent_num] = model.infer_vector(X_test[sent_num], epochs=100)

        new_array = np.array(X_test)
        test_vectors = np.savetxt('reviews_data/movie/vectors/X_test_vectors.txt', new_array, delimiter=',')
        return train_vectors, test_vectors
    elif dataset == 'amazon':
        vectors = [model.dv[vector] for vector in range(len(model.dv))]
        train_vectors = np.array(vectors)
        np.savetxt('reviews_data/amazon/vectors/X_train_vectors.txt', train_vectors, delimiter=',')

        for sent_num in range(
                len(X_test)):  # ~ 32 minutes (on 100 epochs) ~ 18 minutes (on 50 epochs) ~ 4 minutes (on 10 epochs)
            X_test[sent_num] = model.infer_vector(X_test[sent_num], epochs=100)
        test_vectors = np.array(X_test)
        np.savetxt('reviews_data/amazon/vectors/X_test_vectors.txt', test_vectors, delimiter=',')

        return train_vectors, test_vectors
    else:
        raise AttributeError(f"Dataset parameter should be 'movie' or 'amazon', not '{dataset}', depends on the dataset used")
