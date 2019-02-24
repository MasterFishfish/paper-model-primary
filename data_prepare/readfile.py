import xml.etree.ElementTree as ET
import copy
import numpy as np
import pickle
import os
import string

def load_data14semeval(batch_size, dim_w, dataset_name):
    word2idx = {}
    word2idx["<pad>"] = 0
    idxcount = 0
    pad_idx = 0

    train_file = "../datas/%s.xml" % dataset_name
    test_file = "../datas/.%s.xml" % dataset_name

    dataset_14semeval, word2idx, max_aspect_len, max_sent_len = load_xml(word2idx=word2idx,
                                                                         train_filepath=train_file,
                                                                         test_filepath=test_file,
                                                                         pad_symbol="<pad>",
                                                                         pad_idx=pad_idx,
                                                                         idx_cnt=idxcount)

    n_train = len(dataset_14semeval[0])
    n_test = len(dataset_14semeval[1])

    embeddings = get_embeddings(vocab=word2idx, dataset_name=dataset_name, embedding_dim=dim_w)
    embeddings = np.array(embeddings, dtype=np.float32)
    train_set = pad_dataset(dataset=dataset_14semeval[0], bs=batch_size)
    test_set = pad_dataset(dataset=dataset_14semeval[1], bs=batch_size)

    return [train_set, test_set], [n_train, n_test], word2idx, embeddings, [max_sent_len, max_aspect_len]

def read_data14semeval_xml(path):
    Samples = []
    sid = 0
    tree = ET.parse(path)
    root = tree.getroot()

    for sentence in root:
        for asp_terms in sentence.iter('aspectTerms'):
            # iter the aspects of on sentence.
            for asp_term in asp_terms.findall('aspectTerm'):
                row_text = sentence.find("text").text.lower()

                asp = asp_term.get("term").lower()
                label = asp_term.get("polarity").lower()
                if label == "positive":
                    y = 1
                elif label == "negative":
                    y = 0
                elif label == "neutral":
                    y = 2
                else:
                    continue
                sample = {}
                sample["y"] = y
                row_text = row_text.replace(".", " . ")
                row_text = row_text.replace(",", " , ")
                row_text = row_text.replace(";", " ; ")
                row_text = row_text.replace("!", " ! ")
                row_text = row_text.replace("'", " ' ")
                row_text = row_text.replace(")", " ) ")
                row_text = row_text.replace("(", " ( ")
                row_text = row_text.replace(":", " : ")
                row_text = row_text.replace('"', ' " ')

                asp = asp.replace(".", " . ")
                asp = asp.replace(",", " , ")
                asp = asp.replace(";", " ; ")
                asp = asp.replace("!", " ! ")
                asp = asp.replace("'", " ' ")
                asp = asp.replace(")", " ) ")
                asp = asp.replace("(", " ( ")
                asp = asp.replace(":", " : ")
                asp = asp.replace('"', ' " ')
                # tokens 为去掉空格符号的单词list
                tokens = row_text.strip().split()
                # asp 为去掉空白字符之后的asp单词列表
                asp_tokens = asp.strip().split()
                words, target_words = [], []
                d = []

                asp_from = int(asp_term.get("from"))
                left_most = len(row_text[0:asp_from].strip().split())
                right_most = left_most + len(asp_tokens) - 1
                for t in tokens:
                    words.append(t)
                for ast in asp_tokens:
                    target_words.append(ast)
                for pos in range(len(tokens)):
                    if pos < left_most:
                        d.append(right_most - pos)
                    else:
                        d.append(pos - left_most)

                # sample["sent"] 为一个string
                sample["sent"] = row_text.strip()
                # words["sent_words"] 为一个list
                # 包含了aspect的单词
                sample["sent_words"] = words.copy()
                sample["target_words"] = target_words.copy()
                sample["sent_words_count"] = len(words)
                sample["target_words_count"] = len(sample["target_words"])
                sample["dist"] = d.copy()
                sample["sid"] = sid
                sample['target_begin'] = left_most
                # +1 to note the single aspect note
                sample['target_end'] = right_most + 1
                sid += 1
                Samples.append(sample)
            return Samples


def pad_dataset(Samples, batch_size):
    n_samples = len(Samples)
    n_padsamples = batch_size - ( n_samples % batch_size )
    newdataset = copy.copy(Samples)
    newdataset = newdataset.extend(Samples[:n_padsamples])
    return newdataset


def pad_seq(maxlen, Samples, field, pad_symbol):
    n_sample = len(Samples)
    for i in range(n_sample):
        assert isinstance(Samples[i][field], list)
        while len(Samples[i][field]) < maxlen:
            Samples[i][field].append(pad_symbol)
    return Samples


def shuffle_dataset():
    pass


def transform_twitter():
    pass


def process_target(target_words):
    """
    """

    target_words_filtered = []
    punct = string.punctuation
    for w in target_words:
        if w in punct or w == '':
            continue
        target_words_filtered.append(w)
    return '-'.join(target_words_filtered)


def build_vocab(Samples, word2idx, idx_cnt, pad_idx):
    idx_count = idx_cnt
    sample_num = len(Samples)
    for i in range(sample_num):
        for str in Samples[i]["sent_words"]:
            if str not in word2idx:
                if idx_count == pad_idx:
                    idx_count += 1
                word2idx[str] = idx_count
                idx_count += 1
    return word2idx, idx_count


def set_wordid(Samples, word2idx, fields):
    n_samples = len(Samples)
    for i in range(n_samples):
        sent = Samples[i][fields]
        assert isinstance(Samples[i][fields], list)
        if fields == "sent_words":
            Samples[i]["sent_words_id"] = [word2idx[w] for w in sent]
            sent_words_id = Samples["sent_words_id"]
            Samples[i]["sent_id_reverse"] = sent_words_id[::-1]
        else:
            Samples[i]["target_words_id"] = [word2idx[w] for w in sent]
            target_words_id = Samples["target_words_id"]
            Samples[i]["target_id_reverse"] = target_words_id[::-1]
    return Samples



def load_xml(word2idx, train_filepath, test_filepath, pad_symbol, idx_cnt, pad_idx):
    train_Samples = read_data14semeval_xml(train_filepath)
    test_Samples = read_data14semeval_xml(test_filepath)

    # create vocab
    word2idx, idx_cnt = build_vocab(train_Samples, word2idx, idx_cnt, pad_idx)
    word2idx, idx_cnt = build_vocab(test_Samples, word2idx, idx_cnt, pad_idx)

    # pad the context
    all_train_sent_len = [samples["sent_words_count"] for samples in train_Samples]
    all_test_sent_len = [samples["sent_words_count"] for samples in test_Samples]
    max_sent_len = max(all_train_sent_len) if max(all_train_sent_len) > max(all_test_sent_len) else max(all_test_sent_len)
    train_Samples = pad_seq(maxlen=max_sent_len, Samples=train_Samples, field="sent_words", pad_symbol=pad_symbol)
    test_Samples = pad_seq(maxlen=max_sent_len, Samples=test_Samples, field="sent_words", pad_symbol=pad_symbol)

    # pad the aspect
    all_train_aspect_len = [samples["target_words_count"] for samples in train_Samples]
    all_test_aspect_len = [samples["target_words_count"] for samples in test_Samples]
    max_aspect_len = max(all_train_aspect_len) if max(all_train_aspect_len) > max(all_test_aspect_len) else max(all_test_aspect_len)
    train_Samples = pad_seq(maxlen=max_aspect_len, Samples=train_Samples, field="target_words", pad_symbol=pad_symbol)
    test_Samples = pad_seq(maxlen=max_aspect_len, Samples=test_Samples, field="target_words", pad_symbol=pad_symbol)

    # create idedword sentence
    train_Samples = set_wordid(train_Samples, word2idx=word2idx, fields="sent_words")
    test_Samples = set_wordid(test_Samples, word2idx=word2idx, fields="sent_words")

    # create idedword aspectseqence
    train_Samples = set_wordid(train_Samples, word2idx=word2idx, fields="target_words")
    test_Samples = set_wordid(test_Samples, word2idx=word2idx, fields="target_words")

    return [train_Samples, test_Samples], word2idx, max_aspect_len, max_sent_len


def get_embeddings(vocab, dataset_name, embedding_dim):
    """
    """
    if dataset_name == '14semeval_laptop' or dataset_name == '14semeval_rest':
        emb_file = '../datas/glove_840B_300d.txt'  # path of the pre-trained word embeddings
        pkl = '../datas/%s_840B.pkl' % dataset_name  # word embedding file of the current dataset
    elif dataset_name == 'Twitter':
        emb_file = '../datas/glove_840B_300d.txt'
        pkl = '../datas/%s_840B.pkl' % dataset_name
    print("Loading embeddings from %s or %s ... " % (emb_file, pkl))
    n_emb = 0
    if not os.path.exists(pkl):
        # 为 pad 留出位置
        embeddings = np.zeros((len(vocab), embedding_dim), dtype='float32')
        with open(emb_file, encoding='utf-8') as fp:
            for line in fp:
                eles = line.strip().split()
                w = eles[0]
                # if embeddings.shape[1] != len(eles[1:]):
                #	embeddings = np.zeros((len(vocab) + 1, len(eles[1:])), dtype='float32')
                n_emb += 1
                if w in vocab:
                    try:
                        embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                    except ValueError:
                        print("Not Found %s" % embeddings[vocab[w]])
        print("Find %s word embeddings!!" % n_emb)
        pickle.dump(embeddings, open(pkl, 'wb'))
    else:
        embeddings = pickle.load(open(pkl, 'rb'))
    return embeddings

