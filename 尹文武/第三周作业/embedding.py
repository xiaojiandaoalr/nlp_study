# import torch
# import torch.nn as nn

TXT_PATH = "./大唐乘风录.txt"

def read_text(txt_path = TXT_PATH):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def gen_vocab(txt):
    vocab = {"<pad>": 0, "<unk>": 1}
    for index, word in enumerate(sorted(set(txt))):
        vocab[word] = index + 2

    return vocab

def encode_text(text, vocab, maxlength):
    ids = [vocab.get(char, vocab["<unk>"]) for char in text][:maxlength]
    ids += [vocab["<pad>"]] * (maxlength - len(ids))
    return ids

def get_text_ids(text_data, vocab, maxlength=5):
    return [encode_text(txt, vocab, maxlength) for txt, target_idx in text_data]

# vocab_set = gen_vocab(read_text(TXT_PATH))
# texts = ["你好呗", "今天前天好晴朗", "处处好风光", "徒具上乘功夫"]
# text_ids = [encode_text(t, vocab_set, maxlength=5) for t in texts]
# # print(torch.LongTensor(text_ids))
# embedding = nn.Embedding(len(vocab_set), 64, padding_idx=0, max_norm=1.0, norm_type=2)
# print(embedding.weight)
# e = embedding(torch.LongTensor(text_ids))
# print(e)