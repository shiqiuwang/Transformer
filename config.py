# config.py

class Config(object):
    N = 1  # 6 in Transformer Paper
    d_model = 100  # 512 in Transformer Paper
    d_ff = 512  # 2048 in Transformer Paper
    h = 5
    dropout = 0.1
    output_size = 6
    lr = 0.0001
    max_epochs = 20
    batch_size = 128
    max_sen_len = 39
