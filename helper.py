# Arora, Priyank
# 1001-55-3349
# 2018-12-09
# Assignment-06-02

def unpickle(file, encoding='bytes'):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding=encoding)
    return dict
