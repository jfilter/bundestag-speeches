import sys
import keras


from keras_text import experiment

from keras_text.preprocessing import FastTextTokenizer
from keras_text.data import Dataset
from keras_text.models import YoonKimCNN, TokenModelFactory

data_path = '/Users/filter/data/all.csv'
proc_path = './proc.bin'
max_len = 200


def train_cnn(lr=0.001, batch_size=1024, dropout_rate=0.5, filter_sizes=[3, 4, 5], num_filters=20, results_base_dir=None):
    word_encoder_model = YoonKimCNN(
        filter_sizes=filter_sizes, num_filters=num_filters, dropout_rate=dropout_rate)
    train(word_encoder_model, lr, batch_size, results_base_dir)


def train(word_encoder_model, lr, batch_size, results_base_dir):
    ds = Dataset.load(proc_path)

    factory = TokenModelFactory(
        # 4, ds.tokenizer.token_index, max_tokens=max_len, embedding_type=None, embedding_dims=300)
        ds.num_classes, ds.tokenizer.token_index, max_tokens=max_len, embedding_type="fasttext.de", embedding_dims=300)

    model = factory.build_model(
        token_encoder_model=word_encoder_model, trainable_embeddings=False)

    experiment.train({'x': ds.X, 'y': ds.y, 'validation_split': 0.1},
                     model, word_encoder_model, epochs=2)


def build_dataset():
    tokenizer = FastTextTokenizer()

    experiment.setup_data(tokenizer, proc_path, max_len=max_len, load_csv_args={
        'data_path': data_path, 'class_col': 'speaker_party', 'limit': None})


def train_stacked():
    pass


def main():
    if len(sys.argv) != 2:
        raise ValueError('You have to specify a positional command!')
    if sys.argv[1] == 'setup':
        build_dataset()
    if sys.argv[1] == 'traincnn':
        train_cnn()
    if sys.argv[1] == 'trainstacked':
        train_stacked()


if __name__ == '__main__':
    main()
