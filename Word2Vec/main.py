import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, 1)

    def forward(self, input, context):
        i_embed = self.embeddings(input)
        c_embed = self.embeddings(context)
        x = torch.mul(i_embed, c_embed)
        output = self.linear(x)
        return output.squeeze(1)


class SkipGramNegativeSampling:
    def __init__(self, corpus, window_size=2, neg_samples=5, learning_rate=0.01, embed_size=100):
        self.corpus = corpus
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.learning_rate = learning_rate
        self.embed_size = embed_size

        self.word_to_idx = {}
        self.vocab_size = 0
        self.data = None
        self.input_data = None
        self.context_data = None
        self.value_data = None

        self.model = None

    def prepare_data(self):
        for i, word in enumerate(set(self.corpus)):
            self.word_to_idx[word] = i
        self.vocab_size = len(self.word_to_idx)
        self.data = [self.word_to_idx[word] for word in self.corpus]

    def generate_training_data(self):
        self.input_data = []
        self.context_data = []
        self.value_data = []
        for i in range(len(self.data)):
            # add context words
            cur_context = []
            cur_size = np.random.randint(self.window_size // 2, self.window_size + 1)
            for j in range(max(0, i - cur_size), min(len(self.data), i + cur_size + 1)):
                cur_context.append(self.data[j])
                if i == j:
                    continue
                self.input_data.append(torch.LongTensor([self.data[i]]))
                self.context_data.append(torch.LongTensor([self.data[j]]))
                self.value_data.append(torch.FloatTensor([1]))
            # add negative samples
            cur_neg = np.random.randint(self.neg_samples // 2, self.neg_samples + 1)
            for _ in range(cur_neg):
                negative_idx = np.random.randint(len(self.data))
                while self.data[negative_idx] in cur_context:
                    negative_idx = np.random.randint(len(self.data))
                self.input_data.append(torch.LongTensor([self.data[i]]))
                self.context_data.append(torch.LongTensor([self.data[negative_idx]]))
                self.value_data.append(torch.FloatTensor([0]))

    def train(self, num_epochs=5, batch_size=8):
        self.prepare_data()
        self.generate_training_data()

        self.model = Word2Vec(self.vocab_size, self.embed_size)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        dataset = TensorDataset(torch.tensor(self.input_data),
                                torch.tensor(self.context_data),
                                torch.tensor(self.value_data))
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        for epoch in range(num_epochs):
            for input_tensor, context_tensor, value in dataloader:
                optimizer.zero_grad()
                output = self.model(input_tensor, context_tensor)
                loss = criterion(output, value)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def get_result(self):
        return dict(map(lambda word:
                        (word,
                         self.model.embeddings(torch.LongTensor([self.word_to_idx[word]])).cpu().detach().numpy()[0]
                         ),
                        list(self.word_to_idx.keys())))


def train(data: str):
    """
    return: w2v_dict: dict
            - key: string (word)
            - value: np.array (embedding)
    """
    corpus = data.split()
    num_epochs = 10
    batch_size = 8
    window_size = 8
    neg_samples = 12
    learning_rate = 0.01
    embed_size = 150
    model = SkipGramNegativeSampling(corpus, window_size, neg_samples, learning_rate, embed_size)
    model.train(num_epochs, batch_size)
    return model.get_result()


def show_PCA(embeds):
    keys = list(embeds.keys())
    embed_values = [embeds[key] for key in keys]

    pca = PCA(n_components=2)
    components = pca.fit_transform(embed_values)
    plt.title('PCA visualisation')
    plt.plot(components[:, 0], components[:, 1], 'x')
    for key, (x, y) in zip(keys, components):
        plt.text(x, y, str(key), color="blue", fontsize=12)
    plt.show()


def main():
    text = " ".join([str(i) for _ in range(20) for i in range(20)])
    res = train(text)
    show_PCA(res)


if __name__ == "__main__":
    main()