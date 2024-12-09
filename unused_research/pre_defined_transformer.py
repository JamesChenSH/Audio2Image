import torch
import torch.nn as nn
from data_processing.build_database import AudioImageDataset



def train(model, train_loader, test_loader, num_epochs, lr, device, model_name):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for i, (audio, img) in enumerate(train_loader):
            audio = audio.to(device)
            img = img.to(device)

            optimizer.zero_grad()
            output = model(audio, img[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), img[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, loss: {loss.item()}")

        # model.eval()
        # with torch.no_grad():
        #     total = 0
        #     correct = 0
        #     for audio, img in test_loader:
        #         audio = audio.to(device)
        #         img = img.to(device)

        #         output = model(audio, img)
        #         _, predicted = torch.max(output, 1)
        #         total += img.size(0)
        #         correct += (predicted == img).sum().item()

            # print(f"Epoch {epoch}, test accuracy: {correct / total}")

    torch.save(model.state_dict(), f"model/{model_name}.pt")



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.img_embedding = nn.Embedding(256, 512)
        self.audio_embedding = nn.Linear(441, 512)
        decoder_norm = nn.LayerNorm(
            512, eps=0.98
        )
        self.transformerDecoderLayer = nn.TransformerDecoderLayer(512, 16, batch_first=True)
        self.transformer = nn.TransformerDecoder(self.transformerDecoderLayer, 12, norm=decoder_norm)
        self.out = nn.Linear(512, 256)

    def forward(self, audio, img):
        img = self.img_embedding(img)
        audio = self.audio_embedding(audio)

        x = self.transformer(img, audio)
        out = self.out(x)

        return out



if __name__ == "__main__":
    ds = torch.load("data/DS_airport_499.pt")
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=True)

    model = MyModel()

    model.to('cuda')

    train(model, train_loader, test_loader, num_epochs=200, lr=0.1, device='cuda', model_name="transformer_16_12_512_0.1")
