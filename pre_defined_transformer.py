import torch
import torch.nn as nn
from data_processing.build_database import AudioImageDataset



def train(model, train_loader, test_loader, num_epochs, lr, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for i, (audio, img) in enumerate(train_loader):
            audio = audio.to(device)
            img = img.to(device)

            optimizer.zero_grad()
            print(img.shape, audio.shape)
            output = model(audio, img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for audio, img in test_loader:
                audio = audio.to(device)
                img = img.to(device)

                output = model(audio, img)
                _, predicted = torch.max(output, 1)
                total += img.size(0)
                correct += (predicted == img).sum().item()

            print(f"Epoch {epoch}, test accuracy: {correct / total}")

    torch.save(model.state_dict(), f"model/{model_name}.pt")




class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.img_embedding = nn.Embedding(256, 512)
        self.audio_embedding = nn.Linear(441, 512)

        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12, num_decoder_layers=12, dim_feedforward=512, dropout=0.1, batch_first=True)

    def forward(self, audio, img):
        img = self.img_embedding(img)
        audio = self.audio_embedding(audio)

        output = self.transformer(audio, img)

        return output



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
