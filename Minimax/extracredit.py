import torch, random, math, json
from extracredit_embedding import ChessDataset, initialize_weights
# import matplotlib.pyplot as plt

DTYPE=torch.float32
DEVICE=torch.device("cpu")

def trainmodel():

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=8*8*15, out_features=1)
    )
    epochs = 1000
    overfit_trend = 0
    lr = 0.000005
    optim = torch.optim.Adam(lr=lr, eps=1e-8, weight_decay=0.0001, params=model.parameters())
    # optim = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.1, weight_decay=0.001)
    loss_fn = torch.nn.L1Loss()
    losses = []
    valid_losses = []

    # ... and if you do, this initialization might not be relevant any more ...
    # model[1].weight.data = initialize_weights()
    with torch.no_grad():
        # model[1].weight.uniform_(-1.0 / (8*8*15), 1.0 / (8*8*15))
        model[1].weight.data = initialize_weights()
        model[1].bias.data = torch.zeros(1)

    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    validset = ChessDataset(filename='extracredit_validation.txt')
    validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False)
    for epoch in range(epochs):
        epoch_loss = 0
        valid_loss = 0
        for x,y in trainloader:
            model_output = model(x)
            loss = loss_fn(model_output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        epoch_loss /= len(trainloader)
        print(f"Epoch: {epoch}, loss: {epoch_loss / len(trainloader):4.5f}", end="\r")
        losses.append(epoch_loss)

        with torch.no_grad():
            for x,y in validloader:
                model_output = model(x)
                loss = loss_fn(model_output, y)
                valid_loss += loss.item()

            valid_loss /= len(validloader)
            # print(f"Epoch: {epoch}, loss: {epoch_loss / len(validloader):4.5f}", end="\r")

            # judge overfitting
            if len(valid_losses) > 0 and valid_loss > valid_losses[-1]:
                overfit_trend += 1
                torch.save(model, 'model_ckpt_{}.pkl'.format(epoch))
            elif overfit_trend > 0:
                overfit_trend -= 1
            if overfit_trend > 5:
                break
            else:
                valid_losses.append(valid_loss)

    # ... after which, you should save it as "model_ckpt.pkl":
    torch.save(model, 'model_ckpt.pkl')

    # # plot loss
    # plt.plot(losses)
    # plt.plot(valid_losses)
    # plt.show()

###########################################################################################
# def trainmodel():
#
#     # model = torch.nn.Sequential(
#     #     torch.nn.BatchNorm2d(15),
#     #     torch.nn.Conv2d(15, 30, 3, 1, 1),
#     #     torch.nn.BatchNorm2d(30),
#     #     torch.nn.Sigmoid(),
#     #     torch.nn.Conv2d(30, 30, 3, 1, 1),
#     #     torch.nn.BatchNorm2d(30),
#     #     torch.nn.Sigmoid(),
#     #     torch.nn.Conv2d(30, 15, 3, 1, 1),
#     #     torch.nn.BatchNorm2d(15),
#     #     torch.nn.Sigmoid(),
#     #     torch.nn.Flatten(),
#     #     torch.nn.Linear(in_features=960, out_features=960),
#     #     # torch.nn.Sigmoid(),
#     #     # torch.nn.Linear(in_features=512, out_features=512),
#     #     torch.nn.Sigmoid(),
#     #     torch.nn.Linear(in_features=960, out_features=512),
#     #     torch.nn.Sigmoid(),
#     #     torch.nn.Linear(in_features=512, out_features=128),
#     #     torch.nn.Sigmoid(),
#     #     torch.nn.Linear(in_features=128, out_features=32),
#     #     torch.nn.Sigmoid(),
#     #     torch.nn.Linear(in_features=32, out_features=1),
#     #     # torch.nn.Sigmoid(),
#     #     # torch.nn.Linear(in_features=32, out_features=1),
#     # )
#
#     lr = 0.01
#     epochs = 3000
#     optim = torch.optim.Adam(lr=lr, eps=1e-08, weight_decay=0, params=model.parameters())
#     loss_fn = torch.nn.MSELoss()
#     losses = []
#     valid_losses = []
#
#     # ... and you might want to put some code here to train your model:
#     trainset = ChessDataset(filename='extracredit_train.txt')
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
#     validset = ChessDataset(filename='extracredit_validation.txt')
#     validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False)
#     for epoch in range(epochs):
#         epoch_loss = 0
#         valid_loss = 0
#         for x,y in trainloader:
#             model_output = model(x)
#             loss = loss_fn(model_output, y)
#
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             epoch_loss += loss.item()
#
#         epoch_loss /= len(trainloader)
#         print(f"Epoch: {epoch}, loss: {epoch_loss / len(trainloader):4.5f}", end="\r")
#         losses.append(epoch_loss)
#
#         with torch.no_grad():
#             for x,y in validloader:
#                 model_output = model(x)
#                 loss = loss_fn(model_output, y)
#                 valid_loss += loss.item()
#
#             valid_loss /= len(validloader)
#             # print(f"Epoch: {epoch}, loss: {epoch_loss / len(validloader):4.5f}", end="\r")
#             valid_losses.append(valid_loss)
#
#     # ... after which, you should save it as "model_ckpt.pkl":
#     torch.save(model, 'model_ckpt.pkl')
#
#     # plot loss
#     plt.plot(losses)
#     plt.plot(valid_losses)
#     plt.show()

###########################################################################################
if __name__=="__main__":
    trainmodel()

# trainmodel()
    
    