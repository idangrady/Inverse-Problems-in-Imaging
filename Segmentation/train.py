import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET








def save_checkpoint(state, filenames ="Filename.tar"):
    print("saved Checkpoint")
    torch.save(state, filenames)

def load_checkpoint(checkpoint, model):
    print("Load checkpoint")

    model.load.state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device = "cuda"):

    """:arg
    pretict class for each individual pixel
    """
    correct = 0
    num_pixels = 0

    for x,y in loader:
        x = x.to(device)
        y =y.to(device)

        preds = torch.sigmoid((model(x)))


def trainfun(loader, model, optimizer, loss_fun, scaler):

    loop = tqdm(loader)

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device = DEVICE)

        target = target.float().unsqueeze(1).to(device = DEVICE)

        #forward
        with torch.coda.amp.autocast():
            predictions = model(data)

            loss = loss_fun(predictions, target)

        optimizer.zero_grad()
        scaler.scale(loss).backwards()
        scaler.step(optimizer)
        scaler.update()




def main():
    # train_transform = A.Compose(
    #     [
    #         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.Rotate(limit=35, p=1.0),
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.1),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )


    # val_transforms = A.Compose(
    #     [
    #         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    train_loader =0
    model = UNET(in_channels=3, out_channels=1).to(DEVICE) # to change to multiple classes ->change to cross entorypy
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        trainfun(train_loader, model, optimizer, loss_fn, scaler)
