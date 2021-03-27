import torch
from torch.utils.tensorboard import SummaryWriter

import architecture
import config
import dataset
import loss


def validate(dataloader, model, criterion, optimizer, epoch, writer):
    model.eval()

    all_labels = []
    all_preds = []
    all_filenames = []

    loss_avg = 0
    for batch_index, sample in enumerate(dataloader):
        inputs = sample["tensor"].to(config.device)
        labels = sample["label"].to(config.device)
        filenames = sample["filename"]

        with torch.no_grad():
            preds = model(inputs)

        loss_batch = criterion(preds, labels)
        loss_avg += loss_batch.item()

        all_labels.append(labels)
        all_preds.append(preds)
        all_filenames.append(filenames)

        print('\r', "Curr loss:", loss_batch.item(), end='')

        if config.debug is True:
            break
    print()

    map_avg = 0  # TODO: calculate mAP
    loss_avg /= len(dataloader)
    writer.add_scalars("loss/compare", {"validation": loss_avg}, epoch)

    return loss_avg, map_avg


def train(dataloader, model, criterion, optimizer, epoch, writer):
    model.train()

    all_labels = []
    all_preds = []
    all_filenames = []

    loss_avg = 0
    for batch_index, sample in enumerate(dataloader):
        inputs = sample["tensor"].to(config.device)
        labels = sample["label"].to(config.device)
        filenames = sample["filename"]

        optimizer.zero_grad()
        preds = model(inputs)
        loss_batch = criterion(preds, labels)
        loss_batch.backward()
        optimizer.step()

        loss_avg += loss_batch.item()

        all_labels.append(labels)
        all_preds.append(preds)
        all_filenames.append(filenames)

        if config.debug is True:
            break

    map_avg = 0  # TODO: calculate mAP
    loss_avg /= len(dataloader)
    writer.add_scalars("loss/compare", {"train": loss_avg}, epoch)

    return loss_avg, map_avg


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)
    print("Run name:", config.run_description)

    torch.autograd.set_detect_anomaly(True)

    # Metrics
    writer = SummaryWriter(config.curr_run_dir)

    # Data
    dataset_train = dataset.YOLODataset(
        config.images_dir,
        config.labels_dir,
        config.train_csv_path,
        config.desired_image_size,
        config.grid_size,
        config.class_amount,
    )
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.loader_workers)

    dataset_val = dataset.YOLODataset(
        config.images_dir,
        config.labels_dir,
        config.train_csv_path,
        config.desired_image_size,
        config.grid_size,
        config.class_amount,
    )
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config.batch_size, shuffle=True, num_workers=config.loader_workers)

    print("Images in training: " + str(len(dataset_train)))
    print("Dataloader train len: " + str(len(dataloader_train)))
    print("Images in validating: " + str(len(dataset_val)))
    print("Dataloader val len: " + str(len(dataloader_val)))
    print()

    # Model
    model = architecture.YOLOv1(config.class_amount, config.bbox_pred_amount)
    model = model.to(config.device)
    print("Model has been loaded successfully")

    # Criterion
    criterion = loss.YOLOLoss(config.grid_size, config.bbox_pred_amount, config.class_amount, reduction="mean")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), config.lr, amsgrad=True)

    # Train loop
    for epoch in range(config.epochs_amount):
        print("Epoch:", epoch)
        print("Training started")
        loss_avg_train, map_avg_train = train(dataloader_train, model, criterion, optimizer, epoch, writer)
        print("Avg training loss:", loss_avg_train)
        print("Avg training mAP:", map_avg_train)

        print("Validation started")
        loss_avg_val, map_avg_val = validate(dataloader_val, model, criterion, optimizer, epoch, writer)
        print("Avg validation loss:", loss_avg_val)
        print("Avg validation mAP:", map_avg_val)

        print()

        if config.debug is True:
            break
