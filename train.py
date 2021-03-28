import torch
from torch.utils.tensorboard import SummaryWriter

import architecture
import config
import dataset
import loss
import utils


def validate(dataloader, model, criterion, optimizer, epoch, writer):
    model.eval()

    all_labels = []
    all_outputs = []
    all_filenames = []

    loss_avg = 0
    for batch_index, sample in enumerate(dataloader):
        inputs = sample["tensor"].to(config.device)
        labels = sample["label"].to(config.device)

        filenames = sample["filename"]
        all_filenames = all_filenames + filenames

        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.reshape(-1, config.grid_size, config.grid_size, config.class_amount + config.bbox_pred_amount * 5)
            loss_batch = criterion(outputs.clone(), labels.clone())
            loss_avg += loss_batch.item()

        all_labels.append(labels)
        all_outputs.append(outputs)

        print('\r', "Curr loss:", loss_batch.item(), end='')

        if config.debug is True:
            break
    print()

    loss_avg /= len(dataloader)

    all_outputs = torch.cat(all_outputs, dim=0)  # (ds_size, grid_size, grid_size, class_amount + 5 * bbox_pred_amount)
    all_labels = torch.cat(all_labels, dim=0)  # (ds_size, grid_size, grid_size, class_amount + 5)

    all_labels = outputs_to_preds(all_labels, config.grid_size, 1, config.class_amount)  # (ds_size, grid_size, grid_size, 6)
    all_preds = outputs_to_preds(all_outputs, config.grid_size, config.bbox_pred_amount, config.class_amount)  # (ds_size, grid_size, grid_size, 6)

    bboxes_pred = preds_to_bboxes_list_with_names(
        all_preds,
        all_filenames,
        use_nms=True,
        iou_threshold=config.iou_threshold,
        confidence_threshold=config.confidence_threshold
    )

    bboxes_label = preds_to_bboxes_list_with_names(
        all_labels,
        all_filenames,
        use_nms=False,
        iou_threshold=config.iou_threshold,
        confidence_threshold=config.confidence_threshold
    )

    mAP = utils.mAP(bboxes_pred, bboxes_label, config.iou_threshold, config.class_amount)

    writer.add_scalars("loss/compare", {"validation": loss_avg}, epoch)
    writer.add_scalars("metrics/mAP", {"validation": mAP}, epoch)

    return loss_avg, mAP


def train(dataloader, model, criterion, optimizer, epoch, writer):
    model.train()

    all_labels = []
    all_outputs = []
    all_filenames = []

    loss_avg = 0
    for batch_index, sample in enumerate(dataloader):
        inputs = sample["tensor"].to(config.device)
        labels = sample["label"].to(config.device)

        filenames = sample["filename"]
        all_filenames = all_filenames + filenames

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.reshape(-1, config.grid_size, config.grid_size, config.class_amount + config.bbox_pred_amount * 5)
        loss_batch = criterion(outputs.clone(), labels.clone())
        loss_batch.backward()
        optimizer.step()

        loss_avg += loss_batch.item()

        all_labels.append(labels)
        all_outputs.append(outputs)

        print('\r', "Curr loss:", loss_batch.item(), end='')

        if config.debug is True:
            break
    print()

    loss_avg /= len(dataloader)

    all_outputs = torch.cat(all_outputs, dim=0)  # (ds_size, grid_size, grid_size, class_amount + 5 * bbox_pred_amount)
    all_labels = torch.cat(all_labels, dim=0)  # (ds_size, grid_size, grid_size, class_amount + 5)

    all_labels = outputs_to_preds(all_labels, config.grid_size, 1, config.class_amount)  # (ds_size, grid_size, grid_size, 6)
    all_preds = outputs_to_preds(all_outputs, config.grid_size, config.bbox_pred_amount, config.class_amount)  # (ds_size, grid_size, grid_size, 6)

    bboxes_pred = preds_to_bboxes_list_with_names(
        all_preds,
        all_filenames,
        use_nms=True,
        iou_threshold=config.iou_threshold,
        confidence_threshold=config.confidence_threshold
    )

    bboxes_label = preds_to_bboxes_list_with_names(
        all_labels,
        all_filenames,
        use_nms=False,
        iou_threshold=config.iou_threshold,
        confidence_threshold=config.confidence_threshold
    )

    mAP = utils.mAP(bboxes_pred, bboxes_label, config.iou_threshold, config.class_amount)

    writer.add_scalars("loss/compare", {"train": loss_avg}, epoch)
    writer.add_scalars("metrics/mAP", {"train": mAP}, epoch)

    return loss_avg, mAP


def outputs_to_preds(outputs, grid_size, bbox_output_amount, class_amount):
    # Get obj presented scores to choose responsible bbox
    bboxes_obj_presented_score = []
    for bbox_output_number in range(bbox_output_amount):
        obj_presented_index = class_amount + bbox_output_number * 5
        obj_presented_score = outputs[..., obj_presented_index:obj_presented_index + 1]  # (pred_amount, grid_size, grid_size, 1)
        bboxes_obj_presented_score.append(obj_presented_score)
    bboxes_obj_presented_score = torch.stack(bboxes_obj_presented_score)  # (bbox_output_amount, pred_amount, grid_size, grid_size, 1)

    # Get responsible bbox number
    obj_presented_score_maxes, bbox_responsible_number = torch.max(bboxes_obj_presented_score, dim=0)  # (pred_amount, grid_size, grid_size, 1) both

    # Get responsible bboxes
    bbox_pred_responsible = torch.zeros(*outputs.shape[:-1], 4).to(outputs.device)
    for bbox_output_number in range(bbox_output_amount):
        # Select pred bboxes
        bbox_pred_start_index = class_amount + 5 * bbox_output_number + 1
        bbox_pred_end_index = bbox_pred_start_index + 4
        bbox_pred = outputs[..., bbox_pred_start_index:bbox_pred_end_index].clone()  # (pred_amount, grid_size, grid_size, 4)

        # Update responsible bboxes
        bbox_responsible_mask = (bbox_responsible_number == bbox_output_number)  # (pred_amount, grid_size, grid_size, 1)
        bbox_pred_responsible = bbox_pred_responsible + bbox_responsible_mask * bbox_pred

    # Cell relative to image relative responsible bboxes
    cell_indices_x = torch.arange(grid_size).repeat(outputs.shape[0], grid_size, 1).unsqueeze(-1).to(outputs.device)
    cell_indices_y = cell_indices_x.permute(0, 2, 1, 3)

    cell_size = 1 / grid_size
    bbox_pred_responsible[..., 0:1] = cell_indices_x * cell_size + bbox_pred_responsible[..., 0:1] * cell_size
    bbox_pred_responsible[..., 1:2] = cell_indices_y * cell_size + bbox_pred_responsible[..., 1:2] * cell_size
    bbox_pred_responsible[..., 2:3] = bbox_pred_responsible[..., 2:3] * cell_size
    bbox_pred_responsible[..., 3:4] = bbox_pred_responsible[..., 3:4] * cell_size

    # Get class predictions
    pred_class = outputs[..., :class_amount].argmax(-1).unsqueeze(-1)

    # Convert into tensor of bboxes = [..., 6] where each  bbox = [pred_class, obj_score, x_c, y_c, w, h]
    preds = torch.cat([pred_class, obj_presented_score_maxes, bbox_pred_responsible], dim=-1)
    return preds


def preds_to_bboxes_list_with_names(preds, filenames, use_nms, iou_threshold, confidence_threshold):
    pred_amount, grid_size = preds.shape[:2]

    all_bboxes = []
    for pred_number in range(pred_amount):
        pred_bboxes = []
        for i in range(grid_size):
            for j in range(grid_size):
                bbox = [x.cpu().item() for x in preds[pred_number, i, j, :]]
                pred_bboxes.append(bbox)

        if use_nms is True:
            saved_bboxes = utils.nms(pred_bboxes, iou_threshold, confidence_threshold)
            for saved_bbox in saved_bboxes:
                bbox = [filenames[pred_number]] + saved_bbox
                all_bboxes.append(bbox)
        else:
            for pred_bbox in pred_bboxes:
                pred_bbox_confidence = pred_bbox[1]
                if pred_bbox_confidence > confidence_threshold:
                    bbox = [filenames[pred_number]] + pred_bbox
                    all_bboxes.append(bbox)
    return all_bboxes


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
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=False, num_workers=config.loader_workers)

    dataset_val = dataset.YOLODataset(
        config.images_dir,
        config.labels_dir,
        config.val_csv_path,
        config.desired_image_size,
        config.grid_size,
        config.class_amount,
    )
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=config.loader_workers)

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
    criterion = loss.YOLOLoss(config.grid_size, config.bbox_pred_amount, config.class_amount, reduction="sum")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    # Train loop
    for epoch in range(config.epochs_amount):
        print("Epoch:", epoch)
        print("Training started")
        loss_avg_train, mAP_train = train(dataloader_train, model, criterion, optimizer, epoch, writer)
        print("Avg loss:", loss_avg_train)
        print("mAP:", mAP_train)

        print("Validation started")
        loss_avg_val, mAP_val = validate(dataloader_val, model, criterion, optimizer, epoch, writer)
        print("Avg loss:", loss_avg_val)
        print("mAP:", mAP_val)

        print()

        if config.debug is True:
            break
