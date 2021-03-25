import torch

import utils


class YOLOLoss(torch.nn.Module):
    def __init__(self, grid_size, bbox_pred_amount, class_amount, lambda_coord=5, lambda_noobj=0.5, reduction='sum', eps=1e-5):
        super(YOLOLoss, self).__init__()

        self.grid_size = grid_size
        self.bbox_pred_amount = bbox_pred_amount
        self.class_amount = class_amount

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.reduction = reduction
        self.eps = eps

        self.mse = torch.nn.MSELoss(reduction=self.reduction)

    def forward(self, predictions, targets):
        """
        Calculates YOLOLoss for given predictions and targets
        :param predictions: tensor with shape (batch_size, grid_size * grid_size * (class_amount + 5 * bbox_pred_amount))
        :param targets: tensor with shape (batch_size, grid_size, grid_size, (class_amount + 5 * bbox_pred_amount))
        :return: loss value
        """

        predictions = predictions.reshape(-1, self.grid_size, self.grid_size, self.class_amount + self.bbox_pred_amount * 5)  # (batch_size, grid_size, grid_size, (class_amount + 5 * bbox_pred_amount))

        # Select target bboxes
        bbox_target_start_index = self.class_amount + 1
        bbox_target_end_index = bbox_target_start_index + 4
        bbox_targets = targets[..., bbox_target_start_index:bbox_target_end_index]  # (batch_size, grid_size, grid_size, 4)

        # Calculate iou between gt and bbox_pred_i
        bboxes_ious = []
        for bbox_pred_number in range(self.bbox_pred_amount):
            bbox_pred_start_index = self.class_amount + 5 * bbox_pred_number + 1
            bbox_pred_end_index = bbox_pred_start_index + 4
            bbox_pred = predictions[..., bbox_pred_start_index:bbox_pred_end_index]  # (batch_size, grid_size, grid_size, 4)

            iou = utils.bboxes_iou(bbox_pred, bbox_targets)  # (batch_size, grid_size, grid_size, 1)
            bboxes_ious.append(iou)

        # Get numbers of responsible bboxes
        bboxes_ious = torch.stack(bboxes_ious)  # (bbox_pred_amount, batch_size, grid_size, grid_size, 1)
        iou_maxes, bbox_responsible_number = torch.max(bboxes_ious, dim=0)  # (batch_size, grid_size, grid_size, 1) both

        # Find attributes of responsible bboxes (bbox coords and obj prob)
        bbox_pred_responsible = torch.zeros(*predictions.shape[:-1], 4)  # (batch_size, grid_size, grid_size, 4)
        obj_presented_pred_responsible = torch.zeros(*predictions.shape[:-1], 1)  # (batch_size, grid_size, grid_size, 1)
        for bbox_pred_number in range(self.bbox_pred_amount):
            # Select pred bboxes
            bbox_pred_start_index = self.class_amount + 5 * bbox_pred_number + 1
            bbox_pred_end_index = bbox_pred_start_index + 4
            bbox_pred = predictions[..., bbox_pred_start_index:bbox_pred_end_index]  # (batch_size, grid_size, grid_size, 4)

            # Select pred obj prob
            obj_presented_pred_index = self.class_amount + 5 * bbox_pred_number
            obj_presented_pred = predictions[..., obj_presented_pred_index:obj_presented_pred_index + 1]  # (batch_size, grid_size, grid_size, 1)

            # Update responsible attributes
            bbox_responsible_mask = (bbox_responsible_number == bbox_pred_number) * 1  # (batch_size, grid_size, grid_size, 1)
            bbox_pred_responsible += bbox_responsible_mask * bbox_pred
            obj_presented_pred_responsible += bbox_responsible_mask * obj_presented_pred

        # Fix bbox predictions to avoid numerical errors
        gradient_sign = torch.sign(bbox_pred_responsible[..., 2:4])
        bbox_pred_responsible[..., 2:4] = torch.abs(bbox_pred_responsible[..., 2:4])  # since at the beginning NN may predict negative values for width and height of bbox
        bbox_pred_responsible[..., 2:4] = torch.sqrt(bbox_pred_responsible[..., 2:4] + self.eps)  # eps to avoid sqrt(0)
        bbox_pred_responsible[..., 2:4] *= gradient_sign

        bbox_targets[..., 2:4] = torch.sqrt(bbox_targets[..., 2:4])

        # Select gt obj presented in cell mask
        obj_presented_target_index = class_amount
        obj_presented_target = targets[..., obj_presented_target_index:obj_presented_target_index + 1]  # (batch_size, grid_size, grid_size, 1)

        # Loss coordinates
        bbox_pred_responsible *= obj_presented_target  # Loss calculated only over cells where object exist
        bbox_targets *= obj_presented_target  # Loss calculated only over cells where object exist
        loss_coords = self.mse(
            torch.flatten(bbox_pred_responsible),
            torch.flatten(bbox_targets),
        )

        #  Loss obj presented
        obj_presented_pred_responsible *= obj_presented_target  # Loss calculated only over cells where object exist
        loss_obj_presented = self.mse(
            torch.flatten(obj_presented_pred_responsible),
            torch.flatten(obj_presented_target),
        )

        # Loss no obj presented
        loss_no_obj_presented = 0
        for bbox_pred_number in range(self.bbox_pred_amount):
            # Select pred obj prob
            obj_presented_pred_index = self.class_amount + 5 * bbox_pred_number
            obj_presented_pred = predictions[..., obj_presented_pred_index:obj_presented_pred_index + 1]  # (batch_size, grid_size, grid_size, 1)

            loss_no_obj_presented += self.mse(
                torch.flatten((1 - obj_presented_target) * obj_presented_pred),
                torch.flatten(1 - obj_presented_target),
            )

        # Loss classification
        class_pred = obj_presented_target * predictions[..., :class_amount]  # Loss calculated only over cells where object exist
        class_target = obj_presented_target * targets[..., :class_amount]  # Loss calculated only over cells where object exist
        loss_classification = self.mse(
            torch.flatten(class_pred),
            torch.flatten(class_target),
        )

        # Complete loss
        loss = self.lambda_coord * loss_coords + loss_obj_presented + self.lambda_noobj * loss_no_obj_presented + loss_classification

        return loss


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    # Reproducibility
    torch.manual_seed(0)

    import random
    random.seed(0)

    import numpy as np
    np.random.seed(0)

    # Loss settings
    grid_size = 7
    bbox_pred_amount = 2
    class_amount = 20
    criterion = YOLOLoss(grid_size, bbox_pred_amount, class_amount)

    # Fake input generation
    batch_size = 8
    predictions = torch.randn(batch_size, grid_size * grid_size * (class_amount + 5 * bbox_pred_amount))
    print("predictions", predictions.shape)

    targets = torch.randn(batch_size, grid_size, grid_size, class_amount + 5 * 1)
    obj_presented_index = class_amount
    targets[..., obj_presented_index] = (targets[..., obj_presented_index] > 0.5) * 1
    targets = torch.abs(targets)

    print("targets", targets.shape)
    print("targets[0, 0, 0, :]", targets[0, 0, 0, :])

    loss = criterion(predictions, targets)
    print("loss", loss.shape, loss)
