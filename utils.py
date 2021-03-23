import torch


def bboxes_iou(bboxes1, bboxes2, eps=1e-5):
    """
    bboxes with shape (..., 4) where
    (..., 0) - x1
    (..., 1) - y1
    (..., 2) - x2
    (..., 3) - y2
    and x1 < x2, y1 < y2
    """

    bboxes1_x1 = bboxes1[..., 0:1] - bboxes1[..., 2:3] / 2
    bboxes1_y1 = bboxes1[..., 1:2] - bboxes1[..., 3:4] / 2
    bboxes1_x2 = bboxes1[..., 0:1] + bboxes1[..., 2:3] / 2
    bboxes1_y2 = bboxes1[..., 1:2] + bboxes1[..., 3:4] / 2

    bboxes2_x1 = bboxes2[..., 0:1] - bboxes2[..., 2:3] / 2
    bboxes2_y1 = bboxes2[..., 1:2] - bboxes2[..., 3:4] / 2
    bboxes2_x2 = bboxes2[..., 0:1] + bboxes2[..., 2:3] / 2
    bboxes2_y2 = bboxes2[..., 1:2] + bboxes2[..., 3:4] / 2

    intersection_x1 = torch.max(bboxes1_x1, bboxes2_x1)
    intersection_y1 = torch.max(bboxes1_y1, bboxes2_y1)
    intersection_x2 = torch.min(bboxes1_x2, bboxes2_x2)
    intersection_y2 = torch.min(bboxes1_y2, bboxes2_y2)
    intersection = (intersection_x2 - intersection_x1).clamp(0) * (intersection_y2 - intersection_y1).clamp(0)

    bboxes1_area = abs((bboxes1_x2 - bboxes1_x1) * (bboxes1_y2 - bboxes1_y1))
    bboxes2_area = abs((bboxes2_x2 - bboxes2_x1) * (bboxes2_y2 - bboxes2_y1))
    union = bboxes1_area + bboxes2_area - intersection

    iou = intersection / (union + eps)

    return iou
