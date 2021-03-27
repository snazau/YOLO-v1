from collections import Counter
import cv2
import numpy as np
import torch


def hwc2chw(tensor):
    return np.transpose(tensor, (2, 0, 1))


def chw2hwc(tensor):
    return np.transpose(tensor, (1, 2, 0))


def plot_one_box(x, img, color, label, line_thickness=None):
    # print(type(line_thickness), line_thickness)
    line_thickness = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    if label:
        font_thickness = max(line_thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)


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


def mAP(bboxes_pred, bboxes_gt, threshold, class_amount, eps=1e-5):
    """
    bbox_pred:  list of bboxes
    bboxes_gt: list of bboxes
    threshold: float IoU threshold
    bbox is a list: [image_id, class_pred, confidence, x1, y1, x2, y2]
    """

    APs = []
    for class_eval in range(class_amount):
        class_specific_pred_bboxes = [bbox_pred for bbox_pred in bboxes_pred if bbox_pred[1] == class_eval]
        class_specific_gt_bboxes = [bbox_gt for bbox_gt in bboxes_gt if bbox_gt[1] == class_eval]

        # image_id_to_bbox_gt_mapping needs to exclude multiple TP for single gt bbox
        is_gt_bbox_used = Counter([bbox[0] for bbox in class_specific_gt_bboxes])
        for image_id, bboxes_gt_amount in is_gt_bbox_used.items():
            is_gt_bbox_used[image_id] = torch.zeros(bboxes_gt_amount)

        # Sort pred bboxes by confidence score
        class_specific_pred_bboxes.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(class_specific_pred_bboxes))
        FP = torch.zeros(len(class_specific_pred_bboxes))
        gt_bboxes_amount = len(class_specific_gt_bboxes)
        if gt_bboxes_amount == 0:
            continue

        # Create image_id to gt bboxes mapping
        image_id_to_gt_bboxes_mapping = {}

        image_ids_of_pred_bboxes = [bbox[0] for bbox in class_specific_pred_bboxes]
        for image_id in image_ids_of_pred_bboxes:
            image_id_to_gt_bboxes_mapping[image_id] = []

        for bbox_gt in class_specific_gt_bboxes:
            image_id = bbox_gt[0]
            if image_id not in image_id_to_gt_bboxes_mapping:
                image_id_to_gt_bboxes_mapping[image_id] = []
            image_id_to_gt_bboxes_mapping[image_id].append(bbox_gt)

        # Check TPs and FPs
        for index_bbox_pred, bbox_pred in enumerate(class_specific_pred_bboxes):
            image_id_pred = bbox_pred[0]
            image_specific_gt_bboxes = image_id_to_gt_bboxes_mapping[image_id_pred]

            best_iou = -1
            best_iou_index = -1
            for index_bbox_gt, bbox_gt in enumerate(image_specific_gt_bboxes):
                iou = bboxes_iou(torch.tensor(bbox_pred[3:]), torch.tensor(bbox_gt[3:]))
                if iou > best_iou:
                    best_iou = iou
                    best_iou_index = index_bbox_gt

            if best_iou > threshold:
                if is_gt_bbox_used[image_id_pred][best_iou_index] == 0:
                    TP[index_bbox_pred] = 1
                    is_gt_bbox_used[image_id_pred][best_iou_index] = 1
                else:
                    FP[index_bbox_pred] = 1
            else:
                FP[index_bbox_pred] = 1

        # Calculate precision and recall values
        TP_cumulative_sum = torch.cumsum(TP, dim=0)
        FP_cumulative_sum = torch.cumsum(FP, dim=0)
        recall_values = TP_cumulative_sum / (gt_bboxes_amount + eps)
        precision_values = TP_cumulative_sum / (TP_cumulative_sum + FP_cumulative_sum + eps)

        # Add (0, 1) point at the beginning of the PR-curve
        precision_values = torch.cat((torch.tensor([1]), precision_values))
        recall_values = torch.cat((torch.tensor([0]), recall_values))
        PR_curve_area = torch.trapz(precision_values, recall_values)
        APs.append(PR_curve_area)

    mAP = sum(APs) / len(APs)
    print(mAP)
    return mAP
