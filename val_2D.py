import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from torch.nn import functional as F

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        # dice = metric.binary.dc(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt, voxelspacing=[10, 1, 1])
        # asd = metric.binary.asd(pred, gt, voxelspacing=[10, 1, 1])
        # return dice, hd95, asd
        return 0 , 50, 10
    else:
        return 0 , 50, 10
# def calculate_metric_percase(pred, gt):
#     pred = torch.tensor(pred > 0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#     gt = torch.tensor(gt > 0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
#     if pred.sum() > 0 and gt.sum() > 0:
#         # Initialize metrics
#         dice_metric = DiceMetric(include_background=False)
#         hd_metric = HausdorffDistanceMetric(percentile=95)
#         asd_metric = SurfaceDistanceMetric()
        
#         # Calculate metrics
#         dice = dice_metric(pred, gt).item()
#         hd95 = hd_metric(pred, gt).item()
#         asd = asd_metric(pred, gt).item()
#         return dice, hd95, asd
#     else:
#         return 0, 50, 10
def calculate_metric_percase_easy(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=[1, 1])
        asd = metric.binary.asd(pred, gt, voxelspacing=[1, 1])
        return dice, hd95, asd
    else:
        return 0 , 50, 10
    

def test_single_volume(image, label, net, classes, patch_size=[256, 256], batch_size=8):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if  image.shape[0] == 3:
        prediction = np.zeros_like(label)
        
        
            
        slice = image
        modalities,  x, y = slice.shape[0], slice.shape[1], slice.shape[2]
        slice = zoom(slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)  # M,N,H,W
        # print('slice',slice.shape)
        net.eval()
        input = torch.from_numpy(slice).unsqueeze(0).float().cuda()  # N,M ,H,W
        # print('input',input.shape)
        with torch.no_grad():
            out = net(input)
            if isinstance(out, tuple):
                out = out[0]
            out = torch.argmax(torch.softmax(out, dim=1), dim=1)# N,C, H,W->N,H,W
            out = out.cpu().detach().numpy()
            # print('out',out.shape)
            pred = zoom(out, (1, x / patch_size[0], y / patch_size[1]), order=0)  # change this to zoom the last two dimensions
            prediction = pred.squeeze(0)  # change this to assign the prediction along the second dimension
    elif len(image.shape) == 3:
        prediction = np.zeros_like(label)
        ind_x = np.array([i for i in range(image.shape[0])])
        for ind in ind_x[::batch_size]:
            if ind + batch_size < image.shape[0]:
                slice = image[ind:ind + batch_size, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(
                    slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                print('input',input.shape)
                net.eval()
                with torch.no_grad():
                    out = net(input)
                    if isinstance(out, tuple):
                        out = out[0]
                    out = torch.argmax(torch.softmax(
                        out, dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    print('out',out.shape)
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[ind:ind + batch_size, ...] = pred
            else:
                slice = image[ind:, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(
                    slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()
                with torch.no_grad():
                    out = net(input)
                    if isinstance(out, tuple):
                        out = out[0]
                    out = torch.argmax(torch.softmax(out, dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[ind:, ...] = pred
        # slice = image# change this to slice along the second dimension
        # modalities, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
        # slice = zoom(slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)  # change this to zoom the last two dimensions
        # # print('slice',slice.shape)
        # net.eval()
        # input = torch.from_numpy(slice).unsqueeze(0).float().cuda()  
        # # print('input',input.shape)
        # with torch.no_grad():
        #     out = torch.argmax(torch.softmax(net(input), dim=1), dim=1)
        #     out = out.squeeze(0).cpu().detach().numpy()
        #     # print('out',out.shape)
        #     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # change this to zoom the last two dimensions
        #     prediction = pred  # change this to assign the prediction along the second dimension
    elif len(image.shape) == 4:
        prediction = np.zeros_like(label)
        ind_x = np.array([i for i in range(image.shape[1])])  # change this to slice along the second dimension
        for ind in ind_x[::batch_size]:
            if ind + batch_size < image.shape[1]:  # change this to check the second dimension
                slice = image[:, ind:ind + batch_size, ...]  # change this to slice along the second dimension
                modalities, thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2], slice.shape[3]
                slice = zoom(slice, (1, 1, patch_size[0] / x, patch_size[1] / y), order=0)  # M,N,H,W
                # print('slice',slice.shape)
                net.eval()
                input = torch.from_numpy(slice).permute(1, 0, 2, 3).float().cuda()  # N,M ,H,W
                # print('input',input.shape)
                with torch.no_grad():
                    out = net(input)
                    if isinstance(out, tuple):
                        out = out[0]
                    out = torch.argmax(torch.softmax(out, dim=1), dim=1)# N,C, H,W->N,H,W
                    out = out.cpu().detach().numpy()
                    # print('out',out.shape)
                    pred = zoom(out, (1, x / patch_size[0], y / patch_size[1]), order=0)  # change this to zoom the last two dimensions
                    prediction[ind:ind + batch_size, ...] = pred  # change this to assign the prediction along the second dimension
            else:
                slice = image[:, ind:, ...]  # change this to slice along the second dimension
                modalities, thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2], slice.shape[3]
                slice = zoom(slice, (1, 1, patch_size[0] / x, patch_size[1] / y), order=0)  # change this to zoom the last two dimensions
                net.eval()
                input = torch.from_numpy(slice).permute(1, 0, 2, 3).float().cuda()
                with torch.no_grad():
                    out = net(input)
                    if isinstance(out, tuple):
                        out = out[0]
                    out = torch.argmax(torch.softmax(out, dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / patch_size[0], y / patch_size[1]), order=0)  # change this to zoom the last two dimensions
                    prediction[ind:, ...] = pred  # change this to assign the prediction along the second dimension
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = net(input)
            if isinstance(out, tuple):
                out = out[0]
            out = torch.argmax(torch.softmax(
                out, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    if  image.shape[0] == 3:
        for i in range(1, classes):
            metric_list.append(calculate_metric_percase_easy(
                prediction == i, label == i))
    else:      
        for i in range(1, classes):
                metric_list.append(calculate_metric_percase(
                    prediction == i, label == i))
    return prediction,metric_list


# def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
#     image, label = image.squeeze(0).cpu().detach(
#     ).numpy(), label.squeeze(0).cpu().detach().numpy()
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         ind_x = np.array([i for i in range(image.shape[0])])

#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             slice = zoom(
#                 slice, (patch_size[0] / x, patch_size[1] / y), order=0)
#             input = torch.from_numpy(slice).unsqueeze(
#                 0).unsqueeze(0).float().cuda()
#             net.eval()
#             with torch.no_grad():
#                 out = torch.argmax(torch.softmax(
#                     net(input), dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
#                 pred = zoom(
#                     out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(
#                 net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(
#             prediction == i, label == i))

#     # return metric_list, image, prediction, label
#     return metric_list

#
def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_main, _, _, _ = net(input)
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list




def test_single_volume_multitask(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    # return metric_list, image, prediction, label
    return metric_list

import numpy as np
from sklearn.metrics import confusion_matrix


class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=0):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        if len(ground_truth.shape) > 1 or len(prediction.shape) > 1:
            ground_truth = ground_truth.flatten()
            prediction = prediction.flatten()
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        if self.overall_confusion_matrix is not None:
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_mIoU(self,smooth=1e-5):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=0)
        predicted_set = self.overall_confusion_matrix.sum(axis=1)
        union =  ground_truth_set + predicted_set - intersection

        intersection_over_union = (intersection + smooth ) / (union.astype(np.float32) + smooth)
        iou_list = [round(case,4) for case in intersection_over_union]
        mean_intersection_over_union = np.mean(intersection_over_union)
        
        return mean_intersection_over_union, iou_list
    
    def init_op(self):
        self.overall_confusion_matrix = None





class RunningDice():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Dice 
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=0):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        if len(ground_truth.shape) > 1 or len(prediction.shape) > 1:
            ground_truth = ground_truth.flatten()
            prediction = prediction.flatten()
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        if self.overall_confusion_matrix is not None:
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_dice(self,smooth=1e-5):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=0)
        predicted_set = self.overall_confusion_matrix.sum(axis=1)
        union =  ground_truth_set + predicted_set

        intersection_over_union = (2*intersection + smooth ) / (union.astype(np.float32) + smooth)
        dice_list = [round(case,4) for case in intersection_over_union]
        mean_intersection_over_union = np.mean(intersection_over_union[1:])
        
        return mean_intersection_over_union, dice_list
    
    def init_op(self):
        self.overall_confusion_matrix = None


def binary_dice(predict, target, smooth=1e-5, reduction='mean'):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A numpy array of shape [N, *]
        target: A numpy array of shape same with predict
    Returns:
        DSC numpy array according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[
        0], "predict & target batch size don't match"
    predict = predict.reshape(predict.shape[0], -1)  #N，H*W
    target = target.reshape(target.shape[0], -1)  #N，H*W

    inter = np.sum(np.multiply(predict, target), axis=1)  #N
    union = np.sum(predict + target, axis=1)  #N

    dice = (2 * inter + smooth) / (union + smooth)  #N

    if reduction == 'mean':
        # nan mean
        dice_index = dice != 1.0
        dice = dice[dice_index]
        return dice.mean()
    else:
        return dice  #N


def compute_dice(predict, target, ignore_index=0, reduction='mean'):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    N, num_classes, _, _ = predict.size()
    assert predict.shape == target.shape, 'predict & target shape do not match'
    predict = F.softmax(predict, dim=1)

    predict = torch.argmax(predict, dim=1).detach().cpu().numpy()  #N*H*W
    target = torch.argmax(target, dim=1).detach().cpu().numpy()  #N*H*W

    if reduction == 'mean':
        dice_array = -1.0 * np.ones((num_classes, ), dtype=np.float32)  #C
    else:
        dice_array = -1.0 * np.ones((num_classes, N), dtype=np.float32)  #CN

    for i in range(num_classes):
        if i != ignore_index:
            if i not in predict and i not in target:
                continue
            dice = binary_dice((predict == i).astype(np.float32),
                               (target == i).astype(np.float32),
                               reduction=reduction)
            dice_array[i] = np.round(dice, 4)

    if reduction == 'mean':
        dice_array = np.where(dice_array == -1.0, np.nan, dice_array)
        return np.nanmean(dice_array[1:])
    else:
        dice_array = np.where(dice_array == -1.0, 1.0,
                              dice_array).transpose(1, 0)  #CN -> NC
        return dice_array
 
def binary_dice_loss(predict, target, smooth=1e-5, reduction='mean'):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A numpy array of shape [N, *]
        target: A numpy array of shape same with predict
    Returns:
        DSC numpy array according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[
        0], "predict & target batch size don't match"
    predict = predict.reshape(predict.shape[0], -1)  #N，H*W
    target = target.reshape(target.shape[0], -1)  #N，H*W

    inter = np.sum(np.multiply(predict, target), axis=1)  #N
    union = np.sum(predict + target, axis=1)  #N

    dice = (2 * inter + smooth) / (union + smooth)  #N

    if reduction == 'mean':
        # nan mean
        dice_index = dice != 1.0
        dice = dice[dice_index]
        return 1 - dice.mean()
    else:
        return 1- dice  #N


def compute_dice_loss(predict, target, ignore_index=0, reduction='mean'):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    N, num_classes, _, _ = predict.size()
    assert predict.shape == target.shape, 'predict & target shape do not match'
    predict = F.softmax(predict, dim=1)

    predict = torch.argmax(predict, dim=1).detach().cpu().numpy()  #N*H*W
    target = torch.argmax(target, dim=1).detach().cpu().numpy()  #N*H*W

    if reduction == 'mean':
        dice_array = -1.0 * np.ones((num_classes, ), dtype=np.float32)  #C
    else:
        dice_array = -1.0 * np.ones((num_classes, N), dtype=np.float32)  #CN

    for i in range(num_classes):
        if i != ignore_index:
            if i not in predict and i not in target:
                continue
            dice = binary_dice_loss((predict == i).astype(np.float32),
                               (target == i).astype(np.float32),
                               reduction=reduction)
            dice_array[i] = np.round(dice, 4)

    if reduction == 'mean':
        dice_array = np.where(dice_array == -1.0, np.nan, dice_array)
        return np.nanmean(dice_array[1:])
    else:
        dice_array = np.where(dice_array == -1.0, 1.0,
                              dice_array).transpose(1, 0)  #CN -> NC
        return dice_array
 