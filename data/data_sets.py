import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Value
import h5py
import cv2
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import torch.nn.functional as F
import SimpleITK as sitk
from dataclasses import dataclass


from monai.transforms import (
    Compose,
    RandRotate90,
    RandFlip,
    RandZoom,
    RandSpatialCrop,
    RandGaussianNoise,
    RandAdjustContrast,
    SpatialPad,
    CenterSpatialCrop,
    IntensityStats,
)

def create_transforms(is_train=True):
    if is_train:
        return Compose([
            # RandRotate90(prob=0.2, spatial_axes=(0, 1)),
            RandFlip(prob=0.2, spatial_axis=0),
            RandZoom(prob=0.2, min_zoom=0.9, max_zoom=1.1),
            RandSpatialCrop(roi_size=(180, 224, 224), random_size=False), 
            RandGaussianNoise(prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrast(prob=0.2, gamma=(0.9, 1.1)),
        ])
    else:
        return Compose([
            CenterSpatialCrop(roi_size=(180, 224, 224)),
        ])

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def round_float(value, decimals=2):
    return round(value, decimals)

class MultiModalDicomDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, csv_path,is_train,target_size,tokenizer=None,data_ratio = 1.0,seed=8796):
        super(MultiModalDicomDataset, self).__init__()
        self.base_dir = base_dir
        self.csv_data = pd.read_csv(csv_path)  # 原来是 self.excel_data = pd.read_excel(excel_path)
        self.patient_data = []
        self.target_size = target_size
        self.class_counts = {0: 0, 1: 0,2:0}  # 初始化类别计数字典
        self.transforms = create_transforms(is_train)

        for dose_folder in os.listdir(base_dir):
            dose_path = os.path.join(base_dir, dose_folder, "RP")
            if os.path.isdir(dose_path):
                self.csv_data['dose_match'] = self.csv_data.apply(lambda row: self.match_dose_folder(dose_folder, row['处方剂量']), axis=1)
                for patient_folder in os.listdir(dose_path):
                    patient_path = os.path.join(dose_path, patient_folder)
                    if os.path.isdir(patient_path):
                        match = self.csv_data[(self.csv_data['编号'] == patient_folder) & 
                                                (self.csv_data['dose_match'] == True)]
                        if not match.empty:
                            # 检查是否所有必要的数据都存在
                            if self.check_data_completeness(patient_path):
                                csv_data = match.iloc[0]
                                label = self.get_classification_label(csv_data['RP分级'])
                                #print(label)
                                if label == -1:
                                    continue
                                self.patient_data.append({
                                    'patient_id': patient_folder,
                                    'dose_folder': dose_folder,
                                    'path': patient_path,
                                    'csv_data': match.iloc[0]
                                })
                                # 更新类别计数
                                self.class_counts[label] += 1
                            else:
                                print(f"Skipping incomplete data for patient: {patient_folder}")
        
        random.seed(seed)

        if data_ratio < 1.0:
            # 分层抽样：按类别保持比例抽取样本
            class_groups = defaultdict(list)
            for idx, item in enumerate(self.patient_data):
                label = self.get_classification_label(item['csv_data']['RP分级'])
                class_groups[label].append(idx)
            
            sampled_indices = []
            for label, indices in class_groups.items():
                n = max(1, int(len(indices) * data_ratio))  # 确保至少保留1个样本
                sampled_indices.extend(random.sample(indices, n))
            
            random.shuffle(sampled_indices)  # 打乱顺序
            self.patient_data = [self.patient_data[i] for i in sampled_indices]
            
            # 重新统计类别分布
            self.class_counts = defaultdict(int)
            for item in self.patient_data:
                label = self.get_classification_label(item['csv_data']['RP分级'])
                self.class_counts[label] += 1

        # 打印类别统计信息
        print("Class distribution:")
        for class_label, count in self.class_counts.items():
           print(f"Class {class_label}: {count} samples")

        total_samples = sum(self.class_counts.values())
        print(f"Total samples: {total_samples}")



    def match_dose_folder(self,dose_folder, prescribed_dose):
        prescribed_dose = str(prescribed_dose)
        # 将文件夹名中的逗号替换为斜杠
        dose_folder_normalized = dose_folder.replace(",", "/")
        
        # 检查完全匹配
        if dose_folder_normalized == prescribed_dose:
            return True
        
        # 检查不带分隔符的匹配
        if dose_folder.replace(",", "") == prescribed_dose.replace("/", ""):
            return True
        
        return False

    def check_data_completeness(self, patient_dir):
        path_list = os.listdir(patient_dir)
        has_ct = any("CT" in path for path in path_list)
        has_dose = any("RTDOSE" in path for path in path_list)
        has_st = any("RTst" in path for path in path_list)
        return has_ct and has_dose and has_st

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, index):
        # try: 
        #print("current index:",index)
        patient = self.patient_data[index]
        #logging.info(f"Processing item {index} for patient {patient['path']}")
        patient_dir = patient['path']

        path_list = os.listdir(patient_dir)
        ct_dir = next(os.path.join(patient_dir, path) for path in path_list if "CT" in path)
        dose_path = next(os.path.join(patient_dir, path) for path in path_list if "RTDOSE" in path)
        st_path = next(os.path.join(patient_dir, path) for path in path_list if "RTst" in path)

        # 读取CT数据
        ct_slices,ct_spacing = self.load_dicom_series(ct_dir)
        # 预处理 CT 数据
        ct_preprocessed = self.preprocess_ct(ct_slices)
        ct_tensor = torch.from_numpy(ct_preprocessed).float()
        ct_tensor, new_ct_spacing = self.resize_ct_with_spacing(ct_tensor, ct_spacing, self.target_size)


        # 读取剂量数据
        dose_file = os.path.join(dose_path, os.listdir(dose_path)[0])
        dose_data_ori = self.load_single_dicom(dose_file)
        dose_data = self.reshape_dose_data(dose_data_ori,self.target_size)

        # 读取结构数据
        st_file = os.path.join(st_path, os.listdir(st_path)[0])
        st_data = self.load_single_dicom(st_file)
        st_data = self.reshape_dose_data(st_data,self.target_size)
        st_tensor = torch.from_numpy(st_data).float() if isinstance(st_data, np.ndarray) else st_data

        # 添加批次维度
        if ct_tensor.dim() == 3:
            ct_tensor = ct_tensor.unsqueeze(0)
        if dose_data.dim() == 3:
            dose_data = dose_data.unsqueeze(0)
        if st_tensor.dim() == 3:
            st_tensor = st_tensor.unsqueeze(0)

        # 应用变换
        ct_tensor = self.transforms(ct_tensor)
        dose_data = self.transforms(dose_data)
        st_tensor = self.transforms(st_tensor)

        ## 处理Excel数据
        csv_row = patient['csv_data']
        text_input = self.process_text_input(csv_row)
        texts = torch.tensor(text_input)

        # 获取分类标签
        classification_label = self.get_classification_label(csv_row['RP分级'])

        return {
            'ct': ct_tensor,
            'dose': dose_data,
            'structure': st_tensor,
            'text': texts,
            'label': classification_label,
            'patient_id': patient['path'],
            'dose_folder': patient['dose_folder'],
            'ct_spacing': new_ct_spacing,
        }
        # except Exception as e:
        #     print(f"Error processing item {index}: {str(e)}")
        #     #return None

    def resize_ct_with_spacing(self, tensor_input, original_spacing, target_size):
        # 确保输入是 3D 张量 (D, H, W)
        if tensor_input.dim() == 2:
            tensor_input = tensor_input.unsqueeze(0)  # 添加深度维度
        elif tensor_input.dim() == 4:
            tensor_input = tensor_input.squeeze(0)  # 如果是 (1, D, H, W)，去掉第一个维度

        if tensor_input.dim() != 3:
            raise ValueError(f"Expected 3D tensor_input, got shape {tensor_input.shape}")

        original_size = torch.tensor(tensor_input.shape)
        target_size = torch.tensor(target_size)

        # 计算缩放因子
        scale_factors = original_size.float() / target_size.float()

        # 计算新的像素间距
        new_spacing = torch.tensor(original_spacing) * scale_factors

        # 将 3D 张量转换为 5D 张量 (N, C, D, H, W)
        tensor_5d = tensor_input.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度

        # 使用三线性插值调整大小
        resized = F.interpolate(tensor_5d, size=tuple(target_size.tolist()), mode='trilinear', align_corners=False)
        
        # 去除额外的维度，返回到 3D
        resized = resized.squeeze(0).squeeze(0)

        return resized, new_spacing.tolist()

    
    def preprocess_ct(self, ct_data):
        """预处理 CT 数据"""
        # 将 HU 值限制在一个合理的范围内
        ct_data = np.clip(ct_data, -1024, 1024)
        
        # 标准化到 [0, 1] 范围
        ct_data = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min())
        
        return ct_data


    def load_dicom_series(self, ct_folder):
        #读取文件夹中的CT序列（假设CT图像是以DICOM格式存储的）
        ct_series = sitk.ImageSeriesReader()

        # 获取指定目录下所有CT序列文件（DICOM文件）
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(ct_folder)
    
        # 将文件读取为一个3D图像
        ct_series.SetFileNames(dicom_names)
        ct_image = ct_series.Execute()
        metadata_dict = ct_image.GetSpacing()

        # 显示图像的尺寸信息
        #print(ct_image.GetSize())
        np_img = sitk.GetArrayFromImage(ct_image)
        ct_spacing = np.array(metadata_dict)
        ct_spacing = np.concatenate(([ct_spacing[-1]], ct_spacing[:-1]))
        return np_img,ct_spacing

    
    def load_single_dicom(self, file_path):
        try:
            dcm = pydicom.dcmread(file_path)
            if dcm.Modality == 'RTSTRUCT':
                return self.extract_rtstruct_info(dcm,self.dose)
            elif dcm.Modality == 'RTDOSE':
                # 处理 RTDOSE 文件
                self.dose = dcm
                dose_array = dcm.pixel_array * dcm.DoseGridScaling
                self.shape = dose_array.shape
                return dose_array.astype(np.float32)
            else:
                # print(f"Warning: Unsupported DICOM modality: {dcm.Modality}")
                return None
        except Exception as e:
            print(f"Error reading DICOM file {file_path}: {str(e)}")
            return None

    def physical_to_pixel_coords(self, points, origin, pixel_spacing):
        # 将物理坐标转换为像素坐标
        pixel_coords = (points - origin) / pixel_spacing
        return pixel_coords

    def mm_to_pixel(self, mm_coords: list|tuple, pixel_spacing: list|tuple,
                    image_position: list|tuple, image_orientation: list|tuple):
        """
        将毫米坐标转换为像素坐标。

        参数:
        - mm_coords: 要转换的毫米坐标，[x, y, z]

        返回:
        - pixel_coords: 对应的像素坐标，[row, column]
        """
        # 提取必要的DICOM属性
        # pixel_spacing = dcm.PixelSpacing  # [row_spacing, column_spacing] 单位为mm
        # image_position = dcm.ImagePositionPatient  # [x, y, z] 单位为mm
        # image_orientation = dcm.ImageOrientationPatient  # 列表，包含6个值

        # 方向余弦
        row_cosine = np.array(image_orientation[:3])
        col_cosine = np.array(image_orientation[3:])

        # 计算像素间距
        row_spacing, col_spacing, z_spacing = pixel_spacing

        # 计算相对于图像原点的偏移
        relative_position = np.array(mm_coords) - np.array(image_position)

        # 计算行和列索引
        row = np.dot(relative_position, row_cosine) / row_spacing
        column = np.dot(relative_position, col_cosine) / col_spacing
        if type(row) == np.float64 and type(column) == np.float64:
            return np.concatenate([np.array(int(round(row)))[..., np.newaxis, np.newaxis],
                                   np.array(int(round(column)))[..., np.newaxis, np.newaxis]], axis=1).astype(np.int32)
        else:
            return np.concatenate([np.int32(np.round(row))[..., np.newaxis], np.int32(np.round(column))[..., np.newaxis]], axis=1).astype(np.int32)


    def draw_fill_contour(self, dimension, roi, origin, pixelspacing, orientation):
        image_position = np.array(origin)
        results = np.zeros(dimension, dtype=np.uint8)
        for plane in roi.keys():
            points_list = roi[plane]  # x, y, z
            plane_index = int(round((plane - origin[2]) / pixelspacing[2]))
            plane_index = np.clip(plane_index, 0, dimension[0] - 1)
            temp_mask = np.zeros([dimension[1], dimension[2]], dtype=np.uint8)
            for points in points_list:
                # points_index = self.mm_to_pixel(points, image_position, pixelspacing)
                points_index = self.mm_to_pixel(points, pixelspacing, image_position, orientation)
                points_index[:, 0] = np.clip(points_index[:, 0], 0, dimension[2] - 1)  # x
                points_index[:, 1] = np.clip(points_index[:, 1], 0, dimension[1] - 1)  # y

                points_index = points_index.astype(np.int32)
                cv2.drawContours(temp_mask, [points_index], 0, 255, -1)

                results[plane_index, temp_mask > 0] = 1
        return results

    def extract_rtstruct_info(self, dcm,dose):
        required_rois = ['PTV', 'BODY', 'HEART', 'TOTAL LUNG', 'SPINAL CORD', 'L LUNG', 'R LUNG', 'PTV-G', 'PTV-C']
        masks = {roi: np.zeros(self.shape, dtype=np.uint8) for roi in required_rois}

        # if not hasattr(dcm, 'ROIContourSequence'):
        #     print("Warning: RTSTRUCT file does not contain ROIContourSequence")
        #     return np.stack(list(masks.values()), axis=0)
        depth, height, width = self.shape
        overall_mask = np.zeros([len(required_rois), depth, height, width], dtype=np.uint8)
        origin = dose.ImagePositionPatient
        orientation = dose.ImageOrientationPatient

        # 获取像素间距
        pixel_spacing = dose.PixelSpacing
        slice_thickness = dose.SliceThickness

        # 将像素间距和切片间距组合成一个包含三个元素的列表
        pixel_spacing = [pixel_spacing[0], pixel_spacing[1], slice_thickness]

        roi_name_map = {}
        for roi_seq in dcm.StructureSetROISequence:
            roi_name_map[roi_seq.ROINumber] = roi_seq.ROIName

        roi_aliases = {
            'L LUNG': ['L LUNG', 'LLUNG', 'L_LUNG', 'L-LUNG'],
            'R LUNG': ['R LUNG', 'RLUNG', 'R_LUNG', 'R-LUNG'],
            'SPINAL CORD': ['SC', 'SPINAL CORD', 'SPINAL_CORD', 'SPINAL-CORD'],
            'HEART': ['HEART'],
            'PTV': ['PTV'],
            'PTV-G': ['PTVG', 'PTV-G', 'PTV_G', 'PTV60'],
            'PTV-C': ['PTVC', 'PTV-C', 'PTV_C', 'PTV50'],
            'TOTAL LUNG': ['TOTAL LUNG', 'TOTAL_LUNG', 'TOTALLUNG', 'TOTAL-LUNG'],
            'BODY': ['BODY']
        }

        for roi_contour in dcm.ROIContourSequence:
            roi_name = roi_name_map.get(roi_contour.ReferencedROINumber)
            if not roi_name:
                continue

            standardized_roi_name = None
            for key, aliases in roi_aliases.items():
                if roi_name.upper() in [alias.upper() for alias in aliases]:
                    standardized_roi_name = key
                    break

            if not standardized_roi_name or standardized_roi_name not in required_rois:
                continue

            if not hasattr(roi_contour, 'ContourSequence'):
                #print(f"Warning: ROI {roi_name} does not have ContourSequence")
                continue

            roi_points_dict = {}
            for contour in roi_contour.ContourSequence:
                if not hasattr(contour, 'ContourData'):
                    #print(f"Warning: Contour in ROI {roi_name} does not have ContourData")
                    continue

                contour_data = np.array(contour.ContourData)

                if contour_data.ndim == 1:
                    points = contour_data.reshape((-1, 3))
                elif contour_data.ndim == 2:
                    points = contour_data
                else:
                    #print(f"Warning: Unexpected ContourData shape in ROI {roi_name}")
                    continue

                if points.shape[0] < 3:
                    #print(f"Warning: Contour in ROI {roi_name} has less than 3 points")
                    continue

                plane = points[0, 2]
                if plane not in roi_points_dict:
                    roi_points_dict[plane] = []
                roi_points_dict[plane].append(points)

            masks[standardized_roi_name] = self.draw_fill_contour(self.shape, roi_points_dict, origin, pixel_spacing, orientation)
        return np.stack(list(masks.values()), axis=0)

    def reshape_dose_data(self, dose_data,target_shape):
        if not isinstance(dose_data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(dose_data)}")
        
        # 将数据转换为 float32 类型
        dose_data = dose_data.astype(np.float32)
        
        # 转换为 PyTorch tensor
        dose_tensor = torch.from_numpy(dose_data).float()
        
        # 添加批次维度和通道维度（如果需要）
        if dose_tensor.dim() == 3:
            dose_tensor = dose_tensor.unsqueeze(0).unsqueeze(0)
        elif dose_tensor.dim() == 4:
            dose_tensor = dose_tensor.unsqueeze(0)
        
        # 目标形状
        #target_shape = (dose_data.shape[0], 512, 512)
    
        # 使用 F.interpolate 进行重采样
        resampled_data = F.interpolate(dose_tensor, size=target_shape, mode='trilinear', align_corners=False)
        
        # 去除多余的维度
        resampled_data = resampled_data.squeeze(0)
        
        return resampled_data

    def physical_to_pixel_coords(self, physical_coords):
        pixel_coords = (physical_coords - self.ct_origin) / self.ct_spacing
        return pixel_coords

    def process_text_input(self, row):
        std_value = {
            '1. 白细胞计数':[3.5,9.5],
            '7. 中性粒细胞计数':[1.8,6.3],
            '8淋巴细胞计数':[1.1,3.2],
            '9单核细胞计数':[0.1,0.6],
            '10嗜酸性粒细胞计数':[0,0.5],
            '11嗜碱性粒细胞计数':[0,0.06],
            '12血小板计数':[125,350],
            '17红细胞计数':[4.3,5.8],
            '18血红蛋白浓度':[130,175],
            '系统性免疫炎症指数（SII）':[70.31,2004.55],
            '血小板淋巴细胞比率（PLR）':[39.06,318.18],
            '淋巴细胞-单核细胞比率（LMR）':[1.83,32],
            '中性粒细胞-淋巴细胞比率（NLR）':[0.56,5.73]
        }
        text_columns = row[['1. 白细胞计数', '7. 中性粒细胞计数', '8淋巴细胞计数', '9单核细胞计数', 
                            '10嗜酸性粒细胞计数', '11嗜碱性粒细胞计数', '12血小板计数', 
                            '17红细胞计数', '18血红蛋白浓度','系统性免疫炎症指数（SII）','血小板淋巴细胞比率（PLR）',
                            '淋巴细胞-单核细胞比率（LMR）','中性粒细胞-淋巴细胞比率（NLR）']]
        
        values = []
        for i, (col_name, val) in enumerate(zip(text_columns.index, text_columns)):
            if not pd.isna(val):
                if val < std_value[col_name][0]:
                    values.append(round_float(1))
                elif val > std_value[col_name][round_float(1)]:
                    values.append(round_float(2))
                else:
                    #values.append(round_float((val - std_value[col_name][0]) / (std_value[col_name][1] - std_value[col_name][0])))
                    values.append(round_float(3))
            else:
                values.append(0)

        # 如果列表为空，返回一个全零张量
        if not values:
            return torch.zeros(len(text_columns), dtype=torch.float32)
        
        # 将值列表转换为张量
        #tensor = torch.tensor(values, dtype=torch.float32)
        tensor = values 
        # 如果张量长度小于预期列数，用零填充
        if len(tensor) < len(text_columns):
            padded_tensor = torch.zeros(len(text_columns), dtype=torch.float32)
            padded_tensor[:len(tensor)] = tensor
            return padded_tensor
        return tensor

    def get_classification_label(self, value):
        if value == "0":
            return 0
        elif value == "1-2":
            return -1
        elif value == "≥3":
            return 1
        else:
            raise "wrong label"

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def get_my_data(args, is_train=True):
    if is_train:
        dataset = MultiModalDicomDataset(args.data_dir,args.csv_file,is_train,args.target_size)
    else:
        dataset = MultiModalDicomDataset(args.data_dir,args.csv_file,is_train,args.target_size)

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=custom_collate
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    # return DataInfo(dataloader, sampler)
    return dataloader

