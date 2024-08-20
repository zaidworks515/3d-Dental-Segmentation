import torch
import torch.nn as nn
from sklearn import neighbors
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pygco import cut_from_graph
from torch.autograd import Variable
import json
import torch.nn.functional as F
from scipy.spatial import distance_matrix
#for plotting
import pyvista as pv
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import datetime





class MeshSegNet(nn.Module):
    def __init__(self, num_classes=17, num_channels=15, with_dropout=True, dropout_p=0.5):
        super(MeshSegNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p

        # MLP-1 -shape: [64, 64]
        self.mlp1_conv1 = torch.nn.Conv1d(self.num_channels, 64, 1)
        self.mlp1_conv2 = torch.nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)

        # FTM (feature-transformer module)
        self.fstn = STNkd(k=64)

        # GLM-1 (graph-contrained learning modulus)
        self.glm1_conv1_1 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_conv1_2 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_bn1_1 = nn.BatchNorm1d(32)
        self.glm1_bn1_2 = nn.BatchNorm1d(32)
        self.glm1_conv2 = torch.nn.Conv1d(32+32, 64, 1)
        self.glm1_bn2 = nn.BatchNorm1d(64)

        # MLP-2
        self.mlp2_conv1 = torch.nn.Conv1d(64, 64, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(64)
        self.mlp2_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)
        self.mlp2_conv3 = torch.nn.Conv1d(128, 512, 1)
        self.mlp2_bn3 = nn.BatchNorm1d(512)

        # GLM-2 (graph-contrained learning modulus)
        self.glm2_conv1_1 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_2 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_3 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_bn1_1 = nn.BatchNorm1d(128)
        self.glm2_bn1_2 = nn.BatchNorm1d(128)
        self.glm2_bn1_3 = nn.BatchNorm1d(128)
        self.glm2_conv2 = torch.nn.Conv1d(128*3, 512, 1)
        self.glm2_bn2 = nn.BatchNorm1d(512)

        # MLP-3
        self.mlp3_conv1 = torch.nn.Conv1d(64+512+512+512, 256, 1)
        self.mlp3_conv2 = torch.nn.Conv1d(256, 256, 1)
        self.mlp3_bn1_1 = nn.BatchNorm1d(256)
        self.mlp3_bn1_2 = nn.BatchNorm1d(256)
        self.mlp3_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.mlp3_conv4 = torch.nn.Conv1d(128, 128, 1)
        self.mlp3_bn2_1 = nn.BatchNorm1d(128)
        self.mlp3_bn2_2 = nn.BatchNorm1d(128)

        # Output
        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x, a_s, a_l):
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        # MLP-1
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))

        # FTM
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x_ftm = torch.bmm(x, trans_feat)

        # GLM-1
        sap = torch.bmm(a_s, x_ftm)
        sap = sap.transpose(2, 1)
        x_ftm = x_ftm.transpose(2, 1)
        x = F.relu(self.glm1_bn1_1(self.glm1_conv1_1(x_ftm)))
        glm_1_sap = F.relu(self.glm1_bn1_2(self.glm1_conv1_2(sap)))
        x = torch.cat([x, glm_1_sap], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))

        # MLP-2
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)

        # GLM-2
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = torch.bmm(a_s, x_mlp2)
        sap_2 = torch.bmm(a_l, x_mlp2)
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = sap_1.transpose(2, 1)
        sap_2 = sap_2.transpose(2, 1)
        x = F.relu(self.glm2_bn1_1(self.glm2_conv1_1(x_mlp2)))
        glm_2_sap_1 = F.relu(self.glm2_bn1_2(self.glm2_conv1_2(sap_1)))
        glm_2_sap_2 = F.relu(self.glm2_bn1_3(self.glm2_conv1_3(sap_2)))
        x = torch.cat([x, glm_2_sap_1, glm_2_sap_2], dim=1)
        x_glm2 = F.relu(self.glm2_bn2(self.glm2_conv2(x)))

        # GMP
        x = torch.max(x_glm2, 2, keepdim=True)[0]

        # Upsample
        x = torch.nn.Upsample(n_pts)(x)

        # Dense fusion
        x = torch.cat([x, x_ftm, x_mlp2, x_glm2], dim=1)

        # MLP-3
        x = F.relu(self.mlp3_bn1_1(self.mlp3_conv1(x)))
        x = F.relu(self.mlp3_bn1_2(self.mlp3_conv2(x)))
        x = F.relu(self.mlp3_bn2_1(self.mlp3_conv3(x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp3_bn2_2(self.mlp3_conv4(x)))

        # output
        x = self.output_conv(x)
        x = x.transpose(2,1).contiguous()
        x = torch.nn.Softmax(dim=-1)(x.view(-1, self.num_classes))
        x = x.view(batchsize, n_pts, self.num_classes)

        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class MeshSegNet(nn.Module):
    def __init__(self, num_classes=17, num_channels=15, with_dropout=True, dropout_p=0.5):
        super(MeshSegNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p

        # MLP-1 -shape: [64, 64]
        self.mlp1_conv1 = torch.nn.Conv1d(self.num_channels, 64, 1)
        self.mlp1_conv2 = torch.nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)

        # FTM (feature-transformer module)
        self.fstn = STNkd(k=64)

        # GLM-1 (graph-contrained learning modulus)
        self.glm1_conv1_1 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_conv1_2 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_bn1_1 = nn.BatchNorm1d(32)
        self.glm1_bn1_2 = nn.BatchNorm1d(32)
        self.glm1_conv2 = torch.nn.Conv1d(32+32, 64, 1)
        self.glm1_bn2 = nn.BatchNorm1d(64)

        # MLP-2
        self.mlp2_conv1 = torch.nn.Conv1d(64, 64, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(64)
        self.mlp2_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)
        self.mlp2_conv3 = torch.nn.Conv1d(128, 512, 1)
        self.mlp2_bn3 = nn.BatchNorm1d(512)

        # GLM-2 (graph-contrained learning modulus)
        self.glm2_conv1_1 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_2 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_3 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_bn1_1 = nn.BatchNorm1d(128)
        self.glm2_bn1_2 = nn.BatchNorm1d(128)
        self.glm2_bn1_3 = nn.BatchNorm1d(128)
        self.glm2_conv2 = torch.nn.Conv1d(128*3, 512, 1)
        self.glm2_bn2 = nn.BatchNorm1d(512)

        # MLP-3
        self.mlp3_conv1 = torch.nn.Conv1d(64+512+512+512, 256, 1)
        self.mlp3_conv2 = torch.nn.Conv1d(256, 256, 1)
        self.mlp3_bn1_1 = nn.BatchNorm1d(256)
        self.mlp3_bn1_2 = nn.BatchNorm1d(256)
        self.mlp3_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.mlp3_conv4 = torch.nn.Conv1d(128, 128, 1)
        self.mlp3_bn2_1 = nn.BatchNorm1d(128)
        self.mlp3_bn2_2 = nn.BatchNorm1d(128)

        # Output
        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x, a_s, a_l):
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        # MLP-1
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))

        # FTM
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x_ftm = torch.bmm(x, trans_feat)

        # GLM-1
        sap = torch.bmm(a_s, x_ftm)
        sap = sap.transpose(2, 1)
        x_ftm = x_ftm.transpose(2, 1)
        x = F.relu(self.glm1_bn1_1(self.glm1_conv1_1(x_ftm)))
        glm_1_sap = F.relu(self.glm1_bn1_2(self.glm1_conv1_2(sap)))
        x = torch.cat([x, glm_1_sap], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))

        # MLP-2
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)

        # GLM-2
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = torch.bmm(a_s, x_mlp2)
        sap_2 = torch.bmm(a_l, x_mlp2)
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = sap_1.transpose(2, 1)
        sap_2 = sap_2.transpose(2, 1)
        x = F.relu(self.glm2_bn1_1(self.glm2_conv1_1(x_mlp2)))
        glm_2_sap_1 = F.relu(self.glm2_bn1_2(self.glm2_conv1_2(sap_1)))
        glm_2_sap_2 = F.relu(self.glm2_bn1_3(self.glm2_conv1_3(sap_2)))
        x = torch.cat([x, glm_2_sap_1, glm_2_sap_2], dim=1)
        x_glm2 = F.relu(self.glm2_bn2(self.glm2_conv2(x)))

        # GMP
        x = torch.max(x_glm2, 2, keepdim=True)[0]

        # Upsample
        x = torch.nn.Upsample(n_pts)(x)

        # Dense fusion
        x = torch.cat([x, x_ftm, x_mlp2, x_glm2], dim=1)

        # MLP-3
        x = F.relu(self.mlp3_bn1_1(self.mlp3_conv1(x)))
        x = F.relu(self.mlp3_bn1_2(self.mlp3_conv2(x)))
        x = F.relu(self.mlp3_bn2_1(self.mlp3_conv3(x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp3_bn2_2(self.mlp3_conv4(x)))

        # output
        x = self.output_conv(x)
        x = x.transpose(2,1).contiguous()
        x = torch.nn.Softmax(dim=-1)(x.view(-1, self.num_classes))
        x = x.view(batchsize, n_pts, self.num_classes)

        return x



def segment(obj_path, model_path, jaw):

    def voxel_filter(point_cloud, leaf_size):
        same_points_list = {}
        filtered_points = []

        # step1 Calculate boundary points
        x_max, y_max, z_max = np.amax(point_cloud, axis=0)  # 计算 x,y,z三个维度的最值
        x_min, y_min, z_min = np.amin(point_cloud, axis=0)

        # step2 Determine the size of the voxel
        size_r = leaf_size

        # step3 Calculate the dimensions of each volex voxel grid
        Dx = (x_max - x_min) // size_r + 1
        Dy = (y_max - y_min) // size_r + 1
        Dz = (z_max - z_min) // size_r + 1

        # print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))

        # step4 Calculate the value of each point in each dimension in the volex grid
        h = list()  # h is a list of saved indexes
        for i in range(len(point_cloud)):
            hx = np.floor((point_cloud[i][0] - x_min) // size_r)
            hy = np.floor((point_cloud[i][1] - y_min) // size_r)
            hz = np.floor((point_cloud[i][2] - z_min) // size_r)
            h.append(hx + hy * Dx + hz * Dx * Dy)

        # step5 Sort h values
        h = np.array(h)
        h_indice = np.argsort(h)  # Extract the index and return the index of the elements in h sorted from small to large.
        h_sorted = h[h_indice]  # Ascending order
        count = 0  # used for accumulation of dimensions
        step = 20

        # Put points with the same h value into the same grid and filter them
        for i in range(1, len(h_sorted)):  # 0-19999 data points
            if h_sorted[i] == h_sorted[i - 1] and (i != len(h_sorted) - 1):
                continue

            elif h_sorted[i] == h_sorted[i - 1] and (i == len(h_sorted) - 1):
                point_idx = h_indice[count:]
                key = h_sorted[i - 1]
                same_points_list[key] = point_idx
                _G = np.mean(point_cloud[point_idx], axis=0)  # center of gravity of all points
                _d = np.linalg.norm(point_cloud[point_idx] - _G, axis=1, ord=2)  # Calculate distance to center of gravity
                _d.sort()
                inx = [j for j in range(0, len(_d), step)]  # Get the index of the specified interval element
                for j in inx:
                    index = point_idx[j]
                    filtered_points.append(point_cloud[index])
                count = i

            elif h_sorted[i] != h_sorted[i - 1] and (i == len(h_sorted) - 1):
                point_idx1 = h_indice[count:i]
                key1 = h_sorted[i - 1]
                same_points_list[key1] = point_idx1
                _G = np.mean(point_cloud[point_idx1], axis=0)  # center of gravity of all points
                _d = np.linalg.norm(point_cloud[point_idx1] - _G, axis=1, ord=2)  # Calculate distance to center of gravity
                _d.sort()
                inx = [j for j in range(0, len(_d), step)]  # Get the index of the specified interval element
                for j in inx:
                    index = point_idx1[j]
                    filtered_points.append(point_cloud[index])

                point_idx2 = h_indice[i:]
                key2 = h_sorted[i]
                same_points_list[key2] = point_idx2
                _G = np.mean(point_cloud[point_idx2], axis=0)  # center of gravity of all points
                _d = np.linalg.norm(point_cloud[point_idx2] - _G, axis=1, ord=2)  # Calculate distance to center of gravity
                _d.sort()
                inx = [j for j in range(0, len(_d), step)]  # Get the index of the specified interval element
                for j in inx:
                    index = point_idx2[j]
                    filtered_points.append(point_cloud[index])
                count = i

            else:
                point_idx = h_indice[count: i]
                key = h_sorted[i - 1]
                same_points_list[key] = point_idx
                _G = np.mean(point_cloud[point_idx], axis=0)  # center of gravity of all points
                _d = np.linalg.norm(point_cloud[point_idx] - _G, axis=1, ord=2)  # Calculate distance to center of gravity
                _d.sort()
                inx = [j for j in range(0, len(_d), step)]  # Get the index of the specified interval element
                for j in inx:
                    index = point_idx[j]
                    filtered_points.append(point_cloud[index])
                count = i

        # Change the point cloud format to array and return it externally
        # print(f'filtered_points[0]为{filtered_points[0]}')
        filtered_points = np.array(filtered_points, dtype=np.float64)

        return filtered_points,same_points_list



    def read_obj(obj_path, jaw):
#         jaw = 'upper'
        points = []
        faces = []
        with open(obj_path) as file:
            for line in file:
                strs = line.split()
                if len(strs) == 0:
                    continue

                if strs[0] == "v":
                    points.append((float(strs[1]), float(strs[2]), float(strs[3])))
                elif strs[0] == "f":
                    faces.append((int(strs[1]), int(strs[2]), int(strs[3])))
                elif len(strs) > 1 and strs[1][:5] == 'lower':
                    jaw = 'lower'
                elif len(strs) > 1 and strs[1][:5] == 'upper':
                    jaw = 'upper'
                else:
                    continue

        points = np.array(points)
        faces = np.array(faces)
        if jaw is None:
            raise ValueError("Jaw type not found in OBJ file")

        return points, faces, jaw

    def obj2pcd(obj_path, jaw):
        points, _, jaw = read_obj(obj_path, jaw)
        pcd_list = []
        num_points = np.shape(points)[0]
        for i in range(num_points):
            new_line = f"{points[i, 0]} {points[i, 1]} {points[i, 2]}"
            pcd_list.append(new_line.split())

        pcd_points = np.array(pcd_list).astype(np.float64)

        return pcd_points, jaw

    def mesh_grid(pcd_points):
        new_pcd, _ = voxel_filter(pcd_points, 0.6)
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(new_pcd)
        pcd_new.estimate_normals()
        distances = pcd_new.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 6 * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_new,
            o3d.utility.DoubleVector([radius, radius * 2]))

        vertices = np.asarray(mesh.vertices)  # Convert vertices to NumPy array
        return vertices, mesh

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


    num_classes = 17
    num_channels = 15

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Predicting
    with torch.no_grad():
        pcd_points, jaw = obj2pcd(obj_path, jaw)
        vertices, mesh = mesh_grid(pcd_points)

        vertices_points = np.asarray(mesh.vertices)
        triangles_points = np.asarray(mesh.triangles)
        N = triangles_points.shape[0]
        cells = np.zeros((triangles_points.shape[0], 9))
        cells = vertices_points[triangles_points].reshape(triangles_points.shape[0], 9)

        mean_cell_centers = mesh.get_center()
        cells[:, 0:3] -= mean_cell_centers[0:3]
        cells[:, 3:6] -= mean_cell_centers[0:3]
        cells[:, 6:9] -= mean_cell_centers[0:3]

        v1 = np.zeros([triangles_points.shape[0], 3], dtype='float32')
        v2 = np.zeros([triangles_points.shape[0], 3], dtype='float32')
        v1[:, 0] = cells[:, 0] - cells[:, 3]
        v1[:, 1] = cells[:, 1] - cells[:, 4]
        v1[:, 2] = cells[:, 2] - cells[:, 5]
        v2[:, 0] = cells[:, 3] - cells[:, 6]
        v2[:, 1] = cells[:, 4] - cells[:, 7]
        v2[:, 2] = cells[:, 5] - cells[:, 8]
        mesh_normals = np.cross(v1, v2)
        mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
        mesh_normals[:, 0] /= mesh_normal_length[:]
        mesh_normals[:, 1] /= mesh_normal_length[:]
        mesh_normals[:, 2] /= mesh_normal_length[:]

        # prepare input
        points = vertices_points.copy()
        points[:, 0:3] -= mean_cell_centers[0:3]
        normals = np.nan_to_num(mesh_normals).copy()
        barycenters = np.zeros((triangles_points.shape[0], 3))
        s = np.sum(vertices_points[triangles_points], 1)
        barycenters = 1 / 3 * s
        center_points = barycenters.copy()
        barycenters -= mean_cell_centers[0:3]

        # normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        # normalization
        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
            cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
            cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
            barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
            normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

        # concatenate
        X = np.column_stack((cells, barycenters, normals))

        # computing A_S and A_L
        A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
        A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
        D = distance_matrix(X[:, 9:12], X[:, 9:12])
        A_S[D < 0.1] = 1.0
        A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

        A_L[D < 0.2] = 1.0
        A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

        # numpy -> torch.tensor
        X = X.transpose(1, 0)
        X = X.reshape([1, X.shape[0], X.shape[1]])
        X = torch.from_numpy(X).to(device, dtype=torch.float)
        A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
        A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
        A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
        A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

        tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
        patch_prob_output = tensor_prob_output.cpu().numpy()



    def voxel_upsample(same_points_list, point_cloud, filtered_points, filter_labels, leaf_size):
        upsample_label = []
        upsample_point = []
        upsample_index = []

        # step1 Calculate boundary points
        x_max, y_max, z_max = np.amax(point_cloud, axis=0) # Calculate the maximum value of the three dimensions x, y, z
        x_min, y_min, z_min = np.amin(point_cloud, axis=0)

        # step2 Determine the size of the voxel
        size_r = leaf_size

        # step3 Calculate the dimensions of each volex voxel grid
        Dx = (x_max - x_min) // size_r + 1
        Dy = (y_max - y_min) // size_r + 1
        Dz = (z_max - z_min) // size_r + 1
        print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))

        # step4 Calculate the value of each point (sampled point) in each dimension within the volex grid
        h = list()
        for i in range(len(filtered_points)):
            hx = np.floor((filtered_points[i][0] - x_min) // size_r)
            hy = np.floor((filtered_points[i][1] - y_min) // size_r)
            hz = np.floor((filtered_points[i][2] - z_min) // size_r)
            h.append(hx + hy * Dx + hz * Dx * Dy)

        # step5 Query the dictionary same_points_list based on the h value
        h = np.array(h)
        count = 0
        for i in range(1, len(h)):
            if h[i] == h[i - 1] and i != (len(h) - 1):
                continue

            elif h[i] == h[i - 1] and i == (len(h) - 1):
                label = filter_labels[count:]
                key = h[i - 1]
                count = i

                # Cumulative number of labels, classcount: {‘A’: 2, ‘B’: 1}
                classcount = {}
                for i in range(len(label)):
                    vote = label[i]
                    classcount[vote] = classcount.get(vote, 0) + 1

                # Sort map values
                sortedclass = sorted(classcount.items(), key=lambda x: (x[1]), reverse=True)
                point_index = same_points_list[key]  # Point index list corresponding to h
                for j in range(len(point_index)):
                    upsample_label.append(sortedclass[0][0])
                    index = point_index[j]
                    upsample_point.append(point_cloud[index])
                    upsample_index.append(index)

            elif h[i] != h[i - 1] and (i == len(h) - 1):
                label1 = filter_labels[count:i]
                key1 = h[i - 1]
                label2 = filter_labels[i:]
                key2 = h[i]
                count = i

                classcount = {}
                for i in range(len(label1)):
                    vote = label1[i]
                    classcount[vote] = classcount.get(vote, 0) + 1

                sortedclass = sorted(classcount.items(), key=lambda x: (x[1]), reverse=True)
                point_index = same_points_list[key1]
                for j in range(len(point_index)):
                    upsample_label.append(sortedclass[0][0])
                    index = point_index[j]
                    upsample_point.append(point_cloud[index])
                    upsample_index.append(index)

                classcount = {}
                for i in range(len(label2)):
                    vote = label2[i]
                    classcount[vote] = classcount.get(vote, 0) + 1

                sortedclass = sorted(classcount.items(), key=lambda x: (x[1]), reverse=True)
                point_index = same_points_list[key2]
                for j in range(len(point_index)):
                    upsample_label.append(sortedclass[0][0])
                    index = point_index[j]
                    upsample_point.append(point_cloud[index])
                    upsample_index.append(index)
            else:
                label = filter_labels[count:i]
                key = h[i - 1]
                count = i
                classcount = {}
                for i in range(len(label)):
                    vote = label[i]
                    classcount[vote] = classcount.get(vote, 0) + 1

                sortedclass = sorted(classcount.items(), key=lambda x: (x[1]), reverse=True)
                point_index = same_points_list[key]  # h对应的point index列表
                for j in range(len(point_index)):
                    upsample_label.append(sortedclass[0][0])
                    index = point_index[j]
                    upsample_point.append(point_cloud[index])
                    upsample_index.append(index)

        # Restore the original order of index
        upsample_index = np.array(upsample_index)
        upsample_index_indice = np.argsort(upsample_index) # Extract the index and return the index of the elements in h sorted from small to large.
        upsample_index_sorted = upsample_index[upsample_index_indice]

        upsample_point = np.array(upsample_point)
        upsample_label = np.array(upsample_label)

        # Restore the original order of points and labels
        upsample_point_sorted = upsample_point[upsample_index_indice]
        upsample_label_sorted = upsample_label[upsample_index_indice]

        return upsample_point_sorted, upsample_label_sorted



    def KNN_sklearn_Load_data(voxel_points, center_points, labels):
        # Build model
        model = neighbors.KNeighborsClassifier(n_neighbors=3)
        model.fit(center_points, labels)
        prediction = model.predict(voxel_points.reshape(1, -1))

        return prediction[0]


    # Loading points for knn upsampling
    def Load_data(voxel_points, center_points, labels):
        meshtopoints_labels = []
        for i in range(0, voxel_points.shape[0]):
            meshtopoints_labels.append(KNN_sklearn_Load_data(voxel_points[i], center_points, labels))

        return np.array(meshtopoints_labels)


    def clone_runoob(li1):
        """
        copy list
        """
        li_copy = li1[:]

        return li_copy


    def class_inlier_outlier(label_list, mean_points,cloud, ind, label_index, points, labels):
        label_change = clone_runoob(labels)
        outlier_index = clone_runoob(label_index)
        ind_reverse = clone_runoob(ind)

        # Get the label subscript of the outlier point
        ind_reverse.reverse()
        for i in ind_reverse:
            outlier_index.pop(i)

        # Get outliers
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)
        outlier_points = np.array(outlier_cloud.points)

        for i in range(len(outlier_points)):
            distance = []
            for j in range(len(mean_points)):
                dis = np.linalg.norm(outlier_points[i] - mean_points[j], ord=2)  # Compute the distance between tooth and GT centroid
                distance.append(dis)
            min_index = distance.index(min(distance))  # Get the index of the label closest to the centroid of the outlier point
            outlier_label = label_list[min_index]  # Get the label of the outlier point
            index = outlier_index[i]
            label_change[index] = outlier_label

        return label_change


    def remove_outlier(points, labels):
        same_label_points = {}

        same_label_index = {}

        mean_points = [] # All label types correspond to the centroid coordinates of the point cloud.

        label_list = []
        for i in range(len(labels)):
            label_list.append(labels[i])
        label_list = list(set(label_list)) # To retrieve the order from small to large, take GT_label=[0, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
        label_list.sort()
        label_list = label_list[1:]

        for i in label_list:
            key = i
            points_list = []
            all_label_index = []
            for j in range(len(labels)):
                if labels[j] == i:
                    points_list.append(points[j].tolist())
                    all_label_index.append(j) # Get the subscript of the label corresponding to the point with label i
            same_label_points[key] = points_list
            same_label_index[key] = all_label_index

            tooth_mean = np.mean(points_list, axis=0)
            mean_points.append(tooth_mean)
            # print(mean_points)

        for i in label_list:
            points_array = same_label_points[i]
            # Build one o3d object
            pcd = o3d.geometry.PointCloud()
            # UseVector3dVector conversion method
            pcd.points = o3d.utility.Vector3dVector(points_array)

            # Perform statistical outlier removal on the point cloud corresponding to label i, find outliers and display them
            # Statistical outlier removal
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)  # cl是选中的点，ind是选中点index

            # Reclassify the separated outliers
            label_index = same_label_index[i]
            labels = class_inlier_outlier(label_list, mean_points, pcd, ind, label_index, points, labels)
            # print(f"label_change{labels[4400]}")

        return labels



    def remove_outlier_main(jaw, pcd_points, labels, instances_labels):
        # original point
        points = pcd_points.copy()
        label = remove_outlier(points, labels)

        # Save json file
        label_dict = {}
        label_dict["id_patient"] = ""
        label_dict["jaw"] = jaw
        label_dict["labels"] = label.tolist()
        label_dict["instances"] = instances_labels.tolist()

        b = json.dumps(label_dict)
        with open('segmentation results/dental-labels' + '.json', 'w') as f_obj:
            f_obj.write(b)
        f_obj.close()



    def mesh_to_points_main(jaw, pcd_points, center_points, labels):
        points = pcd_points.copy()

        # Downsampling
        voxel_points, same_points_list = voxel_filter(points, 0.6)

        after_labels = Load_data(voxel_points, center_points, labels)

        upsample_point, upsample_label = voxel_upsample(same_points_list, points, voxel_points, after_labels, 0.6)

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(upsample_point)
        instances_labels = upsample_label.copy()

        # Reclassify the label of the upper and lower jaws
        for i in tqdm(range(0, upsample_label.shape[0])):
            if jaw == 'upper':
                if (upsample_label[i] >= 1) and (upsample_label[i] <= 8):
                    upsample_label[i] = upsample_label[i] + 10
                elif (upsample_label[i] >= 9) and (upsample_label[i] <= 16):
                    upsample_label[i] = upsample_label[i] + 12
            else:
                if (upsample_label[i] >= 1) and (upsample_label[i] <= 8):
                    upsample_label[i] = upsample_label[i] + 30
                elif (upsample_label[i] >= 9) and (upsample_label[i] <= 16):
                    upsample_label[i] = upsample_label[i] + 32

        remove_outlier_main(jaw, pcd_points, upsample_label, instances_labels)


    round_factor = 100
    patch_prob_output[patch_prob_output < 1.0e-6] = 1.0e-6

    # unaries
    unaries = -round_factor * np.log10(patch_prob_output)
    unaries = unaries.astype(np.int32)
    unaries = unaries.reshape(-1, num_classes)

    # parawisex
    pairwise = (1 - np.eye(num_classes, dtype=np.int32))

    cells = cells.copy()

    cell_ids = np.asarray(triangles_points)
    lambda_c = 20
    edges = np.empty([1, 3], order='C')
    for i_node in tqdm(range(cells.shape[0])):
        # Find neighbors
        nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
        nei_id = np.where(nei == 2)
        for i_nei in nei_id[0][:]:
            if i_node < i_nei:
                cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]) / np.linalg.norm(
                    normals[i_node, 0:3]) / np.linalg.norm(normals[i_nei, 0:3])

                if cos_theta >= 1.0:
                    cos_theta = 0.9999
                theta = np.arccos(cos_theta)
                phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                if theta > np.pi / 2.0:
                    edges = np.concatenate(
                        (edges, np.array([i_node, i_nei, -np.log10(theta / np.pi) * phi]).reshape(1, 3)), axis=0)
                else:
                    beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                    edges = np.concatenate(
                        (edges, np.array([i_node, i_nei, -beta * np.log10(theta / np.pi) * phi]).reshape(1, 3)),
                        axis=0)

    edges = np.delete(edges, 0, 0)
    edges[:, 2] *= lambda_c * round_factor
    edges = edges.astype(np.int32)

    refine_labels = cut_from_graph(edges, unaries, pairwise)
    refine_labels = refine_labels.reshape([-1, 1])

    predicted_labels_3 = refine_labels.reshape(refine_labels.shape[0])
    mesh_to_points_main(jaw, pcd_points, center_points, predicted_labels_3)
    
    
    
    def custom_plotting(obj_path):
        mesh = pv.read(obj_path)

        with open('segmentation results/dental-labels.json', 'r') as file:
            labels_data = json.load(file)

        labels = labels_data['labels']

        instances = labels['instances']

        selected_labels = {1, 2, 3, 9, 10, 11}  # Set of labels for red color

        # Initialize vertex colors
        vertex_colors = np.ones((len(mesh.vertices), 3))  # RGB colors, default black
        vertex_colors[:, :] = [0.0, 0.0, 0.0]  # Set default color to black

        color_mapping = {
            label: [1.0, 0.0, 0.0]  # Red color for selected labels
            for label in selected_labels
        }

        for vertex_idx, instance_label in enumerate(instances):
            if instance_label in color_mapping:
                vertex_colors[vertex_idx] = color_mapping[instance_label]


        assert len(labels) == mesh.n_points or len(labels) == mesh.n_cells

        if len(labels) == mesh.n_points:
            mesh.point_data['Labels'] = np.array(labels, dtype=int)
        elif len(labels) == mesh.n_cells:
            mesh.cell_data['Labels'] = np.array(labels, dtype=int)

        plotter = pv.Plotter(notebook=False)  

        cmap = cm.get_cmap('jet', 27)
        colors = cmap(np.linspace(0, 1, 27))
        colormap = mcolors.ListedColormap(colors)

        plotter.add_mesh(mesh, scalars='Labels', show_scalar_bar=True, cmap=colormap, clim=[0, 27])
        plotter.set_background('white')

        glb_file = 'segmentation results/publicmodels/plotted_output.glb'
        plotter.export_gltf(glb_file, save_normals=True)
        # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # glb_file = f'segmentation results/public/models/plotting_{current_datetime}.glb'
        

        print(f"Saved 3D mesh visualization as {glb_file}")

        return f'The plotted GLB is saved as {glb_file}'
    
    
    custom_plotting(obj_path)
        
    
            
    return 'Segmentation & Plotting completed'  # Because the file (json output) will be created and save in the directory 




obj_path = 'AI model + obj file/01533UH2_upper.obj'
model_path = 'AI model + obj file/model.tar'

segment(obj_path, model_path, jaw='upper') # by default it is upper, please take the input from user