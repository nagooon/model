import numpy as np
import os
import torch
from torch.autograd import Variable
import pickle
import open3d as o3d
from glob import glob


def load_data(config, train = True):
	input_folder = config.get_string("input_rep.input_hand_data_path")
	base_obj_file = "/home/jeonghyeon/Documents/Projects/scanned/glasses/base.obj"
	obj_type = config.get_string("input_rep.obj_type")
	one_hot = config.get_bool("input_rep.one_hot")
	add_contact = config.get_bool("input_rep.add_contact")
	articulation = config.get_bool("output_rep.articulation")
	trans_vel = config.get_bool("output_rep.trans_vel")
	add_abs = config.get_bool("output_rep.add_abs")
	six_dim = config.get_bool("output_rep.six_dim")
	gaussian_noise = config.get_bool("input_rep.gaussian_noise")

	hand_data = []
	transformations = []
	coord_trans_list = []
	base_trans = []
	for person in glob(os.path.join(input_folder, "*")):
		if person.split("/")[-1] != "jeonghwan":
			for scene in glob(os.path.join(person, "*")):
				pkl_in_folder = os.path.join(input_folder, person, scene, "result", "pkl_output")
				npz_out_folder = os.path.join(input_folder, person, scene, "gathered_transformation")
				hd, tr, co, base_tr = hnd_cord_ob_mat(pkl_in_folder, npz_out_folder, base_obj_file, obj_type, one_hot, articulation, trans_vel, add_abs, six_dim, train = train, gaussian_noise = gaussian_noise)
				hand_data.append(hd)
				transformations.append(tr)
				coord_trans_list.append(co)
				base_trans.append(base_tr)
	return hand_data, transformations, coord_trans_list, base_trans



def obj_load(obj_npz_file_dir):
	train_kettle_transformation = []
	train_cup_transformation = []
	train_left_trans = []
	train_right_trans = []
	train_left_joints = []
	train_right_joints = []

	test_kettle_transformation = []
	test_cup_transformation = []
	test_left_trans = []
	test_right_trans = []
	test_left_joints = []
	test_right_joints = []

	# Loading all the transformation for each scene and dividing by train and test
	npz_list = sorted(glob(os.path.join(obj_npz_file_dir, "right*.npz")))[1:]
	train_list = npz_list[0:1]
	# For every scene, extract kettle and cup transformation
	for idx in range(len(train_list)):
		train_kettle_transformation.append([])
		train_cup_transformation.append([])
		train_right_trans.append([])
		train_left_trans.append([])
		train_right_joints.append([])
		train_left_joints.append([])
		scene_trans = train_list[idx]
		trans_file = np.load(scene_trans, allow_pickle=True)
		for frame_num, item_trans in trans_file.items():
			item_trans = item_trans[()]
			kettle_trans = item_trans["kettle"]["base"]
			cup_trans = item_trans["cup"]["base"]
			right_trans = item_trans["right"]["transform"]
			left_trans = item_trans["left"]["transform"]
			right_joints = item_trans["right"]["joints"]
			left_joints = item_trans["left"]["joints"]
			right_joints = right_joints[1:]
			left_joints = left_joints[1:]
			train_kettle_transformation[idx].append(kettle_trans)
			train_cup_transformation[idx].append(cup_trans)
			train_left_trans[idx].append(left_trans)
			train_left_joints[idx].append(left_joints)
			train_right_trans[idx].append(right_trans)
			train_right_joints[idx].append(right_joints)
	
	# Overfitting
	test_kettle_transformation = train_kettle_transformation[0]
	test_cup_transformation = train_cup_transformation[0]
	test_left_joints = train_left_joints[0]
	test_left_trans = train_left_trans[0]
	test_right_joints = train_right_joints[0]
	test_right_trans = train_right_trans[0]

	# return [train_kettle_transformation, train_cup_transformation, train_left_trans, train_left_joints, train_right_trans, train_right_joints], \
	# 	[test_kettle_transformation, test_cup_transformation, test_left_trans, test_left_joints, test_right_trans, test_right_joints] 

	train_x = []
	train_y = []
	test_x = []
	test_y = []

	for idx in range(len(train_kettle_transformation)):
		train_x.append([])
		train_y.append([])
		for i in range(len(train_kettle_transformation[idx])):
			left_trans = train_left_trans[idx][i]
			left_feat = np.concatenate((left_trans[:3,:2].reshape(-1), left_trans[:3, 3].reshape(-1), train_left_joints[idx][i].reshape(-1)))
			right_trans = train_right_trans[idx][i]
			right_feat = np.concatenate((right_trans[:3, :2].reshape(-1), right_trans[:3, 3].reshape(-1), train_right_joints[idx][i].reshape(-1)))
			input_feat = np.concatenate((left_feat, right_feat))
			train_x[idx].append(input_feat)

			kettle_trans = train_kettle_transformation[idx][i]
			cup_trans = train_cup_transformation[idx][i]
			kettle_feat = np.concatenate((kettle_trans[:3, :2].reshape(-1), cup_trans[:3, 3].reshape(-1)))
			cup_feat = np.concatenate((cup_trans[:3, :2].reshape(-1), cup_trans[:3, 3].reshape(-1)))
			output_feat = np.concatenate((kettle_feat, cup_feat))
			train_y[idx].append(output_feat)
	
	test_x = train_x[0]
	test_y = train_y[0]

	return train_x, train_y, test_x, test_y, test_right_trans


	# # Sliding window
	train_s_x = []
	train_s_y = []
	test_s_x = []
	test_s_y = []
	for i in range(len(train_x)):
		tot_frame = len(train_x[i])
		for idx in range(tot_frame - 29):
			train_s_x.append(train_x[i][idx:idx+30])
			train_s_y.append(train_y[i][idx: idx + 30])
	
	tot_frame = len(test_x)
	for idx in range(tot_frame - 29):
		test_s_x.append(test_x[idx: idx + 30])
		test_s_y.append(test_y[idx:idx+30])
	
	return train_s_x, train_s_y, test_s_x, test_s_y, test_right_trans
	
	# train_x = torch.tensor(np.array(train_x), dtype=torch.float32)
	# train_y = torch.tensor(np.array(train_y), dtype=torch.float32)
	# test_x = torch.tensor(np.array(test_kettle_transformation), dtype=torch.float32)
	# test_y = torch.tensor(np.array(test_cup_transformation), dtype=torch.float32)
	
	# return train_x, train_y, test_x, test_y, in_ket_list, in_cup_list
		
	# Left, right_hand_data: T X 21 X 3 
	# Normalize to the joint 0 and calculate velocity
	left_hand_data = np.array(left_hand_data)
	right_hand_data = np.array(right_hand_data)
	left_root = left_hand_data[:,0,:] # T X 1 X 3
	right_root = right_hand_data[:, 0, :] 
	left_rest = left_hand_data[:, 1:, :] # T X 20 X 3
	right_rest = right_hand_data[:, 1:, :]
	left_rest = left_rest - left_root[..., None, :]
	right_rest = right_rest - right_root[..., None, :]
    
	left_velocity = np.diff(left_root, axis=-2)
	right_velocity = np.diff(right_root, axis=-2)
	left_velocity = np.concatenate((0 * left_velocity[..., [0], :], left_velocity), axis=-2)
	right_velocity = np.concatenate((0 * right_velocity[..., [0], :], right_velocity), axis=-2)

	if one_hot:
		for tf in range(left_velocity.shape[0]):
			hand_data.append(np.concatenate((left_rest[tf].reshape(-1), left_velocity[tf], right_rest[tf].reshape(-1), right_velocity[tf], ob_type_one_hot(obj_type))))
	else:
		for tf in range(left_velocity.shape[0]):
			hand_data.append(np.concatenate((left_rest[tf].reshape(-1), left_velocity[tf], right_rest[tf].reshape(-1), right_velocity[tf])))

	# Hand data: T X 126
	hand_data = torch.tensor(np.array(hand_data), dtype=torch.float32)
    
	# Reading the output npz data, transformation: T X 7
	for npz_path in frame_path:
		npz_file = np.load(npz_path, allow_pickle= True)
		npz_trans = np.dot(coord_trans, npz_file["glasses"][()]["transformation"])[:3]
		rot_mat, trans_mat = npz_trans[:, :3], npz_trans[:, 3]
		if six_dim:
			six_rep = rot_mat[:, :2]
			six_rep = np.concatenate((six_rep[:,0].reshape(-1), six_rep[:,1].reshape(-1)))
			trans_feat = np.concatenate((six_rep, trans_mat.reshape((-1))))
		else:
			quat_coeff = rot2quat(rot_mat)
			trans_feat = np.concatenate((quat_coeff, trans_mat.reshape((-1))))
		if articulation:
			trans_feat = np.concatenate((trans_feat, np.array([npz_file["glasses"][()]["part1"], npz_file["glasses"][()]["part2"]])))
		transformation.append(trans_feat)

	# If trans_vel == true, use the velocity of the translation instead
	if trans_vel:
		if six_dim:
			translations = np.array(transformation)[:,6:9]
		else:
			translations = np.array(transformation)[:,4:7]
		trans_diff = np.diff(translations, axis=-2)
		trans_diff = np.concatenate((0*trans_diff[..., [0], :], trans_diff), axis = -2)
		transformation = np.array(transformation)
		if six_dim:
			transformation[:,6:9] = trans_diff.tolist()
		else:
			transformation[:,4:7] = trans_diff.tolist()
		# If we add the absolute coordinate in the first frame object based coordinate, shape becomes T X 10
		if add_abs:
			transformation = np.concatenate((transformation, translations), axis = 1)
		transformation = torch.tensor(transformation, dtype=torch.float32)
	else:
		transformation = torch.tensor(np.array(transformation), dtype=torch.float32)


	# Switch axis for Conv1d, hand_data: 68 X T, transformation: 12 X T
	hand_data = torch.swapaxes(hand_data, 0, 1)
	transformation = torch.swapaxes(transformation, 0, 1)
    
	if train == False:
		if trans_vel:
			return hand_data, transformation, coord_trans, translations[0]
		else:
			return hand_data, transformation, coord_trans, None


    # Making sliding window from the original data, time frame fixed to 60 frames
	total_frames = hand_data.shape[1] - 60 + 1
	temp1 = []
	temp2 = []
	for idx_num in range(total_frames):
		temp1.append(hand_data[:, idx_num:idx_num+60].numpy())
		temp2.append(transformation[:, idx_num:idx_num+60].numpy())
	hand_data = torch.tensor(np.array(temp1), dtype=torch.float32)
	transformation = torch.tensor(np.array(temp2), dtype=torch.float32)

	if trans_vel:
		return hand_data, transformation, coord_trans, translations[0]
	else:
		return hand_data, transformation, coord_trans, None



def ob_type_one_hot(obj_type):
	one_hot = np.zeros(13)
	if obj_type == "laptop":
		one_hot[0] = 1
	if obj_type == "mug":
		one_hot[1] = 1
	if obj_type == "cup":
		one_hot[2] = 1
	if obj_type == "spray":
		one_hot[3] = 1
	if obj_type == "drawer":
		one_hot[4] = 1
	if obj_type == "oven":
		one_hot[5] = 1
	if obj_type == "closet":
		one_hot[6] = 1
	if obj_type == "toy-car":
		one_hot[7] = 1
	if obj_type == "scissor":
		one_hot[8] = 1
	if obj_type == "glasses":
		one_hot[9] = 1
	if obj_type == "trash bin":
		one_hot[10] = 1
	if obj_type == "shampoo bottle":
		one_hot[11] = 1
	if obj_type == "tray":
		one_hot[12] = 1
	return one_hot

def rot2quat(rot):
    """
    input (3,3): rotation matrix 
    output (4,): [qx, qy, qz, qw]
    """
    trace = rot[0,0] + rot[1,1] + rot[2,2]
    quat = np.zeros(4)
    if trace > 0.0:
        s = np.sqrt(trace + 1.)
        quat[3] = s * 0.5
        s = 0.5 / s
        quat[0] = rot[2,1] - rot[1,2] * s
        quat[1] = rot[0,2] - rot[2,0] * s
        quat[2] = rot[1,0] - rot[0,1] * s
    else:
        if rot[0,0] < rot[1,1]:
            if rot[1,1] < rot[2,2]:
                i = 2 
            else:
                i = 1
        else:
            if rot[0,0] < rot[2,2]:
                i = 2
            else:
                i = 0 
        j = (i+1) % 3
        k = (i+2) % 3

        s = np.sqrt(rot[i,i] - rot[j,j] - rot[k,k] + 1.0)
        quat[i] = s * 0.5
        s = 0.5 / s
        quat[3] = (rot[k,j] - rot[j,k]) * s
        quat[j] = (rot[j,i] + rot[i,j]) * s
        quat[k] = (rot[k,i] + rot[i,k]) * s
    return quat

        
def quat_to_rot_numpy(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/np.linalg.norm(norm_quat)
    w, x, y, z = norm_quat[0], norm_quat[1], norm_quat[2], norm_quat[3]

    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = np.array([ [w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz],
                          [2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx],
                          [2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2]])
    return rotMat

def trans_feat_to_4x4(feature, config, base):
    """Convert [B x 7] transformation features into [B x 4 x 4] transformation matrices and [B x 2] articulation
    Args:
    	feature: size = [B, 7]
    Returns:
     	Transformation matrices: size = [B, 4, 4]
        Articulation matrices: size = [B x 2]
          """
    
	# Figuring out the output format

    six_dim = config.get_bool("output_rep.six_dim")
    trans_vel = config.get_bool("output_rep.trans_vel")
    art = config.get_bool("output_rep.articulation")
    
	# Dividing the data

    bn = feature.shape[0]
    transformations = np.zeros((bn, 4, 4))
    articulations = np.zeros((bn, 2))
    if trans_vel:
       base_x, base_y, base_z = base[0], base[1], base[2]	
    for idx in range(bn):
        feat_mat = feature[idx]
        if six_dim:
            six_dim_rep = feat_mat[:6]
            x_raw = six_dim_rep[:3]
            y_raw = six_dim_rep[3:]
            
            x = x_raw / np.linalg.norm(x_raw, ord=2)
            z = np.cross(x, y_raw)
            z = z / np.linalg.norm(z, ord=2)
            y = np.cross(z, x)
            x = x.reshape((3,1))
            y = y.reshape((3,1))
            z = z.reshape((3,1))
            rot_mat = np.concatenate((x, y, z), axis=1)
            if trans_vel:
                base_x += feat_mat[4]
                base_y += feat_mat[5]
                base_z += feat_mat[6]
                trans_mat = np.concatenate((rot_mat, np.array([[base_x], [base_y], [base_z]])), axis = 1)
            else:    
                trans_mat = np.concatenate((rot_mat, feat_mat[4:7].reshape((3,1))), axis = 1)
            trans_mat = np.concatenate((trans_mat, [[0, 0, 0, 1]]), axis=0)
            transformations[idx] = trans_mat 
            if art:
                articulations[idx] = feat_mat[9:]
        else:
            quats = np.concatenate(([feat_mat[3]], feat_mat[:3]))
            rot_mat = quat_to_rot_numpy(quats)
            if trans_vel:
                base_x += feat_mat[4]
                base_y += feat_mat[5]
                base_z += feat_mat[6]
                trans_mat = np.concatenate((rot_mat, np.array([[base_x], [base_y], [base_z]])), axis = 1)
            else:    
                trans_mat = np.concatenate((rot_mat, feat_mat[4:7].reshape((3,1))), axis = 1)
            trans_mat = np.concatenate((trans_mat, [[0, 0, 0, 1]]), axis=0)
            transformations[idx] = trans_mat 
            if art:
                articulations[idx] = feat_mat[7:]
    return transformations, articulations
    