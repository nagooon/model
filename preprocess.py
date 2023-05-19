import numpy as np
for j in range(2, 12):
	print("/home/jeonghyeon/hoi/0511_p2_train/cup_kettle_trans/hand_cut" + str(j) + ".npz")
	hand_data = np.load("/home/jeonghyeon/hoi/0511_p2_train/cup_kettle_trans/hand_cut" + str(j) + ".npz", allow_pickle=True)
	print("/home/jeonghyeon/hoi/0511_p2_train/cup_kettle_trans/cutout_trans" + str(j) + ".npz")
	obj_T = np.load("/home/jeonghyeon/hoi/0511_p2_train/cup_kettle_trans/cutout_trans" + str(j) + ".npz", allow_pickle=True)
	obj_dic = dict(sorted(obj_T.items(), key=lambda x : int(x[0])))
	final_dic = {}
	prev_right_trans = [[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
	prev_left_trans = [[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
	for i in obj_dic:
		final_dic[i] = {}
		obj_dat = obj_dic[i][()]
		final_dic[i]["right"] = {}
		final_dic[i]['left'] = {}

		right_trans = hand_data[i][()]["Right"]["transform"]
		if right_trans is None:
			right_trans = prev_right_trans
		final_dic[i]["right"]["transform"] = right_trans
		final_dic[i]["right"]["joints"] = hand_data[i][()]["Right"]["joints_arr"]
		prev_right_trans = right_trans
		
		left_trans = hand_data[i][()]["Left"]["transform"]
		if left_trans is None:
			left_trans = prev_left_trans
		final_dic[i]["left"]["transform"] = np.linalg.inv(right_trans) @ left_trans
		temp = np.concatenate((hand_data[i][()]["Left"]["joints_arr"].T, np.ones((1, 21))), axis = 0)
		final_dic[i]["left"]["joints"] = ((final_dic[i]["left"]["transform"] @ temp)[:3]).T
		prev_left_trans = left_trans
		
		final_dic[i]["cup"] = {}
		final_dic[i]["cup"]["base"] = np.linalg.inv(right_trans) @ obj_dat["cup"]["base"]
		final_dic[i]["kettle"] = {}
		final_dic[i]["kettle"]["base"] = np.linalg.inv(right_trans) @ obj_dat["kettle"]["base"]
	print("/home/jeonghyeon/hoi/0511_p2_train/cup_kettle_trans/right_wrist_base" + str(j) + ".npz")
	np.savez("/home/jeonghyeon/hoi/0511_p2_train/cup_kettle_trans/right_wrist_base" + str(j) + ".npz", **final_dic)