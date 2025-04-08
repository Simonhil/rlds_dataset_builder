import os
import cv2
from typing import Iterator, Tuple, Any
from scipy.spatial.transform import Rotation

import glob
import numpy as np
import natsort
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tqdm import tqdm
import torch
from pathlib import Path

tf.config.set_visible_devices([], "GPU")
#data_path = "/run/user/1000040/gvfs/ftp:host=nas-irl.local/home/normal_rel_robot_data"

# data_path = "/home/marcelr/uha_test_policy/finetune_data/delta_des_joint_state_euler"
# data_path = "/media/irl-admin/93a784d0-a1be-419e-99bd-9b2cd9df02dc1/preprocessed_data/upgraded_lab/quaternions_fixed/sim_to_polymetis/delta_des_joint_state"

class KitIrlRealKitchenLang(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        # 'image_front': tfds.features.Image(
                        #     shape=(512, 512, 3),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='Front camera RGB observation.',
                        # ),
                        # 'image_top_left': tfds.features.Image(
                        #     shape=(512, 512, 3),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='Top_left camera RGB observation.',
                        # ),
                        # 'image_top_right': tfds.features.Image(
                        #     shape=(512, 512, 3),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='Top_right camera RGB observation.',
                        # ),
                        # 'image_wrist': tfds.features.Image(
                        #     shape=(512, 512, 3),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='wrist camera RGB observation.',
                        # ),
                        'joint_state': tfds.features.Tensor(
                            shape=(12,),
                            dtype=np.float64,
                            doc='Robot joint state. Consists of [7x joint states]',
                        ),
                        'joint_state_velocity': tfds.features.Tensor(
                            shape=(12,),
                            dtype=np.float64,
                            doc='Robot joint velocities. Consists of [7x joint velocities]',
                        ),
                        'end_effector_pos': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Current End Effector position in Cartesian space',
                        ),
                        'end_effector_ori': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Current End Effector orientation in Cartesian space as Euler (xyz)',
                        ),
                        'end_effector_ori_quat': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float64,
                            doc='Current End Effector orientation in Cartesian space as Quaternion',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float64,
                        doc='Delta robot action, consists of [3x delta_end_effector_pos, '
                            '3x delta_end_effector_ori (euler: roll, pitch, yaw), 1x des_gripper_width].',
                    ),
                    'action_abs': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float64,
                        doc='Absolute robot action, consists of [3x delta_end_effector_pos, '
                            '3x delta_end_effector_ori (euler: roll, pitch, yaw), 1x des_gripper_width].',
                    ),
                    'action_joint_state': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float64,
                        doc='Robot action in joint space, consists of [7x joint states]',
                    ),
                    'action_joint_vel': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float64,
                        doc='Robot action in joint space, consists of [7x joint velocities]',
                    ),
                    'delta_des_joint_state': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float64,
                        doc='Delta robot action in joint space, consists of [7x joint states]',
                    ),
                    'action_gripper_width': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Desired gripper width, consists of [1x gripper width] in range [0, 1]',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    # """ 'language_instruction': tfds.features.Text(
                    #     doc='Language Instruction.'
                    # ),
                    # 'language_instruction_2': tfds.features.Text(
                    #     doc='Language Instruction.'
                    # ),
                    # 'language_instruction_3': tfds.features.Text(
                    #     doc='Language Instruction.'
                    # ),
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(3, 512),
                    #     dtype=np.float32,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ), """
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.',
                    ),
                    'traj_length': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Number of samples in trajectorie'
                    )
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=data_path),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        # create list of all examples
        raw_dirs = []
        get_trajectorie_paths_recursive(data_path, raw_dirs)
        print("# of trajectories:", len(raw_dirs))

        # for smallish datasets, use single-thread parsing
        for sample in raw_dirs:
            yield _parse_example(sample, self._embed)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

def _parse_example(episode_path, embed=None):
    data = {}
    leader_path = os.path.join(episode_path, 'leader/*.pt')
    follower_path = os.path.join(episode_path, 'follower/*.pt')
    path = os.path.join(episode_path,'*.pt')
    #path = os.path.join(episode_path, "*.pickle")
    for file in glob.glob(path):

        # Keys contained in .pickle:
        # 'joint_state', 'joint_state_velocity', 'des_joint_state', 'des_joint_vel', 'end_effector_pos', 'end_effector_ori', 'des_gripper_width', 'delta_joint_state',
        # 'delta_des_joint_state', 'delta_end_effector_pos', 'delta_end_effector_ori', 'language_description', 'traj_length'
        #pt_file_path = os.path.join(episode_path, file)
        name = Path(file).stem
        data.update({name : torch.load(file)})
    # for file in glob.glob(episode_path):
    #     name = 'des_' + Path(file).stem
    #     data.update({name : torch.load(file)})
    trajectory_length = data[list(data.keys())[0]].size()[0]

    for feature in list(data.keys()):
        for i in range(len(data[feature])):
            data[f'delta_{feature}'] = torch.zeros_like(data[feature])
            if i == 0:
                data[f'delta_{feature}'][i] = 0
            else:
                data[f'delta_{feature}'][i] = data[feature][i] - data[feature][i-1]








    # front_cam_path = os.path.join(episode_path, 'images/overhead_cam_orig')
    # top_left_cam_path = os.path.join(episode_path, 'images/wrist_cam_left_orig')
    # top_right_cam_path = os.path.join(episode_path, 'images/wrist_cam_right_orig')
    # front_cam_vector = create_img_vector(front_cam_path, trajectory_length)
    # top_left_cam_vector = create_img_vector(top_left_cam_path, trajectory_length)
    # top_right_cam_vector = create_img_vector(top_right_cam_path, trajectory_length)
    # # cam1_image_vector = create_img_vector(cam1_path, trajectory_length)
    # # cam2_image_vector = create_img_vector(cam2_path, trajectory_length)
    # data.update({
    #             'image_top': front_cam_vector, 
    #             'image_wrist_left' : top_left_cam_vector, 
    #             'image_wrist_right' : top_right_cam_vector
    #             })
    episode = []
    for i in range(trajectory_length):
        # compute Kona language embedding
        #language_embedding = embed(data['language_description']).numpy() if embed is not None else [np.zeros(512)]
        # action = np.append(data['delta_end_effector_pos'][i], delta_quat.as_euler("xyz"), axis=0)
        # action = np.append(action, data['des_gripper_width'][i])
        # action_abs = np.append(data['des_end_effector_pos'][i], abs_quat.as_euler("xyz"), axis=0)
        # action_abs = np.append(action_abs, data['des_gripper_width'][i])
        # action = data['delta_ee_pos'][i]
        # action = np.append(action, data['des_gripper_state'][i])
        # action_abs = data['des_ee_pos'][i]
        # action_abs = np.append(action_abs, data['des_gripper_state'][i])
        # action = data['des_joint_state'][i]
        action_all_joint = torch.zeros(14)
        observation_all_joint = torch.zeros(14)

        action_all_joint[:6] = data['leader_joint_pos'][i][:6]
        action_all_joint[6] = data['leader_gripper_joint'][i][0]
        action_all_joint[7:13] = data['leader_joint_pos'][i][6:]
        action_all_joint[13] = data['leader_gripper_joint'][i][1]


        observation_all_joint[:6] = data['follower_joint_pos'][i][:6]
        observation_all_joint[6] = data['follower_gripper_joint'][i][0]
        observation_all_joint[7:13] = data['follower_joint_pos'][i][6:]
        observation_all_joint[13] = data['follower_gripper_joint'][i][1]



        episode.append({
            'observation': {
                # 'image_front': data['image_front'][i],
                # 'image_wrist': data['image_wrist'][i],
                # 'image_top_right' : data['image_top_right'],
                # 'image_top_left' : data['image_top_left'],
                'state': observation_all_joint,
                # 'joint_state_velocity': data['joint_vel'][i],
                # 'end_effector_pos': data['ee_pos'][i][:3],
                # 'end_effector_ori_quat': data['ee_pos'][i][3:], 
                # 'end_effector_ori': Rotation.from_quat(data['ee_pos'][i][3:]).as_euler("xyz"),
            },
            # 'action': action,
            # 'action_abs': action_abs,
            'action':  action_all_joint,
            #'action_joint_state': data['des_joint_pos'][i],
            # 'action_joint_vel': data['des_joint_vel'][i],
            # 'action_gripper_width': data['des_gripper_state'][i],
            # 'delta_des_joint_state': data['delta_des_joint_pos'][i],
            'discount': 1.0,
            #'reward': float(i == (data['traj_length'] - 1)),
            'reward': 0,
            'is_first': i == 0,
            'is_last': i == (trajectory_length - 1),
            'is_terminal': i == (trajectory_length - 1),
            'language_instruction': "transfer right to left ",
            # 'language_instruction_2': data['language_description'][1],
            # 'language_instruction_3': data['language_description'][2],
            # 'language_embedding': language_embedding,
            'frame_index': i,
            #'metadata': {'episode_index': df['episode_index'][step_idx]}
        })

    # create output data sample
    sample = {
        'steps': episode,
        'episode_metadata': {
            'file_path': episode_path,
            'traj_length': trajectory_length,
        }
    }

    # if you want to skip an example for whatever reason, simply return None
    return episode_path, sample

def create_img_vector(img_folder_path, trajectory_length):
    cam_list = []
    img_paths = glob.glob(os.path.join(img_folder_path, '*.png'))
    img_paths = natsort.natsorted(img_paths)
    assert len(img_paths)==trajectory_length, "Number of images does not equal trajectory length!"

    for img_path in img_paths:
        img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        cam_list.append(img_array)
    return cam_list

def get_trajectorie_paths_recursive(directory, sub_dir_list):
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            sub_dir_list.append(directory) if entry == "images" else get_trajectorie_paths_recursive(full_path, sub_dir_list)

if __name__ == "__main__":
    data_path = "/home/sihi/delete/download/EXAMPLE"
    #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    # create list of all examples
    raw_dirs = []
    get_trajectorie_paths_recursive(data_path, raw_dirs)
    for trajectorie_path in tqdm(raw_dirs):
        _, sample = _parse_example(trajectorie_path)
        # print(sample)