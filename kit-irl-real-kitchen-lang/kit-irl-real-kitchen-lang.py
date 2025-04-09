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
                'steps': tfds.features.Dataset(
                    {
                        'is_first': tf.bool,
                        'is_last': tf.bool,
                        'observation': tfds.features.FeaturesDict({
                            'state': tfds.features.Tensor(shape=(14,), dtype=tf.float32),
                            'images_top': tfds.features.Image(shape=(224, 224, 3), dtype=np.uint8), #(340, 420, 3)
                            'images_wrist_left': tfds.features.Image(shape=(224, 224, 3), dtype=np.uint8),
                            'images_wrist_right': tfds.features.Image(shape=(224, 224, 3), dtype=np.uint8),
                        }),
                        'action': tfds.features.Tensor(shape=(14,), dtype=tf.float32),
                        'reward': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'timestamp': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'frame_index': tfds.features.Tensor(shape=(), dtype=tf.int32),
                        'is_terminal': tfds.features.Tensor(shape=(), dtype=tf.bool),
                        'language_instruction': tfds.features.Text(),
                        'discount': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        # 'metadata': tfds.features.FeaturesDict({
                        #     'episode_index': tfds.features.Tensor(shape=(), dtype=tf.int32)
                        # }),
                    }
                ),
                 'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.',
                    ),
                    'traj_length': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Number of samples in trajectorie'
                    )
                }),
            }),
        )
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(data_path='/home/i53/student/shilber/delete/50_easy_transfer'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, data_path) -> Iterator[Tuple[str, Any]]:
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
        #         beam.Create(raw_dirs)
        #         | beam.Map(_parse_example)
        # )

def _parse_example(episode_path, embed=None):
    data = {}
    # leader_path = os.path.join(episode_path, 'leader/*.pt')
    # follower_path = os.path.join(episode_path, 'follower/*.pt')
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

    trajectory_length = len(data[list(data.keys())[0]])
    
    for feature in list(data.keys()):
        for i in range(len(data[feature])):
            data[f'delta_{feature}'] = torch.zeros_like(data[feature])
            if i == 0:
                data[f'delta_{feature}'][i] = 0
            else:
                data[f'delta_{feature}'][i] = data[feature][i] - data[feature][i-1]






  
    top_cam_path = os.path.join(episode_path, 'images/overhead_cam_orig')
    wrist_left_cam_path = os.path.join(episode_path, 'images/wrist_cam_left_orig')
    wrist_right_cam_path = os.path.join(episode_path, 'images/wrist_cam_right_orig')
    # top_cam_path = os.path.join(episode_path, 'images/cam_high_orig')
    # wrist_left_cam_path = os.path.join(episode_path, 'images/cam_left_wrist_orig')
    # wrist_right_cam_path = os.path.join(episode_path, 'images/cam_right_wrist_orig')
    top_cam_vector = create_img_vector(top_cam_path, trajectory_length)
    wrist_left_cam_vector = create_img_vector(wrist_left_cam_path, trajectory_length)
    wrist_right_cam_vector = create_img_vector(wrist_right_cam_path, trajectory_length)
    # cam1_image_vector = create_img_vector(cam1_path, trajectory_length)
    # cam2_image_vector = create_img_vector(cam2_path, trajectory_length)
    data.update({
                'image_top': top_cam_vector, 
                'image_wrist_left' : wrist_left_cam_vector, 
                'image_wrist_right' : wrist_right_cam_vector
                })
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
                'images_wrist_left': data['image_wrist_left'][i],
                'images_wrist_right': data['image_wrist_right'][i],
                'images_top' : data['image_top'][i],
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
            'language_instruction': "cube transfer right to left ",
            # 'language_instruction_2': data['language_description'][1],
            # 'language_instruction_3': data['language_description'][2],
            # 'language_embedding': language_embedding,
            'frame_index': i,
            'timestamp':i
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
  
    img_paths = glob.glob(os.path.join(img_folder_path, '*.jpg'))
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
    data_path = "/home/i53/student/shilber/delete/50_easy_transfer"
    #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    # create list of all examples
    raw_dirs = []
    get_trajectorie_paths_recursive(data_path, raw_dirs)
    for trajectorie_path in tqdm(raw_dirs):
        _, sample = _parse_example(trajectorie_path)
        #print(sample)