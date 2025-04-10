import os
from pathlib import Path
from typing import Iterator, Tuple, Any

import glob
import cv2
import natsort
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import torch
from conversion_utils import MultiThreadedDatasetBuilder






def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    episode_index = 0
    def _parse_example(episode_path):
        # load raw data --> this should change for your dataset
        
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


        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i, step in enumerate(data):
            # compute Kona language embedding
            #if i == 0:
                # only run language embedding once since instruction is constant -- otherwise very slow
                #language_embedding = _embed([step['language_instruction']])[0].numpy()




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
            'episode_index': episode_index,
            'file_path': episode_path,
            'traj_length': trajectory_length,
        }
    }
        
        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)
        episode_index +=1 

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



class ExampleDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 10             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 20  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

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
                    ),
                     'episode_index': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Number of samples'
                    )
                }),
            }),)

    def _split_paths(self):
        """Define filepaths for data splits."""
        print(self.info)
        return {
            'train': glob.glob('/home/i53/student/shilber/delete/50_easy_transfer/*'),
            #'val': glob.glob('data/val/episode_*.npy')
        }

    