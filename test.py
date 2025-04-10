import tensorflow_datasets as tfds

ds = tfds.load("example_dataset",  split='train')
sorted = sorted(tfds.as_numpy(ds), key=lambda x: x['episode_metadata']['episode_index'] )

p = 0
for i in sorted:
    print(i['steps'][0])
    p +=1