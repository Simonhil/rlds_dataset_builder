from matplotlib import pyplot as plt
import tensorflow as tf
file_path = "/home/i53/student/shilber/tensorflow_datasets/kit_irl_real_kitchen_lang/1.0.0/kit_irl_real_kitchen_lang-train.tfrecord-00000-of-00002"
# action_joint_state 0:7
# joint_state 0:7

# Load the TFRecord dataset
dataset = tf.data.TFRecordDataset(file_path)
step = 0
# Parse and inspect the first few examples

for i, raw_record in enumerate(dataset.take(3)):  # Inspect first 3 examples
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    print(f"\nExample {step}:")
    for key, feature in example.features.feature.items():
        # Check the type of the feature
        dtype = feature.WhichOneof('kind')
        value = getattr(feature, dtype).value
        
        # Print key, dtype, and a sample of values
        print(f"  - Key: {key}")
        print(f"    Type: {dtype}")
        print(len(value))
        # if key == "steps/observation/images_wrist_left":
        #     print(len(value))
        #     img = value[600]
       
        #     # Convert tensor to numpy for matplotlib
        #     image = tf.io.decode_image(img, channels=3)  # or channels=1 if grayscale
        #     image = tf.cast(image, tf.uint8)
        #     image_np = image.numpy().squeeze()  # Remove channel if needed
            
        #     # Display the image
        #     plt.imshow(image_np, cmap='gray')
        #     plt.axis('off')
        #     plt.show()
        #print(f"    Values (first 5): {value[:5] if len(value) > 5 else value}")
    step +=1