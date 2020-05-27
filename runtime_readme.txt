 # Export inference graph without weights 

 python export_inference_graph.py   --alsologtostderr   --model_name=mobilenet_v1   --output_file=./train_dir/inference_graph_mobilenet.pb --dataset_name=flowers


#### From inside docker container tf_1.13_opencv4 ######
# Freeze the graph by combining weights and graph nodes
python /tensorflow_src/tensorflow/python/tools/freeze_graph.py \
    --input_graph=./train_dir/inference_graph_mobilenet.pb \
    --input_binary=true \
    --input_checkpoint=./train_dir/model.ckpt-30000 \
    --output_graph=./train_dir/frozen_mobilenet.pb \
    --output_node_names=MobilenetV1/Predictions/Reshape_1

# Must run this in the tensorflow_src directory in root
bazel build tensorflow/tools/graph_transforms:transform_graph


/tensorflow_src/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=frozen_mobilenet.pb \
--out_graph=optimized_mobilenet.pb \
--inputs='input' \
--outputs='MobilenetV1/Predictions/Reshape_1' \
--transforms='strip_unused_nodes(type=float, shape="1,224,224,3") remove_nodes(op=Identity, op=CheckNumerics, op=PlaceholderWithDefault)
 fold_batch_norms
 fold_old_batch_norms'

/tensorflow_src/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=./train_dir/frozen_mobilenet.pb \
--out_graph=./train_dir/optimized_graph.pb \
--inputs='input' \
--outputs='MobilenetV1/Predictions/Reshape_1' \
--transforms='
 strip_unused_nodes(type=float, shape="1,224,224,3")
 remove_nodes(op=Identity, op=CheckNumerics, op=PlaceholderWithDefault)
 fold_batch_norms
 fold_old_batch_norms'


# optimize (DOES NOT WORK when converting the output of this script using ncsdk or neuro)
python3 /tensorflow_src/tensorflow/python/tools/optimize_for_inference.py \
    --input=./frozen_mobilenet.pb \
    --output=./opt_frozen_mobilenet.pb \
    --input_names=input \
    --output_names=MobilenetV1/Predictions/Reshape_1

###################
python create_and_convert_dataset.py
    --dataset_name=flowers     
    --images_dataset_dir=/home/docker/ahmed/datasets/flower_photos     
    --tfrecords_dataset_dir=/home/docker/ahmed/datasets/flower_photos_tfrecord     
    --validation_percentage=20     
    --test_percentage=0



python train_image_classifier.py \
    --train_dir=./train_dir/mobilenet_v1_flowers/output \
    --dataset_dir=/home/docker/ahmed/datasets/flower_photos_tfrecord/flowers  \
    --dataset_name=flowers \
    --batch_size=64 \
    --dataset_split_name=train \
    --model_name=mobilenet_v1 \
    --train_image_size=224 \
    --checkpoint_path=./checkpoints/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt \
    --max_number_of_steps=10000 \
    --checkpoint_exclude_scopes=MobilenetV1/Logits \
    --trainable_scopes=MobilenetV1/Logits, MobilenetV1/MobilenetV1/Conv2d_13



python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=./train_dir/mobilenet_v1_flowers/test/model.ckpt-10000 \
    --dataset_dir=/home/docker/ahmed/datasets/flower_photos_tfrecord/flowers \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1 \
    --eval_image_size=224




#################################

python create_and_convert_dataset.py \
    --dataset_name=tylenol    \
    --images_dataset_dir=/home/docker/ahmed/datasets/pills/centredPills_Tylenol    \
    --tfrecords_dataset_dir=/home/docker/ahmed/datasets/tylenol_photos_tfrecord  \   
    --validation_percentage=20     
    --test_percentage=0 


python train_image_classifier.py \
    --train_dir=./train_dir/mobilenet_v1_tylenol/output_5 \
    --dataset_dir=/home/docker/ahmed/datasets/tylenol_photos_tfrecord/tylenol  \
    --dataset_name=tylenol \
    --batch_size=32 \
    --dataset_split_name=train \
    --model_name=mobilenet_v1 \
    --train_image_size=224 \
    --checkpoint_path=./checkpoints/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt \
    --max_number_of_steps=10000 \
    --checkpoint_exclude_scopes=MobilenetV1/Logits \
    --trainable_scopes=MobilenetV1/Logits

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=./train_dir/mobilenet_v1_tylenol/output_5/ \
    --dataset_dir=/home/docker/ahmed/datasets/tylenol_photos_tfrecord/tylenol \
    --dataset_name=tylenol \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1 \
    --eval_image_size=224

######################################

python create_and_convert_dataset.py \
    --dataset_name=blocks_20    \
    --images_dataset_dir=/home/docker/ahmed/datasets/blocks_cleaned_1   \
    --tfrecords_dataset_dir=/home/docker/ahmed/datasets/blocks_cleaned_photos_tfrecord --validation_percentage=20 --test_percentage=0 

python train_image_classifier.py \
    --train_dir=./train_dir/mobilenet_v1_blocks/output_11 \
    --dataset_dir=/home/docker/ahmed/datasets/blocks_cleaned_photos_tfrecord/blocks_20 \
    --dataset_name=blocks_20 \
    --batch_size=32 \
    --dataset_split_name=train \
    --model_name=mobilenet_v1_025 \
    --preprocessing_name=mobilenet_v1 \
    --train_image_size=224 \
    --checkpoint_path=./checkpoints/mobilenet_v1_0.25_224/mobilenet_v1_0.25_224.ckpt \
    --max_number_of_steps=15000 \
    --checkpoint_exclude_scopes=MobilenetV1/Logits \
    --trainable_scopes=MobilenetV1/Logits, MobilenetV1/MobilenetV1/Conv2d_13_depthwise, MobilenetV1/MobilenetV1/Conv2d_13_pointwise, MobilenetV1/MobilenetV1/Conv2d_12_depthwise, MobilenetV1/MobilenetV1/Conv2d_12_pointwise


python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=./train_dir/mobilenet_v1_blocks/output_10 \
    --dataset_dir=/home/docker/ahmed/datasets/blocks_cleaned_photos_tfrecord/blocks_20 \
    --dataset_name=blocks_20 \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1_025 \
    --preprocessing_name=mobilenet_v1 \
    --eval_image_size=224


    #########################################
mvNCCompile optimized_mobilenet.pb -in=input -on=MobilenetV1/Predictions/Reshape_1 -s 12 -o output_graph


# bazel build works from inside tensorflow_src. I think the problem is when you install tf from pip some files are missing. Only when you build tf from source the repo you use to build it is configured correctly. I think this is the situation when I use the tf docker container

#### Inception_V1

python train_image_classifier.py \
  --train_dir=./train_dir/inception_v1_flowers/output_3/ \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=/home/docker/ahmed/datasets/flowers \
  --model_name=inception_v1 \
  --checkpoint_path=./checkpoints/inception_v1/inception_v1.ckpt \
  --checkpoint_exclude_scopes=InceptionV1/Logits \
  --max_number_of_steps=10000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=./train_dir/inception_v1_flowers/output_3/model.ckpt-10000 \
    --dataset_dir=/home/docker/ahmed/datasets/flowers \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --model_name=inception_v1 \
    --eval_image_size=224


/tensorflow_src/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=./train_dir/inception_v1_flowers/model.ckpt-10000.meta


 python export_inference_graph.py --model_name=mobilenet_v1_025 --image_dir=  --output_file=./train_dir/mobilenet_v1_blocks/output_11/inference_graph_mobilenet_v1_025.pb --dataset_name=blocks_20


python /tensorflow_src/tensorflow/python/tools/freeze_graph.py \
    --input_graph=./train_dir/mobilenet_v1_blocks/output_11/inference_graph_mobilenet_v1_025.pb \
    --input_binary=true \
    --input_checkpoint=./train_dir/mobilenet_v1_blocks/output_11/model.ckpt-15000 \
    --output_graph=./train_dir/mobilenet_v1_blocks/output_11/frozen_graph_mobilenet_v1_025_blocks.pb \
    --output_node_names=MobilenetV1/Predictions/Reshape_1

/tensorflow_src/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=./train_dir/inception_v1_flowers/output_3/frozen_inceptionv1.pb \
--out_graph=./train_dir/inception_v1_flowers/output_3/optimized_graph.pb \
--inputs='input' \
--outputs='InceptionV1/Logits/Predictions/Reshape_1' \
--transforms='
 strip_unused_nodes(type=float, shape="1,224,224,3")
 remove_nodes(op=Identity, op=CheckNumerics, op=PlaceholderWithDefault)
 fold_batch_norms
 fold_old_batch_norms'


 mvNCCompile optimized_graph.pb -in=input -on=InceptionV1/Logits/Predictions/Reshape_1 -s 12 -o inference_inceptionv1



  python retrain_aug.py --image_dir=/home/docker/ahmed/datasets/blocks_cleaned_1 --dataset_name blocks --architecture mobilenet_0.25_224 --feature_vector L-2 --train_batch_size 128 --validation_batch_size -1 --learning_rate 0.001 --how_many_training_steps 500 --best_result_count_thresh 200 --append_filename 123 --update_bottleneck_file False--proper_data_partition True--hash_full_path False--random_crop 5 --random_brightness 10 --random_rotate 5
