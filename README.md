# ptag-detector

## Version
```
TF Version : 2.5.0
Cuda : 11.4.1
Cudnn: 8.2.2.26
Nvidia: 471.68
```


## Installation
#### Step 1: Partitioning the annotated images to test and train directory
**Script-location: TensorFlow-2.0/scripts/pre-processing**
```sh
python partition_dataset.py -x -i ../../workspace/training/images -r 0.2
```
#### Step 2: Generating TFRecord Test and Train
**Script-location: TensorFlow-2.0/scripts/pre-processing**
``` sh
python generate_tfrecord.py -x D:\Workspace\TensorFlow-2.0\workspace\training\images\train -l D:\Workspace\TensorFlow-2.0\workspace\training\annotations\label_map.pbtxt -o D:\Workspace\TensorFlow-2.0\workspace\training\annotations\train.record
python generate_tfrecord.py -x D:\Workspace\TensorFlow-2.0\workspace\training\images\test -l D:\Workspace\TensorFlow-2.0\workspace\training\annotations\label_map.pbtxt -o D:\Workspace\TensorFlow-2.0\workspace\training\annotations\test.record
```
#### Step 3: Initiate Training
**Script-location: TensorFlow-2.0/workspace/training**

```sh
python model_main_tf2.py --model_dir=models/ssd_mobilenet_v2_fpnlite_320x320 --pipeline_config_path=models/ssd_mobilenet_v2_fpnlite_320x320/pipeline.config --checkpoint_dir=models/ssd_mobilenet_v2_fpnlite_320x320
```
#### Step 4: Export Model (Generate Saved_Model)
**Script-location: TensorFlow-2.0/workspace/training**
```sh
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path D:\Workspace\TensorFlow-2.0\workspace\training\models\ssd_mobilenet_v2_fpnlite_320x320\pipeline.config --trained_checkpoint_dir D:\Workspace\TensorFlow-2.0\workspace\training\models\ssd_mobilenet_v2_fpnlite_320x320 --output_directory D:\Workspace\TensorFlow-2.0\workspace\training\exported-models\ptag-detector-model
```
#### Step 5: Convert Saved Model to TFLite
**Script-location: TensorFlow-2.0/workspace/training**
```sh
python export_tflite_graph_tf2.py --pipeline_config_path D:\Workspace\TensorFlow-2.0\workspace\training\models\ssd_mobilenet_v2_fpnlite_320x320\pipeline.config --trained_checkpoint_dir D:\Workspace\TensorFlow-2.0\workspace\training\models\ssd_mobilenet_v2_fpnlite_320x320 --output_directory D:\Workspace\TensorFlow-2.0\workspace\training\exported-models\ptag-detector-model
```
#### Step 6: Post Training Quantization(This Script may not work properly for TF 2.5.0, Please refer to Tensorflow website)

**Script-location: TensorFlow-2.0/workspace/training**
```sh
tflite_convert --graph_def_file=tflite_model/tflite_graph.pb --output_file=tflite_model/detect.tflite --output_format=TFLITE --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_dev_values=127 --change_concat_input_ranges=false --allow_custom_ops
```
#### Step 7: Convert to edgetpu compatible

```sh
docker run -ti -v e:/:/data ubuntu /bin/bash
:/usr/bin# edgetpu_compiler ../../tflite_model/detect.tflite -o ../../
```






