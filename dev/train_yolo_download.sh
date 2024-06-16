python train.py --mode yolo \
                --name sky_detection \
                --dataset_path label-studio-wqoeufhvbozubvfd/sky_detection/ \
                --weights_path machine-learning-models-kehrdhjlifhliug/weights1/yolov8s.pt \
                --weights_uploader_bucket_name machine-learning-models-kehrdhjlifhliug \
                --weights_uploader_upload_dir_name weights1/yolo \
                --override epochs=2 batch=4
