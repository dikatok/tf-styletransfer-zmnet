#!/bin/bash

echo "Submitting a Cloud ML Engine job..."

REGION="your_region"
TIER="your_tier"
BUCKET="your_bucket"

MODEL_NAME="your_model_name"

PACKAGE_PATH="ml_engine"
TRAIN_FILES="gs://${BUCKET}/path/to/training_files"
MODEL_DIR="gs://${BUCKET}/path/to/model_dir"

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME="train_${MODEL_NAME}_${TIER}_${CURRENT_DATE}"

STYLE_IMG="gs://${BUCKET}/path/to/style_image"
VGG_PATH="gs://${BUCKET}/path/to/vgg_weights"

#uncomment line below to restart the training
#gsutil -m rm -r ${MODEL_DIR}

gcloud ml-engine jobs submit training ${JOB_NAME} \
        --runtime-version=1.7 \
        --job-dir=${MODEL_DIR}\
        --region=${REGION} \
        --scale-tier=${TIER} \
        --module-name="${PACKAGE_PATH}.task" \
        --package-path=${PACKAGE_PATH}  \
        --config=config.yaml \
        -- \
        --train_files=${TRAIN_FILES} \
        --vgg_path=${VGG_PATH} \
        --style_img=${STYLE_IMG}
