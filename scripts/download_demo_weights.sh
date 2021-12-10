if [ -z $NDF_SOURCE_DIR ]; then echo 'Please source "ndf_env.sh" first'
else
mkdir $NDF_SOURCE_DIR/model_weights
wget -O $NDF_SOURCE_DIR/model_weights/ndf_demo_mug_weights.pth https://www.dropbox.com/s/buhbw9q61psizgp/ndf_demo_mug_weights.pth?dl=0
wget -O $NDF_SOURCE_DIR/model_weights/multi_category_weights.pth https://www.dropbox.com/s/hm4hty56ldu1wb5/multi_category_weights.pth?dl=0
fi