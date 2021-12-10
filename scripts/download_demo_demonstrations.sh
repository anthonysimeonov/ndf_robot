if [ -z $NDF_SOURCE_DIR ]; then echo 'Please source "ndf_env.sh" first'
else
wget -O ndf_release_demo_demonstrations.tar.gz https://www.dropbox.com/s/rcnkzkrd0gyvvxh/ndf_release_demo_demonstrations.tar.gz?dl=0 
mkdir -p $NDF_SOURCE_DIR/data/demos
mv ndf_release_demo_demonstrations.tar.gz $NDF_SOURCE_DIR/data/demos
cd $NDF_SOURCE_DIR/data/demos
tar -xzf ndf_release_demo_demonstrations.tar.gz
rm ndf_release_demo_demonstrations.tar.gz
echo "Robot demonstrations for NDF copied to $NDF_SOURCE_DIR/data/demos"
fi
