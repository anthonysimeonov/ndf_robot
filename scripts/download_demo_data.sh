if [ -z $NDF_SOURCE_DIR ]; then echo 'Please source "ndf_env.sh" first'
else
mkdir -p $NDF_SOURCE_DIR/descriptions/demo_objects
wget -O ndf_demo_assets.tar.gz https://www.dropbox.com/s/0igxwn38xvgk74l/ndf_demo_assets.tar.gz?dl=0
mv ndf_demo_assets.tar.gz $NDF_SOURCE_DIR/descriptions/demo_objects
cd $NDF_SOURCE_DIR/descriptions/demo_objects
tar -xzf ndf_demo_assets.tar.gz
rm ndf_demo_assets.tar.gz
echo "Object models for NDF demo copied to $NDF_SOURCE_DIR/descriptions/demo_objects"
fi