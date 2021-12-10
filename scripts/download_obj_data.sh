if [ -z $NDF_SOURCE_DIR ]; then echo 'Please source "ndf_env.sh" first'
else
wget -O ndf_obj_assets.tar.gz https://www.dropbox.com/s/831szjnb8l7gbdh/ndf_obj_assets.tar.gz?dl=0
mv ndf_obj_assets.tar.gz $NDF_SOURCE_DIR/descriptions
cd $NDF_SOURCE_DIR/descriptions
tar -xzf ndf_obj_assets.tar.gz
rm ndf_obj_assets.tar.gz
echo "Object models for NDF copied to $NDF_SOURCE_DIR/descriptions"

cd $NDF_SOURCE_DIR
wget -O ndf_other_assets.tar.gz https://www.dropbox.com/s/fopyjjm3fpc3k7i/ndf_other_assets.tar.gz?dl=0
mkdir $NDF_SOURCE_DIR/assets
mv ndf_other_assets.tar.gz $NDF_SOURCE_DIR/assets
cd $NDF_SOURCE_DIR/assets
tar -xzf ndf_other_assets.tar.gz
rm ndf_other_assets.tar.gz
echo "Additional object-related assets copied to $NDF_SOURCE_DIR/assets"
fi
