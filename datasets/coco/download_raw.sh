mkdir raw
cd raw
mkdir images
cd images

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2014.zip

unzip train2014.zip
unzip val2014.zip
unzip test2014.zip

rm train2014.zip
rm val2014.zip
rm test2014.zip

cd ../
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/annotations/image_info_test2014.zip

unzip annotations_trainval2014.zip
unzip image_info_test2014.zip

rm annotations_trainval2014.zip
rm image_info_test2014.zip
mv annotations/captions_train2014.json annotations/captions_train2014_raw.json
mv annotations/captions_val2014.json annotations/captions_val2014_raw.json
rm annotations/image_info_test2014.json
rm annotations/instances_train2014.json
rm annotations/instances_val2014.json
rm annotations/person_keypoints_train2014.json
rm annotations/person_keypoints_val2014.json

wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip

rm dataset_flickr30k.json
rm dataset_flickr8k.json
rm caption_dataset.zip
