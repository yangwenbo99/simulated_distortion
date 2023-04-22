./main.py  -i /mnt/data/datasets/768/ -o /mnt/data/datasets/wm_simulated/distorted -r 3 --input_filelistg /mnt/data/datasets/wm_simulated/filelist_50k.txt --img_path_rel

# Call json_to_csv.py to convert the json file to csv files, including traing.csv and test.csv
python json_to_csv.py \
	-i /mnt/data/datasets/wm_simulated/distorted/res.json \
	-o /mnt/data/datasets/wm_simulated/full.csv \
	--output-train /mnt/data/datasets/wm_simulated/train.csv \
	--output-test /mnt/data/datasets/wm_simulated/test.csv 

# Call lmdb_pack.py
# dump every generated images, as well as reference images, into a lmdb databasLe
python lmdb_pack.py \
	-d /mnt/data/datasets/wm_simulated/distorted \
	-r /mnt/data/datasets/768 \
	-o /mnt/data/datasets/wm_simulated/lmdb \
	--csv /mnt/data/datasets/wm_simulated/full.csv

# Test the generated lmdb database
python test_lmdb_dataset.py \
	--db_dir /mnt/data/datasets/wm_simulated/lmdb/data/ \
	--csv_path /mnt/data/datasets/wm_simulated/full.csv 
