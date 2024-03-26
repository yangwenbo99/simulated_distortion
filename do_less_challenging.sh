#- ./main.py  -i /mnt/data/datasets/768/ -o /mnt/data/datasets/wm_simulated/distorted_less_challenging -r 3 --input_filelist /mnt/data/datasets/wm_simulated/filelist_50k.txt --img_path_rel -c ./config_less_challenging.json

# Call json_to_csv.py to convert the json file to csv files, including traing.csv and test.csv
#- python json_to_csv.py \
	#- -i /mnt/data/datasets/wm_simulated/distorted_less_challenging/res.json \
	#- -o /mnt/data/datasets/wm_simulated/full_less_challenging.csv \
	#- --output-train /mnt/data/datasets/wm_simulated/train_less_challenging.csv \
	#- --output-test /mnt/data/datasets/wm_simulated/test_less_challenging.csv 
# The counting in json_to_csv.py might be incorrect, but screw it... 

# Call lmdb_pack.py
# dump every generated images, as well as reference images, into a lmdb databasLe
python lmdb_pack.py \
	-d /mnt/data/datasets/wm_simulated/distorted_less_challenging \
	-r /mnt/data/datasets/768 \
	-o /mnt/data/datasets/wm_simulated/lmdb_less_challenging \
	-f /mnt/data/datasets/wm_simulated/full_less_challenging.csv

# Test the generated lmdb database
python test_lmdb_dataset.py \
	--db_dir /mnt/data/datasets/wm_simulated/lmdb_less_challenging/data/ \
	--csv_path /mnt/data/datasets/wm_simulated/full_less_challenging.csv 

