# test
# ./main_grouped.py  -i /mnt/data/datasets/768/ -o /mnt/data/datasets/wm_simulated/distance_learning/g1g2_test --input_filelist /mnt/data/datasets/wm_simulated/filelists/first_50.txt --img_path_rel --ordered_ssim --group_size 1 --indent 4
# 
# ./main_grouped.py  -i /mnt/data/datasets/768/ -o /mnt/data/datasets/wm_simulated/distance_learning/g1g2 --input_filelist /mnt/data/datasets/wm_simulated/filelists/first_50k.txt --img_path_rel --ordered_ssim --group_size 1
# ./main_grouped.py  -i /mnt/data/datasets/768/ -o /mnt/data/datasets/wm_simulated/distance_learning/g3 --input_filelist /mnt/data/datasets/wm_simulated/filelists/next_40k.txt --img_path_rel --same_ref --group_size 2 --num_parameter_group 2
# ./main_grouped.py  -i /mnt/data/datasets/768/ -o /mnt/data/datasets/wm_simulated/distance_learning/g4 --input_filelist /mnt/data/datasets/wm_simulated/filelists/next_next_40k.txt --img_path_rel --same_ref --group_size 1 --ordered_distortion

./main_grouped.py  -i /mnt/data/datasets/768/ -o /mnt/data/datasets/wm_simulated/distance_learning/g1g2 --input_filelist /mnt/data/datasets/wm_simulated/filelists/first_10k.txt --img_path_rel --ordered_ssim --group_size 1
./main_grouped.py  -i /mnt/data/datasets/768/ -o /mnt/data/datasets/wm_simulated/distance_learning/g3 --input_filelist /mnt/data/datasets/wm_simulated/filelists/next_10k.txt --img_path_rel --same_ref --group_size 2 --num_parameter_group 2
./main_grouped.py  -i /mnt/data/datasets/768/ -o /mnt/data/datasets/wm_simulated/distance_learning/g4 --input_filelist /mnt/data/datasets/wm_simulated/filelists/next_next_10k.txt --img_path_rel --same_ref --group_size 1 --ordered_distortion

