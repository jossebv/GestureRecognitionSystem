dataset=$1
approach=$2
network_type=$3
final_points=$4
num_classes=$5
norm_type=$6
epochs=$7
batch_size=$8
extraSettings=$9

# mkdir ../output/$dataset/$approach/norm_$norm_type/$network_type/final_points_$final_points/num_classes_$num_classes/epochs_$epochs/batch_size_$batch_size/

python3 ../python/Results_to_Excel.py ../output/$dataset/$approach/norm_$norm_type/$network_type/final_points_$final_points/num_classes_$num_classes/epochs_$epochs/batch_size_$batch_size/ $dataset $approach $network_type $final_points $num_classes $norm_type $epochs $batch_size ${extraSettings}