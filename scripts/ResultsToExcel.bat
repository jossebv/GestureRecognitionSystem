
set dataset=%1
set approach=%2
set network_type=%3
set final_points=%4
set num_classes=%5
set norm_type=%6
set epochs=%7
set batch_size=%8

mkdir "../output/%dataset%/%approach%/norm_%norm_type%/%network_type%/final_points_%final_points%/num_classes_%num_classes%/epochs_%epochs%/batch_size_%batch_size%/"
python ../python/Results_to_Excel.py ../output/%dataset%/%approach%/norm_%norm_type%/%network_type%/final_points_%final_points%/num_classes_%num_classes%/epochs_%epochs%/batch_size_%batch_size%/ %dataset% %approach% %network_type% %final_points% %num_classes% %norm_type% %epochs% %batch_size%