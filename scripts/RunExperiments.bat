rem dataset approach network_type final_points num_classes norm_type epochs batch_size

call RunPython "IPN_Hand" "None" "LSTM" 970 13 "L0" 50 100
call RunPython "IPN_Hand" "None" "CNN" 970 13 "L0" 50 100

call RunPython "IPN_Hand" "Selection" "LSTM" 150 13 "L0" 50 100
call RunPython "IPN_Hand" "Selection" "CNN" 150 13 "L0" 50 100

call ResultsToExcel "IPN_Hand" "None" "LSTM" 970 13 "L0" 50 100
call ResultsToExcel "IPN_Hand" "None" "CNN" 970 13 "L0" 50 100

call ResultsToExcel "IPN_Hand" "Selection" "LSTM" 150 13 "L0" 50 100
call ResultsToExcel "IPN_Hand" "Selection" "CNN" 150 13 "L0" 50 100
