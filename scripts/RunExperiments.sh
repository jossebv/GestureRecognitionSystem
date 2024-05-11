# dataset approach network_type final_points num_classes norm_type epochs batch_size

./RunPython.sh IPN_HAND None LSTM 970 13 L0 50 100
./RunPython.sh IPN_HAND None CNN 970 13 L0 50 100

./RunPython.sh IPN_HAND Selection LSTM 150 13 L0 50 100
./RunPython.sh IPN_HAND Selection CNN 150 13 L0 50 100

./ResultstoExcel.sh IPN_HAND None LSTM 970 13 L0 50 100 "Padding_both_edges"
./ResultstoExcel.sh IPN_HAND None CNN 970 13 L0 50 100 "Padding_both_edges"

./ResultstoExcel.sh IPN_HAND Selection LSTM 150 13 L0 50 100 "Padding_both_edges"
./ResultstoExcel.sh IPN_HAND Selection CNN 150 13 L0 50 100 "Padding_both_edges"
