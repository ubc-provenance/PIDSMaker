nohup sh -c "\
./run_serial.sh orthrus THEIA_E3 && \
./run_serial.sh orthrus CADETS_E3 --detection.gnn_training.num_epochs=20 --detection.gnn_training.encoder.graph_attention.dropout=0.25 && \
./run_serial.sh orthrus CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.encoder.graph_attention.dropout=0.1" &
nohup sh -c "\
./run_serial.sh orthrus THEIA_E5 --detection.gnn_training.lr=0.000005 && \
./run_serial.sh orthrus CADETS_E5 --detection.gnn_training.node_out_dim=128 --preprocessing.build_graphs.time_window_size=3 && \
./run_serial.sh orthrus CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --detection.gnn_training.lr=0.0001" &
