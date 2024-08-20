nohup sh -c "\
./run_serial.sh best_model CLEARSCOPE_E5 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_1min_neig_10 --detection.gnn_training.encoder.tgn.tgn_neighbor_size=10 && \
./run_serial.sh best_model CLEARSCOPE_E5 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_1min_dropout_0.1 --detection.gnn_training.encoder.graph_attention.dropout=0.1 && \
./run_serial.sh best_model CLEARSCOPE_E5 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_1min_h_32 --detection.gnn_training.node_out_dim=32 && \
./run_serial.sh best_model CLEARSCOPE_E5 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_1min_h_128 --detection.gnn_training.node_out_dim=128" &

nohup sh -c "\
./run_serial.sh best_model CLEARSCOPE_E5 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_1min_word2vec_h_32 --featurization.embed_nodes.emb_dim=32 && \
./run_serial.sh best_model CLEARSCOPE_E5 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_1min_word2vec_h_64 --featurization.embed_nodes.emb_dim=64 && \
./run_serial.sh best_model CLEARSCOPE_E5 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_1min_lr_0.0001 --detection.gnn_training.lr=0.0001 && \
./run_serial.sh best_model CLEARSCOPE_E5 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_1min_lr_0.000005 --detection.gnn_training.lr=0.000005" &

nohup sh -c "\
./run_serial.sh best_model CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_neig_10 --detection.gnn_training.encoder.tgn.tgn_neighbor_size=10 && \
./run_serial.sh best_model CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_dropout_0.1 --detection.gnn_training.encoder.graph_attention.dropout=0.1 && \
./run_serial.sh best_model CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_h_32 --detection.gnn_training.node_out_dim=32 && \
./run_serial.sh best_model CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_h_128 --detection.gnn_training.node_out_dim=128" &

nohup sh -c "\
./run_serial.sh best_model CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_word2vec_h_32 --featurization.embed_nodes.emb_dim=32 && \
./run_serial.sh best_model CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_word2vec_h_64 --featurization.embed_nodes.emb_dim=64 && \
./run_serial.sh best_model CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_lr_0.0001 --detection.gnn_training.lr=0.0001 && \
./run_serial.sh best_model CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --exp=CLEARSCOPE_E5_lr_0.000005 --detection.gnn_training.lr=0.000005" &
