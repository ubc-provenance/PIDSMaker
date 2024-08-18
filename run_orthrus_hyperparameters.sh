nohup sh -c "\
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_neig_5 --detection.gnn_training.encoder.tgn.tgn_neighbor_size=5 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_neig_10 --detection.gnn_training.encoder.tgn.tgn_neighbor_size=10 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_neig_20 --detection.gnn_training.encoder.tgn.tgn_neighbor_size=20 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_neig_50 --detection.gnn_training.encoder.tgn.tgn_neighbor_size=50 && \

./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_h_16 --detection.gnn_training.node_out_dim=16 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_h_32 --detection.gnn_training.node_out_dim=32 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_h_64 --detection.gnn_training.node_out_dim=64" &
nohup sh -c "\
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_h_128 --detection.gnn_training.node_out_dim=128 && \

./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_word2vec_h_32 --featurization.embed_nodes.emb_dim=32 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_word2vec_h_64 --featurization.embed_nodes.emb_dim=64 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_word2vec_h_256 --featurization.embed_nodes.emb_dim=256 && \

./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_lr_0.0001 --detection.gnn_training.lr=0.0001" &
nohup sh -c "\
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_lr_0.00001 --detection.gnn_training.lr=0.00001 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_lr_0.000001 --detection.gnn_training.lr=0.000001 && \

./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_dropout_0 --detection.gnn_training.encoder.graph_attention.dropout=0.0 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_dropout_0.1 --detection.gnn_training.encoder.graph_attention.dropout=0.1 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_dropout_0.25 --detection.gnn_training.encoder.graph_attention.dropout=0.25 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --exp=CADETS_E3_dropout_0.5 --detection.gnn_training.encoder.graph_attention.dropout=0.5" &
