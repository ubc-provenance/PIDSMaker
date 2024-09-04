nohup sh -c "\
./run_serial.sh orthrus THEIA_E3 --exp=E3_embedding --featurization.embed_nodes.used_method=hierarchical_hashing && \
./run_serial.sh orthrus THEIA_E5 --exp=E5_embedding --featurization.embed_nodes.used_method=hierarchical_hashing" &

nohup sh -c "\
./run_serial.sh orthrus THEIA_E3 --exp=E3_embedding_word2vec --featurization.embed_nodes.used_method=word2vec && \
./run_serial.sh orthrus THEIA_E5 --exp=E5_embedding_word2vec --featurization.embed_nodes.used_method=word2vec" &

nohup sh -c "\
./run_serial.sh orthrus THEIA_E3 --exp=E3_encoding --detection.gnn_training.encoder.tgn.use_memory=True --detection.gnn_training.decoder.predict_edge_type.used_method=kairos  && \
./run_serial.sh orthrus THEIA_E5 --exp=E5_encoding --detection.gnn_training.encoder.tgn.use_memory=True --detection.gnn_training.decoder.predict_edge_type.used_method=kairos" &

nohup sh -c "\
./run_serial.sh orthrus THEIA_E3 --exp=E3_clustering --detection.evaluation.node_evaluation.use_kmeans=False  && \
./run_serial.sh orthrus THEIA_E5 --exp=E5_clustering --detection.evaluation.node_evaluation.use_kmeans=False" &
