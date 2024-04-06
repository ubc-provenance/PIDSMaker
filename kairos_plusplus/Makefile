SHELL := /bin/bash
prepare:
	mkdir -p artifact
	mkdir -p ./artifact/graphs/nx/train/
	mkdir -p ./artifact/graphs/nx/val/
	mkdir -p ./artifact/graphs/nx/test/

create_database:
	python create_database.py

construct_graphs:
	python graph_constructor.py

prepare_random_walks:
	mv ./artifact/graphs/nx/e5-graph-08-00_00_00 ./artifact/graphs/nx/train/
	mv ./artifact/graphs/nx/e5-graph-09-00_00_00 ./artifact/graphs/nx/train/
	mv ./artifact/graphs/nx/e5-graph-11-00_00_00 ./artifact/graphs/nx/val/
	mv ./artifact/graphs/nx/graph_5_14/* ./artifact/graphs/nx/test/
	mv ./artifact/graphs/nx/graph_5_15/* ./artifact/graphs/nx/test/

run_random_walk:
	python darpa_preprocess.py --train ./artifact/graphs/nx/train/ \
							  --validation ./artifact/graphs/nx/val/ \
							  --test ./artifact/graphs/nx/test/ \
							  --walk_len 30

node_embedding:
	python node_embedding.py --dataset ./artifact/preprocessed/corpus/train.csv --adjacency-dir ./artifact/preprocessed/train_adj_dir/ --corpus ./artifact/preprocessed/corpus/train.csv --corpus-dir ./artifact/preprocessed/training_set_corpus/ --model-output model.bin --iter 100
	cp ./artifact/w2v_models/nodelabel2vec ./artifact/w2v_models/nodelabel2vec_train
	python node_embedding.py --dataset ./artifact/preprocessed/corpus/validation.csv  --model-input ./artifact/w2v_models/model.bin --matrix-input ./artifact/w2v_models/matrix.bin --adjacency-dir ./artifact/preprocessed/val_adj_dir/ --corpus-dir ./artifact/preprocessed/validation_set_corpus/
	cp ./artifact/w2v_models/nodelabel2vec ./artifact/w2v_models/nodelabel2vec_train_val
	python node_embedding.py --dataset ./artifact/preprocessed/corpus/test.csv  --model-input ./artifact/w2v_models/model.bin --matrix-input ./artifact/w2v_models/matrix.bin --adjacency-dir ./artifact/preprocessed/test_adj_dir/ --corpus-dir ./artifact/preprocessed/testing_set_corpus/

graph_vectorization:
	python embedding.py

prepare_test_sets:
	mkdir -p ./artifact/graphs/vectorized/test/graph_5_14/
	mkdir -p ./artifact/graphs/vectorized/test/graph_5_15/
	mv ./artifact/graphs/vectorized/test/2019-05-14* ./artifact/graphs/vectorized/test/graph_5_14/
	mv ./artifact/graphs/vectorized/test/2019-05-15* ./artifact/graphs/vectorized/test/graph_5_15/

train_gnn_2mlp:
	python train_gnn_2mlp.py

test_gnn_2mlp:
	python test_gnn_2mlp.py

train_lof:
	python train_lof.py -node2vec_train_val_path ./artifact/w2v_models/nodelabel2vec_train_val -lof_path ./artifact/lof_model


anomalous_queue_construction:
	python anomalous_queue_construction_edgebase.py -node2vec_path ./artifact/w2v_models/nodelabel2vec \
											-node2vec_train_val_path ./artifact/w2v_models/nodelabel2vec_train_val \
											-lof_path ./artifact/lof_model

tw_evaluate:
	python tw_evaluation.py

node_evaluate:
	python node_evaluation_edgebase.py


run: prepare construct_graphs prepare_random_walks run_random_walk node_embedding graph_vectorization prepare_test_sets train_gnn test_gnn

run_2mlp: run_random_walk node_embedding graph_vectorization prepare_test_sets train_gnn_2mlp test_gnn_2mlp

clean:
	rm -r ./artifact/