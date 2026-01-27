Featurization methods are used to transform textual attributes of entities (e.g. file paths, process command lines, socket IP addresses and ports) into a vector.
Some methods like `word2vec` and `doc2vec` learn this vector from the text corpus, while others like `hierarchical_hashing` compute the vector in a deterministic way. 

Other methods like `only_type` and `only_ones` simply skip this embedding step and assign either a one-hot encoded type or ones to each entity. Those methods thus do not require any specific argument.
In all methods, the resulting vectors are used as node features during the `training` task.

--8<-- "scripts/args/args_featurizations.md"
