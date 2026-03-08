FIMH: Fungal Identification based on MinHash and HNSW
Overview
This repository contains a high-performance pipeline for rapid biological sequence (DNA/RNA) retrieval and taxonomic classification. It addresses the computational bottlenecks of searching through massive FASTA datasets by combining MinHash for sequence feature extraction (dimensionality reduction) and HNSW (Hierarchical Navigable Small World) graphs for fast Approximate Nearest Neighbor (ANN) search.

The system evaluates search accuracy across seven taxonomic ranks: Kingdom, Phylum, Class, Order, Family, Genus, and Species (k, p, c, o, f, g, s).

Key Features
K-mer Shingling & MinHash: Converts long genomic sequences into k-mers (shingles) and generates MinHash signatures to efficiently approximate Jaccard similarity.

HNSW Graph Search: Utilizes a custom multi-layer graph structure to achieve sub-linear time complexity for nearest neighbor sequence retrieval.

Optimized Storage (HDF5): Handles massive sequence similarity matrices (e.g., 40,000+ sequences) using high-compression HDF5 storage, preventing memory overflows associated with standard pickle serialization.

Taxonomic Accuracy Evaluation: Automatically parses FASTA headers to compare the taxonomy of the queried sequence against its retrieved nearest neighbors, outputting match probabilities at all seven taxonomic levels.

File Structure & Evolution
The project files represent the developmental evolution of the pipeline, focusing on scaling and memory optimization:

hnsw_origin.py: The foundational, pure-algorithmic implementation of the HNSW graph search.

hnsw.py: The initial integration with bioinformatics workflows. It includes FASTA parsing via Biopython, shingle generation, MinHash signature extraction, and taxonomic evaluation.

hnsw-1.py: An intermediate memory-optimized version. It implements batch processing (batch_size = 1000) to save and load large similarity matrices as .pkl.gz files to bypass RAM limitations.

hnsw-2.py: (Recommended) The final and most optimized version. It replaces standard serialization with h5py to store feature and similarity matrices in .h5 format, offering superior read/write speeds and optimal memory management for large datasets.
