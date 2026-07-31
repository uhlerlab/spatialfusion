version 1.0


workflow GenerateNicheformerEmbeddingsForSpatialFusion {
  input {
    File adata
    String technology = "xenium"

    Float? nicheformer_mem_gb
    Int? cpu_cores
    Int? disk_gb
    Int? preemptible_tries

    String nicheformer_embeddings_docker = "vanallenlab/nicheformer-embeddings:workflow-0.1"
  }

  parameter_meta {
    adata: "AnnData (.h5ad) input used for Nicheformer embeddings."
    technology: "Spatial transcriptomics technology used to select bundled Nicheformer reference defaults. Currently supported: xenium, cosmx, and merfish."
    nicheformer_mem_gb: "Optional runtime override for Nicheformer task memory in GB. Default is 24 GB."
    cpu_cores: "Optional runtime override for CPU cores requested by the task. Default is 4."
    disk_gb: "Optional runtime override for local disk requested by the Nicheformer task."
    preemptible_tries: "Optional runtime override for the number of times Cromwell may try the task on preemptible/spot capacity before falling back to a regular VM."
    nicheformer_embeddings_docker: "Docker image containing the Nicheformer embedding script, reference files, and runtime dependencies."
  }

  call RunNicheformerEmbedding {
    input:
      adata = adata,
      technology = technology,
      nicheformer_mem_gb = nicheformer_mem_gb,
      cpu_cores = cpu_cores,
      disk_gb = disk_gb,
      preemptible_tries = preemptible_tries,
      docker = nicheformer_embeddings_docker
  }

  output {
    File nicheformer_parquet = RunNicheformerEmbedding.nicheformer_parquet
  }
}


task RunNicheformerEmbedding {
  input {
    File adata
    String technology = "xenium"

    Int nicheformer_batch_size = 16
    Int max_seq_len = 1500
    Int aux_tokens = 30
    Int chunk_size = 1000
    Int num_workers = 0
    Int embedding_layer = -1
    Int seed = 42
    String split = "train"
    Int modality = 4
    Int species = 5

    Float? nicheformer_mem_gb
    Int? cpu_cores
    Int? disk_gb
    Int? preemptible_tries

    String docker
  }

  parameter_meta {
    technology: "Spatial transcriptomics technology used to select bundled Nicheformer reference defaults."
    nicheformer_batch_size: "Advanced setting for Nicheformer inference batch size."
    max_seq_len: "Advanced setting for Nicheformer maximum sequence length."
    aux_tokens: "Advanced setting for Nicheformer auxiliary token count."
    chunk_size: "Advanced setting for NicheformerDataset chunk size."
    num_workers: "Advanced setting controlling Nicheformer dataloader worker count."
    embedding_layer: "Advanced setting for the Nicheformer layer used for embedding extraction."
    seed: "Advanced setting for the random seed."
    split: "Advanced setting for the NicheformerDataset split label."
    modality: "Advanced setting for the Nicheformer modality metadata code."
    species: "Advanced setting for the Nicheformer species metadata code."
    nicheformer_mem_gb: "Optional runtime override for Nicheformer task memory in GB."
    cpu_cores: "Optional runtime override for CPU cores requested by the task."
    disk_gb: "Optional runtime override for local disk requested by the Nicheformer task."
    preemptible_tries: "Number of times Cromwell may try this task on preemptible/spot capacity before falling back to a regular VM."
    docker: "Docker image containing the embedding script, reference files, and runtime dependencies."
  }

  String nicheformer_output_name = "nicheformer.parquet"
  Float mem_gb = select_first([nicheformer_mem_gb, 24.0])
  Int task_cpu_cores = select_first([cpu_cores, 4])
  Int task_preemptible_tries = select_first([preemptible_tries, 0])
  Int default_disk_gb = ceil(size(adata, "GB") + 40)

  command <<<
    set -eu -o pipefail

    python /app/embed_nicheformer.py \
      --adata "~{adata}" \
      --output-dir "." \
      --technology "~{technology}" \
      --nicheformer-output-name "~{nicheformer_output_name}" \
      --nicheformer-batch-size ~{nicheformer_batch_size} \
      --max-seq-len ~{max_seq_len} \
      --aux-tokens ~{aux_tokens} \
      --chunk-size ~{chunk_size} \
      --num-workers ~{num_workers} \
      --embedding-layer ~{embedding_layer} \
      --seed ~{seed} \
      --split "~{split}" \
      --modality ~{modality} \
      --species ~{species} \
      --overwrite
  >>>

  output {
    File nicheformer_parquet = nicheformer_output_name
  }

  runtime {
    docker: docker
    memory: mem_gb + " GB"
    cpu: task_cpu_cores
    disks: "local-disk " + select_first([disk_gb, default_disk_gb]) + " HDD"
    bootDiskSizeGb: 20
    gpuType: "nvidia-tesla-t4"
    gpuCount: 1
    preemptible: task_preemptible_tries
  }
}
