version 1.0


workflow GenerateScgptEmbeddingsForSpatialFusion {
  input {
    File adata
    Boolean input_is_log_normalized

    Float? scgpt_mem_gb
    Int? cpu_cores
    Int? disk_gb
    Int? preemptible_tries

    String scgpt_embeddings_docker = "vanallenlab/scgpt-embeddings:workflow-0.1"
  }

  parameter_meta {
    adata: "AnnData (.h5ad) input used for scGPT embeddings."
    input_is_log_normalized: "Set to true if the selected AnnData expression layer is already log-normalized."
    scgpt_mem_gb: "Optional runtime override for scGPT task memory in GB. Default is 8 GB."
    cpu_cores: "Optional runtime override for CPU cores requested by the task. Default is 2."
    disk_gb: "Optional runtime override for local disk requested by the scGPT task."
    preemptible_tries: "Optional runtime override for the number of times Cromwell may try the task on preemptible/spot capacity before falling back to a regular VM."
    scgpt_embeddings_docker: "Docker image containing the embedding script, scGPT weights, and runtime dependencies."
  }

  call RunScgptEmbedding {
    input:
      adata = adata,
      input_is_log_normalized = input_is_log_normalized,
      scgpt_mem_gb = scgpt_mem_gb,
      cpu_cores = cpu_cores,
      disk_gb = disk_gb,
      preemptible_tries = preemptible_tries,
      docker = scgpt_embeddings_docker
  }

  output {
    File scgpt_parquet = RunScgptEmbedding.scgpt_parquet
  }
}


task RunScgptEmbedding {
  input {
    File adata
    Boolean input_is_log_normalized

    Int n_hvg = 1200
    Int scgpt_batch_size = 16
    String gene_col = "index"
    String layer_key = "X"
    Int seed = 42
    Int max_seq_len = 1200
    Int input_bins = 51
    String model_run = "pretrained"
    Int num_workers = 0

    Float? scgpt_mem_gb
    Int? cpu_cores
    Int? disk_gb
    Int? preemptible_tries

    String docker
  }

  parameter_meta {
    n_hvg: "Advanced setting for the number of highly variable genes used by scGPT preprocessing."
    scgpt_batch_size: "Advanced setting for scGPT inference batch size."
    gene_col: "Advanced setting for the adata.var gene-name column used by scGPT."
    layer_key: "Advanced setting for the AnnData expression layer used by scGPT."
    seed: "Advanced setting for the scGPT random seed."
    max_seq_len: "Advanced setting for scGPT maximum input sequence length."
    input_bins: "Advanced setting for the number of bins used in scGPT preprocessing."
    model_run: "Advanced setting forwarded to the underlying scFoundation scGPT wrapper."
    num_workers: "Advanced setting controlling scGPT preprocessing and tokenization worker count."
    scgpt_mem_gb: "Optional runtime override for scGPT task memory in GB."
    cpu_cores: "Optional runtime override for CPU cores requested by the task."
    disk_gb: "Optional runtime override for local disk requested by the scGPT task."
    preemptible_tries: "Number of times Cromwell may try this task on preemptible/spot capacity before falling back to a regular VM."
    docker: "Docker image containing the embedding script and runtime dependencies."
  }

  String scgpt_output_name = "scGPT.parquet"
  Float mem_gb = select_first([scgpt_mem_gb, 8.0])
  Int task_cpu_cores = select_first([cpu_cores, 2])
  Int task_preemptible_tries = select_first([preemptible_tries, 0])
  Int default_disk_gb = ceil(size(adata, "GB") + 20)

  command <<<
    set -eu -o pipefail

    python /app/embed_scgpt.py \
      --adata "~{adata}" \
      --output-dir "." \
      --scgpt-weights "/app/scgpt_weights" \
      --input-is-log-normalized ~{if (input_is_log_normalized) then "True" else "False"} \
      --n-hvg ~{n_hvg} \
      --scgpt-batch-size ~{scgpt_batch_size} \
      --gene-col "~{gene_col}" \
      --layer-key "~{layer_key}" \
      --seed ~{seed} \
      --max-seq-len ~{max_seq_len} \
      --input-bins ~{input_bins} \
      --model-run "~{model_run}" \
      --num-workers ~{num_workers} \
      --scgpt-output-name "~{scgpt_output_name}" \
      --overwrite
  >>>

  output {
    File scgpt_parquet = scgpt_output_name
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
