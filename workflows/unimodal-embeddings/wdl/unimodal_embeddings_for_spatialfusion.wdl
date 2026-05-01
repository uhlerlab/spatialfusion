version 1.0


workflow GenerateUnimodalEmbeddingsForSpatialFusion {
  input {
    # Shared inputs. Mode-specific files are optional at the workflow boundary so
    # scGPT-only and UNI-only runs do not require dummy files.
    File adata
    Boolean? input_is_log_normalized
    File? wsi
    File? uni_weights

    # Mode selection input. Most users will leave this as "both".
    String embedding_mode = "both"  # Allowed values: "scgpt", "uni", "both"

    # Optional runtime overrides. By default, this workflow requests 8 GB for scGPT,
    # 14 GB for UNI, and 2 CPU cores for both modes.
    Float? scgpt_mem_gb
    Float? uni_mem_gb
    Int? cpu_cores
    Int? disk_gb
    Int? preemptible_tries

    String unimodal_embeddings_docker = "vanallenlab/unimodal-embeddings:v0.1"
  }

  parameter_meta {
    adata: "Primary input. AnnData (.h5ad) used for scGPT embeddings and for the spatial coordinates consumed by UNI. Spatial coordinates are expected in adata.obsm['spatial']."
    wsi: "Required when embedding_mode is 'uni' or 'both'. Whole-slide image / H&E TIFF used to generate UNI image embeddings. TIFF / OME-TIFF format is expected."
    uni_weights: "Required when embedding_mode is 'uni' or 'both'. UNI model weights file used for UNI image embeddings."
    input_is_log_normalized: "Required when embedding_mode is 'scgpt' or 'both'. Set to true if the selected AnnData expression layer is already log-normalized."
    embedding_mode: "Which embeddings to generate: 'scgpt', 'uni', or 'both'. Defaults to 'both' for the common use case."
    scgpt_mem_gb: "Optional runtime override for scGPT task memory in GB. Default is 8 GB."
    uni_mem_gb: "Optional runtime override for UNI task memory in GB. Default is 14 GB."
    cpu_cores: "Optional runtime override for CPU cores requested by each task. Default is 2."
    disk_gb: "Optional runtime override for local disk requested by each task."
    preemptible_tries: "Optional runtime override for the number of times Cromwell may try a task on preemptible/spot capacity before falling back to a regular VM."
    unimodal_embeddings_docker: "Docker image containing the embedding script and runtime dependencies. Defaults to the published vanallenlab image for this workflow."
  }

  if (embedding_mode == "scgpt" || embedding_mode == "both") {
    call RunScgptEmbedding {
      input:
        adata = adata,
        input_is_log_normalized = select_first([input_is_log_normalized]),
        scgpt_mem_gb = scgpt_mem_gb,
        cpu_cores = cpu_cores,
        disk_gb = disk_gb,
        preemptible_tries = preemptible_tries,
        docker = unimodal_embeddings_docker
    }
  }

  if (embedding_mode == "uni" || embedding_mode == "both") {
    call RunUniEmbedding {
      input:
        adata = adata,
        wsi = select_first([wsi]),
        uni_weights = select_first([uni_weights]),
        uni_mem_gb = uni_mem_gb,
        cpu_cores = cpu_cores,
        disk_gb = disk_gb,
        preemptible_tries = preemptible_tries,
        docker = unimodal_embeddings_docker
    }
  }

  output {
    Array[File] scgpt_parquet = select_all([RunScgptEmbedding.scgpt_parquet])
    Array[File] uni_parquet = select_all([RunUniEmbedding.uni_parquet])
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
    cpu_cores: "Optional runtime override for CPU cores requested by each task."
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

    python /app/unimodal-embeddings.py \
      --mode scgpt \
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


task RunUniEmbedding {
  input {
    File adata
    File wsi
    File uni_weights

    Int uni_batch_size = 512
    String spatial_key = "spatial"

    Float? uni_mem_gb
    Int? cpu_cores
    Int? disk_gb
    Int? preemptible_tries

    String docker
  }

  parameter_meta {
    uni_batch_size: "Advanced setting for UNI patch embedding batch size."
    spatial_key: "Advanced setting for the adata.obsm key containing pixel coordinates used by UNI."
    uni_mem_gb: "Optional runtime override for UNI task memory in GB."
    cpu_cores: "Optional runtime override for CPU cores requested by each task."
    disk_gb: "Optional runtime override for local disk requested by the UNI task."
    preemptible_tries: "Number of times Cromwell may try this task on preemptible/spot capacity before falling back to a regular VM."
    docker: "Docker image containing the embedding script and runtime dependencies."
  }

  String uni_output_name = "UNI.parquet"
  Float mem_gb = select_first([uni_mem_gb, 14.0])
  Int task_cpu_cores = select_first([cpu_cores, 2])
  Int task_preemptible_tries = select_first([preemptible_tries, 0])
  Int default_disk_gb = ceil(size(adata, "GB") + size(wsi, "GB") + size(uni_weights, "GB") + 20)

  command <<<
    set -eu -o pipefail

    python /app/unimodal-embeddings.py \
      --mode uni \
      --adata "~{adata}" \
      --wsi "~{wsi}" \
      --output-dir "." \
      --uni-weights "~{uni_weights}" \
      --uni-batch-size ~{uni_batch_size} \
      --spatial-key "~{spatial_key}" \
      --uni-output-name "~{uni_output_name}" \
      --overwrite
  >>>

  output {
    File uni_parquet = uni_output_name
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
