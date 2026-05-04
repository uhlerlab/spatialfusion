version 1.0


workflow GenerateUniEmbeddingsForSpatialFusion {
  input {
    File adata
    File wsi
    File uni_weights

    Float? uni_mem_gb
    Int? cpu_cores
    Int? disk_gb
    Int? preemptible_tries

    String unimodal_embeddings_docker = "vanallenlab/unimodal-embeddings:workflow-0.2"
  }

  parameter_meta {
    adata: "AnnData (.h5ad) input. Spatial coordinates are expected in adata.obsm['spatial']."
    wsi: "Whole-slide image / H&E TIFF used to generate UNI image embeddings. TIFF / OME-TIFF format is expected."
    uni_weights: "UNI model weights file used for UNI image embeddings."
    uni_mem_gb: "Optional runtime override for UNI task memory in GB. Default is 14 GB."
    cpu_cores: "Optional runtime override for CPU cores requested by the task. Default is 2."
    disk_gb: "Optional runtime override for local disk requested by the UNI task."
    preemptible_tries: "Optional runtime override for the number of times Cromwell may try the task on preemptible/spot capacity before falling back to a regular VM."
    unimodal_embeddings_docker: "Docker image containing the embedding script and runtime dependencies."
  }

  call RunUniEmbedding {
    input:
      adata = adata,
      wsi = wsi,
      uni_weights = uni_weights,
      uni_mem_gb = uni_mem_gb,
      cpu_cores = cpu_cores,
      disk_gb = disk_gb,
      preemptible_tries = preemptible_tries,
      docker = unimodal_embeddings_docker
  }

  output {
    File uni_parquet = RunUniEmbedding.uni_parquet
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
    cpu_cores: "Optional runtime override for CPU cores requested by the task."
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
