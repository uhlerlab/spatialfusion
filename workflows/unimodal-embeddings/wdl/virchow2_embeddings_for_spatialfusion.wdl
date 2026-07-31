version 1.0


workflow GenerateVirchow2EmbeddingsForSpatialFusion {
  input {
    File adata
    File wsi
    File virchow2_weights

    Float? virchow2_mem_gb
    Int? cpu_cores
    Int? disk_gb
    Int? preemptible_tries

    String he_embeddings_docker = "vanallenlab/he-embeddings:workflow-0.1"
  }

  parameter_meta {
    adata: "AnnData (.h5ad) input. Spatial coordinates are expected in adata.obsm['spatial']."
    wsi: "Whole-slide image / H&E TIFF used to generate Virchow2 image embeddings. TIFF / OME-TIFF format is expected."
    virchow2_weights: "Virchow2 model weights file, usually model.safetensors or pytorch_model.bin, from Paige AI."
    virchow2_mem_gb: "Optional runtime override for Virchow2 task memory in GB. Default is 32 GB."
    cpu_cores: "Optional runtime override for CPU cores requested by the task. Default is 4."
    disk_gb: "Optional runtime override for local disk requested by the Virchow2 task."
    preemptible_tries: "Optional runtime override for the number of times Cromwell may try the task on preemptible/spot capacity before falling back to a regular VM."
    he_embeddings_docker: "Docker image containing the H&E embedding scripts and runtime dependencies."
  }

  call RunVirchow2Embedding {
    input:
      adata = adata,
      wsi = wsi,
      virchow2_weights = virchow2_weights,
      virchow2_mem_gb = virchow2_mem_gb,
      cpu_cores = cpu_cores,
      disk_gb = disk_gb,
      preemptible_tries = preemptible_tries,
      docker = he_embeddings_docker
  }

  output {
    File virchow2_parquet = RunVirchow2Embedding.virchow2_parquet
  }
}


task RunVirchow2Embedding {
  input {
    File adata
    File wsi
    File virchow2_weights

    Int virchow2_batch_size = 32
    String spatial_key = "spatial"

    Float? virchow2_mem_gb
    Int? cpu_cores
    Int? disk_gb
    Int? preemptible_tries

    String docker
  }

  parameter_meta {
    virchow2_batch_size: "Advanced setting for Virchow2 patch embedding batch size."
    spatial_key: "Advanced setting for the adata.obsm key containing pixel coordinates used by Virchow2."
    virchow2_mem_gb: "Optional runtime override for Virchow2 task memory in GB."
    cpu_cores: "Optional runtime override for CPU cores requested by the task."
    disk_gb: "Optional runtime override for local disk requested by the Virchow2 task."
    preemptible_tries: "Number of times Cromwell may try this task on preemptible/spot capacity before falling back to a regular VM."
    docker: "Docker image containing the embedding script and runtime dependencies."
  }

  String virchow2_output_name = "Virchow2.parquet"
  Float mem_gb = select_first([virchow2_mem_gb, 32.0])
  Int task_cpu_cores = select_first([cpu_cores, 4])
  Int task_preemptible_tries = select_first([preemptible_tries, 0])
  Int default_disk_gb = ceil(size(adata, "GB") + size(wsi, "GB") + size(virchow2_weights, "GB") + 50)

  command <<<
    set -eu -o pipefail

    python /app/embed_virchow2.py \
      --adata "~{adata}" \
      --wsi "~{wsi}" \
      --output-dir "." \
      --virchow2-weights "~{virchow2_weights}" \
      --virchow2-batch-size ~{virchow2_batch_size} \
      --spatial-key "~{spatial_key}" \
      --virchow2-output-name "~{virchow2_output_name}" \
      --overwrite
  >>>

  output {
    File virchow2_parquet = virchow2_output_name
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
