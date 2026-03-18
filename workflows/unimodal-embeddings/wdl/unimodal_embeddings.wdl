version 1.0


workflow GenerateUnimodalEmbeddings {
  input {
    # Primary user inputs for the common case: run both scGPT and UNI embeddings from one
    # AnnData object plus one H&E / WSI image. These are the inputs most users should set.
    File adata
    File wsi
    File uni_weights
    Boolean input_is_log_normalized

    # Mode selection input. Most users will leave this as "both".
    # If "both" is requested, the workflow scatters over ["scgpt", "uni"]
    # and launches two separate jobs in parallel.
    String embedding_mode = "both"  # Allowed values: "scgpt", "uni", "both"

    # scGPT weights in VA lab unstructured storage bucket.
    File scgpt_weights = "gs://fc-d0a0b6ac-b16f-47ca-99eb-49d88a2ba4c2/scgpt_weights/scgpt_weights.tar.gz"

    # Advanced script parameters with WDL defaults matching the current argparse defaults.
    # Most users should not need to change these.
    Int n_hvg = 1200
    Int scgpt_batch_size = 16
    Int uni_batch_size = 512
    String gene_col = "index"
    String layer_key = "X"
    Int seed = 42
    Int max_seq_len = 1200
    Int input_bins = 51
    String model_run = "pretrained"
    Int num_workers = 0
    String spatial_key = "spatial"

    # Runtime options
    Float run_mem_gb = 32
    Int run_cpu_cores = 2
    Int? run_disk_gb

    String unimodal_embeddings_docker = "vanallenlab/unimodal-embeddings:v0.1"
  }

  parameter_meta {
    adata: "Primary input. AnnData (.h5ad) used for scGPT embeddings and for the spatial coordinates consumed by UNI."
    wsi: "Primary input. Whole-slide image / H&E TIFF used to generate UNI image embeddings."
    uni_weights: "Primary input for the common 'both' or UNI-only use case. This workflow is optimized for runs that include UNI embeddings, so these weights are required."
    input_is_log_normalized: "Primary input. Set to true if the selected AnnData expression layer is already log-normalized."
    embedding_mode: "Which embeddings to generate: 'scgpt', 'uni', or 'both'. Defaults to 'both' for the common use case."
    scgpt_weights: "scGPT weights archive. Defaults to a public gs:// path so most users do not need to provide it explicitly."
    n_hvg: "Advanced setting for the number of highly variable genes used by scGPT preprocessing."
    scgpt_batch_size: "Advanced setting for scGPT inference batch size."
    uni_batch_size: "Advanced setting for UNI patch embedding batch size."
    gene_col: "Advanced setting for the adata.var gene-name column used by scGPT."
    layer_key: "Advanced setting for the AnnData expression layer used by scGPT."
    seed: "Advanced setting for the scGPT random seed."
    max_seq_len: "Advanced setting for scGPT maximum input sequence length."
    input_bins: "Advanced setting for the number of bins used in scGPT preprocessing."
    model_run: "Advanced setting forwarded to the underlying scFoundation scGPT wrapper."
    num_workers: "Advanced setting controlling scGPT preprocessing and tokenization worker count."
    spatial_key: "Advanced setting for the adata.obsm key containing pixel coordinates used by UNI."
    run_mem_gb: "Runtime setting for memory requested by each scattered embedding task."
    run_cpu_cores: "Runtime setting for CPU cores requested by each scattered embedding task."
    run_disk_gb: "Optional runtime override for local disk requested by each scattered embedding task."
    unimodal_embeddings_docker: "Docker image containing the embedding script and runtime dependencies. Defaults to the published vanallenlab image for this workflow."
  }

  # Build a mode list and scatter over it so scGPT and UNI can run as independent jobs.
  # This avoids coupling both embeddings into one Python invocation and lets WDL schedule them in parallel.
  Array[String] requested_modes = if (embedding_mode == "both") then ["scgpt", "uni"] else [embedding_mode]

  scatter (mode in requested_modes) {
    call RunUnimodalEmbedding {
      input:
        adata = adata,
        wsi = wsi,
        uni_weights = uni_weights,
        mode = mode,
        input_is_log_normalized = input_is_log_normalized,
        scgpt_weights = scgpt_weights,
        n_hvg = n_hvg,
        scgpt_batch_size = scgpt_batch_size,
        uni_batch_size = uni_batch_size,
        gene_col = gene_col,
        layer_key = layer_key,
        seed = seed,
        max_seq_len = max_seq_len,
        input_bins = input_bins,
        model_run = model_run,
        num_workers = num_workers,
        spatial_key = spatial_key,
        mem_gb = run_mem_gb,
        cpu_cores = run_cpu_cores,
        disk_gb = run_disk_gb,
        docker = unimodal_embeddings_docker
    }
  }

  output {
    File? scgpt_parquet = if (embedding_mode == "uni") then None else select_first(RunUnimodalEmbedding.scgpt_parquet)
    File? uni_parquet = if (embedding_mode == "scgpt") then None else select_first(RunUnimodalEmbedding.uni_parquet)
  }
}


task RunUnimodalEmbedding {
  input {
    File adata
    File wsi
    File uni_weights
    String mode
    Boolean input_is_log_normalized

    File scgpt_weights

    Int n_hvg
    Int scgpt_batch_size
    Int uni_batch_size
    String gene_col
    String layer_key
    Int seed
    Int max_seq_len
    Int input_bins
    String model_run
    Int num_workers
    String spatial_key

    Float mem_gb = 32
    Int cpu_cores = 2
    Int? disk_gb

    String docker
  }

  String scgpt_output_name = "scGPT.parquet"
  String uni_output_name = "UNI.parquet"
  Int default_disk_gb = ceil(size(adata, "GB") + size(wsi, "GB") + size(scgpt_weights, "GB") + size(uni_weights, "GB") + 20)

  command <<<
    set -eu -o pipefail

    if [ "~{mode}" = "scgpt" ]; then
      # The Python CLI expects a directory for scGPT weights, so unpack the archive first.
      mkdir -p scgpt_weights
      tar -xzf "~{scgpt_weights}" -C scgpt_weights --strip-components=1

      python /app/unimodal-embeddings.py \
        --mode scgpt \
        --adata "~{adata}" \
        --wsi "~{wsi}" \
        --output-dir "." \
        --scgpt-weights "scgpt_weights" \
        --n-hvg ~{n_hvg} \
        --scgpt-batch-size ~{scgpt_batch_size} \
        --gene-col "~{gene_col}" \
        --layer-key "~{layer_key}" \
        ~{if (input_is_log_normalized) then "--log-norm" else ""} \
        --seed ~{seed} \
        --max-seq-len ~{max_seq_len} \
        --input-bins ~{input_bins} \
        --model-run "~{model_run}" \
        --num-workers ~{num_workers} \
        --scgpt-output-name "~{scgpt_output_name}" \
        --overwrite
    fi

    if [ "~{mode}" = "uni" ]; then
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
    fi
  >>>

  output {
    File? scgpt_parquet = if (mode == "scgpt") then scgpt_output_name else None
    File? uni_parquet = if (mode == "uni") then uni_output_name else None
  }

  runtime {
    docker: docker
    memory: mem_gb + " GB"
    cpu: cpu_cores
    disks: "local-disk " + select_first([disk_gb, default_disk_gb]) + " HDD"
    bootDiskSizeGb: 20
    gpuType: "nvidia-tesla-t4"
    gpuCount: 1
    preemptible: 3
  }
}
