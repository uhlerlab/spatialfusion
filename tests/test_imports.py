def test_import_spatialfusion():
    import spatialfusion
    import spatialfusion.models.multi_ae
    import spatialfusion.models.gcn
    import spatialfusion.models.baseline_multi_ae
    import spatialfusion.utils.embed_ae_utils
    import spatialfusion.utils.gcn_utils
    import spatialfusion.utils.embed_gcn_utils
    import spatialfusion.utils.ae_data_loader
    import spatialfusion.utils.baseline_ae_data_loader
    import spatialfusion.utils.pkg_ckpt
    assert True  # If imports succeed, test passes
