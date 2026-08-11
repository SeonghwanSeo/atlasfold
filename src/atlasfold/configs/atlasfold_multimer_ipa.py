from atlasfold.model import model_multimer_ipa as model

multimer_ipa_config = model.AtlasFoldMultimerIPAConfig(
    name="atlasfold-multimer-ipa",
    lm_name="atlaslm-3b",
    channel_s=384,
    channel_s_lm=768,
    channel_z=128,
    position_recycling=model.PositionRecyclingConfig(
        num_bins=15,
        min_bin=3.25,
        max_bin=20.75,
    ),
    trunk=model.TrunkConfig(
        dropout_z=0.25,
        num_heads=16,
        num_tri_heads=4,
        num_lm_blocks=4,
        num_blocks=48,
        num_pair_to_single_blocks=12,
    ),
    template_module=model.TemplateModuleConfig(
        channel_template=64,
        num_blocks=2,
        num_tri_heads=4,
        dropout_z=0.25,
        num_distogram_bins=39,
        min_dist=3.25,
        max_dist=50.75,
    ),
    structure_module=model.StructureModuleConfig(
        num_layer=8,
        num_head=12,
        num_scalar_qk=16,
        num_scalar_v=16,
        num_point_qk=4,
        num_point_v=8,
        num_layer_in_transition=3,
        dropout=0.1,
        sidechain_channel=128,
        sidechain_num_layer=2,
        num_torsion=7,
        position_scale=20.0,
    ),
    distogram_head=model.DistogramHeadConfig(
        num_bins=64,
        min_dist=2.0,
        max_dist=22.0,
    ),
    confidence_head=model.ConfidenceHeadConfig(
        hidden_channel=128,
        num_plddt_bins=50,
        num_pae_bins=64,
        max_pae_error=31.0,
    ),
)

# Match the conventional preset name used by the existing multimer config.
multimer_config = multimer_ipa_config
