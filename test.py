from atlasfold.pretrained import get_runner, load_model

# Monomer prediction
monomer_model = load_model("atlasfold", device="cuda")
folding_runner = get_runner(monomer_model)
seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
monomer_out = folding_runner.fold("test", seq, num_samples=5)
print(monomer_out.best.avg_plddt)
with open("test_monomer.pdb", "w") as f:
    f.write(monomer_out.best.to_pdb())

# Multimer prediction
multimer_model = load_model("atlasfold-m", device="cuda")
multimer_runner = get_runner(multimer_model)
seq1 = (
    "GSEVQLLESGGGLVQAGDSLRLSCAASGRTFSAYAMGWFRQAPGKEREFVAAISWSGNSTYYAD"
    "SVKGRFTISRDNAKNTVYLQMNSLKPEDTAIYYCAARKPMYRVDISKGQNYDYWGQGTQVTVSS"
)
seq2 = "GAMGPGVDTQIFEDPREFLSHLEEYLRQVGGSEEYWLSQIQNHMNGPAKKWWEFKQGSVKNWVEFKKEFLQYSEG"
multimer_out = multimer_runner.fold("test_m", [seq1, seq2], seeds=[1, 2], num_samples=5)
print(multimer_out.best.iptm)
with open("test_multimer.cif", "w") as f:
    f.write(multimer_out.best.to_mmcif())
