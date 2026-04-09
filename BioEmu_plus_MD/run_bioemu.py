from bioemu.sample import main as sample
from Bio import SeqIO
import logging

logging.basicConfig(level=logging.DEBUG)  

for record in SeqIO.parse("musM_OBP5_mature.fasta", "fasta"):
    print(f"Loading {record.id}")
    print(f"Sequence: {record.seq}")
    print(f"Length: {len(record)}")
    SEQUENCE = str(record.seq)

sample(sequence=SEQUENCE, num_samples=220, output_dir='bioemu_output')
