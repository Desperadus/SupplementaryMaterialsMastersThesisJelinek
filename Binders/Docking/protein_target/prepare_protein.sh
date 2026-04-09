pdb2pqr \
  --ff=AMBER \
  --with-ph=7.5 \
  OBP5.pdb OBP5.pqr

mk_prepare_receptor.py --read_pqr OBP5.pqr -p --output_basename prepared
