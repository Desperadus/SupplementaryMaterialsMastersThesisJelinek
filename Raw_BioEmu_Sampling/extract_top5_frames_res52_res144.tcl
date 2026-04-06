set top_file "topology.pdb"
set traj_file "samples.xtc"
set out_dir "top5_longest_res52_res144_frames"
set frame_list {3970 4487 771 4257 2554}

file mkdir $out_dir

mol new $top_file type pdb waitfor all
mol addfile $traj_file type xtc waitfor all

set all_atoms [atomselect top "all"]

foreach frame $frame_list {
    $all_atoms frame $frame
    $all_atoms update
    set out_file [format "%s/frame_%04d.pdb" $out_dir $frame]
    $all_atoms writepdb $out_file
}

quit
