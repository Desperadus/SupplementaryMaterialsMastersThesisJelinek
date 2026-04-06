set top_file "topology.pdb"
set traj_file "samples.xtc"
set resid1 33
set resid2 37
set atom_name "CA"
set out_file "ca_distance_res33_res37.dat"

mol new $top_file type pdb waitfor all
mol addfile $traj_file type xtc waitfor all

set sel1 [atomselect top "resid $resid1 and name $atom_name"]
set sel2 [atomselect top "resid $resid2 and name $atom_name"]

set outfile [open $out_file "w"]
puts $outfile "# frame distance_angstrom"

set nframes [molinfo top get numframes]
for {set frame 0} {$frame < $nframes} {incr frame} {
    $sel1 frame $frame
    $sel2 frame $frame
    set xyz1 [lindex [$sel1 get {x y z}] 0]
    set xyz2 [lindex [$sel2 get {x y z}] 0]
    set dist [veclength [vecsub $xyz1 $xyz2]]
    puts $outfile "$frame $dist"
}

close $outfile
quit
