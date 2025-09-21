############################################################################################
### Scripts for placement-aware partitioning
### DATE: 2023-05-22
############################################################################################

### ----------------------------------------------------------------------------------------
### Set global variable here
set case_name $::env(TRITONPART_CASE_NAME)
set run_tmp_dir $::env(PLACER_RUNTMP_DIR)

puts "using case_name: $case_name"

set num_parts 2
set balance_constraint 2
set seed 0
set hypergraph_file "${run_tmp_dir}/${case_name}.hgr"
set placement_file "${run_tmp_dir}/${case_name}.flattened-2d.embedding.dat"
set solution_file "${run_tmp_dir}/${case_name}.hgr.part.${num_parts}"

### ----------------------------------------------------------------------------------------
### TritonPart with placement information
### Here we recommend using a small placement_wt_factors
puts "Start TritonPart for hypergraph partitioning with placement"
triton_part_hypergraph -hypergraph_file $hypergraph_file -num_parts $num_parts \
  -balance_constraint $balance_constraint \
  -seed $seed \
  -placement_file ${placement_file} -placement_wt_factors { 0.0000 0.0000 } \
  -placement_dimension 2

exit
### Finish