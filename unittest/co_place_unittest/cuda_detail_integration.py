"""Check CUDA detailed placement with all placement extensions loaded.

Run with --runtime /path/to/install on a CUDA host. Loading only the ISM
extension misses interactions with the other registered CUDA kernel stubs.
"""
import argparse
import sys
from pathlib import Path


def run(runtime):
    runtime=Path(runtime).resolve()
    sys.path[:0]=[str(runtime),str(runtime/'dreamplace'),
                 str(runtime/'thirdparty/DREAMPlace'),
                 str(runtime/'thirdparty/DREAMPlace/dreamplace')]
    import torch
    import dreamplace.BasicPlace  # Load the same operators as production.
    from dreamplace.ops.independent_set_matching import independent_set_matching
    from dreamplace.ops.global_swap import global_swap
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    torch.set_num_threads(1)
    f=lambda v:torch.tensor(v,dtype=torch.float32,device='cuda')
    i=lambda v:torch.tensor(v,dtype=torch.int32,device='cuda')
    # Four legal cells in two rows, with two horizontal two-pin nets.
    pos=f([0,2,6,8, 0,1,0,1])
    common=dict(node_size_x=f([2]*4),node_size_y=f([1]*4),
        flat_region_boxes=f([]),flat_region_boxes_start=i([0]),
        node2fence_region_map=i([-1]*4),flat_net2pin_map=i([0,2,1,3]),
        flat_net2pin_start_map=i([0,2,4]),pin2net_map=i([0,1,0,1]),
        flat_node2pin_map=i([0,1,2,3]),flat_node2pin_start_map=i([0,1,2,3,4]),
        pin2node_map=i([0,1,2,3]),pin_offset_x=f([1]*4),pin_offset_y=f([.5]*4),
        net_mask=torch.ones(2,dtype=torch.uint8,device='cuda'),
        xl=0.,yl=0.,xh=10.,yh=2.,site_width=1.,row_height=1.,
        num_bins_x=2,num_bins_y=2,num_movable_nodes=4,num_terminal_NIs=0,
        num_filler_nodes=0,batch_size=4,max_iters=2)
    op=independent_set_matching.IndependentSetMatching(set_size=4,**common)
    result=op(pos)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(result).all())
    swap=global_swap.GlobalSwap(**common)
    result=swap(result)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(result).all())
    print('CUDA_DETAIL_INTEGRATION_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',required=True)
    run(parser.parse_args().runtime)
