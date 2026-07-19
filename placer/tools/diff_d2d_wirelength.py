'''
Author: D2D-placer
Description: Differentiable d2d wirelength + co-placement helper.

Implements coplace (run_gp.cpp) style 2.5D global placement that
simultaneously optimizes top tier cells, bottom tier cells, and D2D
terminal_NIs under a single Nesterov optimizer with three independent
ElectricPotential layers and a per-placedb weighted-average wirelength
loss.

Layout of mov_node_pos_all (1D tensor):
    [ x_top_mov | x_bot_mov | x_term_mov |
      x_top_fil | x_bot_fil | x_term_fil |
      y_top_mov | y_bot_mov | y_term_mov |
      y_top_fil | y_bot_fil | y_term_fil ]

Each layer's gradient is produced by the dreamplace ``PlaceObj``'s
``obj_and_grad_fn``, which already integrates wirelength + density
penalty + diagonal preconditioner. The three per-placedb gradients are
then scattered back to ``mov_node_pos_all`` so that a single Nesterov
optimizer steps all movable cells / terminals / fillers at once.

Fillers remain part of the joint optimizer state.  Leaving them static would
skew the per-placedb density landscape; memory reduction is instead obtained
by compacting net topology and shortening tensor lifetimes.

The same ``mov_node_pos_all`` is shared by three placedbs:
  * dp_tier[0]: x/y_top_mov drives movable cells; x/y_term_mov drives
    terminal_NIs (fixed in dp_tier[0] view); x/y_top_fil drives fillers.
  * dp_tier[1]: same with bot.
  * dp_terminal: x/y_term_mov drives movable cells (the terminal_NIs
    themselves); x/y_term_fil drives fillers.
'''

import logging
import os
import numpy as np
import torch

import dreamplace.PlaceObj as PlaceObj
import dreamplace.EvalMetrics as EvalMetrics
import dreamplace.ops.hpwl.hpwl as hpwl

logger = logging.getLogger(__name__)


def _compact_net_arrays(placedb, ignore_net_degree):
    """Build a dense pin topology containing only valid low-fanout nets.

    DREAMPlace normally retains every pin and applies a net mask inside the
    wirelength kernel.  That avoids the arithmetic for high-fanout nets, but
    it does not avoid copying/storing their topology or constructing their pin
    positions.  Co-placement owns short-lived placedbs, so it can safely use a
    compact topology for the duration of the joint global-placement stage.

    The returned pin ids are renumbered to ``[0, num_kept_pins)``.  Pin offsets
    and pin-to-node maps are gathered in that same order.
    """
    starts64 = np.asarray(placedb.flat_net2pin_start_map, dtype=np.int64)
    degrees64 = starts64[1:] - starts64[:-1]
    keep_nets = np.logical_and(degrees64 >= 2,
                               degrees64 < int(ignore_net_degree))
    kept_net_ids = np.flatnonzero(keep_nets)
    kept_degrees64 = degrees64[keep_nets]

    # flat_net2pin_map is ordered by net.  Expanding only a boolean mask keeps
    # the temporary construction on host memory and avoids a second full GPU
    # topology while the compact maps are being assembled.
    keep_pin_slots = np.repeat(keep_nets, degrees64)
    original_flat_net2pin = np.asarray(placedb.flat_net2pin_map)
    selected_pin_ids = original_flat_net2pin[keep_pin_slots].astype(
        np.int64, copy=False)

    num_kept_pins = int(selected_pin_ids.size)
    num_kept_nets = int(kept_net_ids.size)
    compact_starts = np.empty(num_kept_nets + 1, dtype=np.int32)
    compact_starts[0] = 0
    if num_kept_nets:
        np.cumsum(kept_degrees64, dtype=np.int64,
                  out=compact_starts[1:])

    compact_flat_net2pin = np.arange(num_kept_pins, dtype=np.int32)
    compact_pin2net = np.repeat(
        np.arange(num_kept_nets, dtype=np.int32), kept_degrees64)
    compact_pin2node = np.asarray(
        placedb.pin2node_map)[selected_pin_ids].astype(np.int32, copy=False)
    compact_net_weights = np.asarray(placedb.net_weights)[kept_net_ids]
    compact_pin_offset_x = np.asarray(
        placedb.pin_offset_x)[selected_pin_ids]
    compact_pin_offset_y = np.asarray(
        placedb.pin_offset_y)[selected_pin_ids]

    num_nodes = int(placedb.num_nodes)
    pin_counts = np.bincount(compact_pin2node,
                             minlength=num_nodes).astype(np.float32,
                                                        copy=False)
    pin_net_weights = compact_net_weights[compact_pin2net]
    node_pin_weights = np.bincount(
        compact_pin2node,
        weights=pin_net_weights,
        minlength=num_nodes).astype(compact_net_weights.dtype, copy=False)

    return {
        "flat_net2pin_map": compact_flat_net2pin,
        "flat_net2pin_start_map": compact_starts,
        "pin2net_map": compact_pin2net,
        "pin2node_map": compact_pin2node,
        "pin_offset_x": compact_pin_offset_x,
        "pin_offset_y": compact_pin_offset_y,
        "net_weights": compact_net_weights,
        "pin_counts": pin_counts,
        "node_pin_weights": node_pin_weights,
        "original_num_nets": int(degrees64.size),
        "original_num_pins": int(original_flat_net2pin.size),
        "kept_num_nets": num_kept_nets,
        "kept_num_pins": num_kept_pins,
    }


class _CompactPinPos(object):
    """Autograd-friendly pin-position gather for a compact pin topology."""

    def __init__(self, pin2node_map, pin_offset_x, pin_offset_y):
        # index_select requires int64 indices.  Keeping the converted map here
        # avoids converting a multi-million-entry tensor on every objective.
        self.pin2node_map = pin2node_map.long()
        self.pin_offset_x = pin_offset_x
        self.pin_offset_y = pin_offset_y

    def __call__(self, pos):
        num_nodes = pos.numel() // 2
        pin_x = torch.index_select(pos[:num_nodes], 0,
                                   self.pin2node_map).add(
                                       self.pin_offset_x)
        pin_y = torch.index_select(pos[num_nodes:], 0,
                                   self.pin2node_map).add(
                                       self.pin_offset_y)
        return torch.cat((pin_x, pin_y))


class _CachedPinWeightSum(object):
    """Return the static low-fanout per-node net-weight sum."""

    def __init__(self, node_pin_weights):
        self.node_pin_weights = node_pin_weights

    def __call__(self, _net_weights):
        return self.node_pin_weights


class D2DCoPlace(object):
    """Co-placement helper for two-tier std cells + D2D terminals.

    Wraps three dreamplace ``PlaceObj`` instances (top tier, bot tier,
    terminal) and exposes a single ``obj_and_grad_fn`` operating on
    ``mov_node_pos_all`` (which contains every layer's movable + filler
    nodes plus the shared D2D terminal_NIs) so that a single Nesterov
    optimizer drives all three placedbs simultaneously.
    """

    def __init__(self, d2d_placer):
        self.placer = d2d_placer
        self.params = d2d_placer.params
        self.dreamplace = d2d_placer.dreamplace
        self.num_terminal_NIs = d2d_placer.num_terminal_NIs

        self.dp_top = self.dreamplace.dp_tier[0]
        self.dp_bot = self.dreamplace.dp_tier[1]
        self.dp_term = self.dreamplace.dp_terminal

        # ----- per-placedb sizes -----
        # n_*_mov:    number of movable cells (cells in tier; terminal_NIs in dp_term)
        # n_*_phys:   number of physical nodes (movable + fixed terminals + terminal_NIs)
        # n_*_total:  number of nodes (physical + fillers); pos length is 2*total
        # n_*_fil:    number of fillers in this placedb
        self.n_top_mov = self.dp_top.placedb.num_movable_nodes
        self.n_bot_mov = self.dp_bot.placedb.num_movable_nodes
        self.n_term_mov = self.dp_term.placedb.num_movable_nodes
        self.n_top_phys = self.dp_top.placedb.num_physical_nodes
        self.n_bot_phys = self.dp_bot.placedb.num_physical_nodes
        self.n_term_phys = self.dp_term.placedb.num_physical_nodes
        self.n_top_total = self.dp_top.placedb.num_nodes
        self.n_bot_total = self.dp_bot.placedb.num_nodes
        self.n_term_total = self.dp_term.placedb.num_nodes
        self.n_top_fil = self.n_top_total - self.n_top_phys
        self.n_bot_fil = self.n_bot_total - self.n_bot_phys
        self.n_term_fil = self.n_term_total - self.n_term_phys

        # Range of D2D terminal_NIs inside each tier placedb pos:
        #   [num_movable + num_terminals, num_movable + num_terminals + num_terminal_NIs)
        # In typical D2D-placer setups num_terminals == 0 so terminal_NIs
        # immediately follow movable cells.
        self.top_term_lo = (self.dp_top.placedb.num_movable_nodes
                            + self.dp_top.placedb.num_terminals)
        self.top_term_hi = (self.top_term_lo
                            + self.dp_top.placedb.num_terminal_NIs)
        self.bot_term_lo = (self.dp_bot.placedb.num_movable_nodes
                            + self.dp_bot.placedb.num_terminals)
        self.bot_term_hi = (self.bot_term_lo
                            + self.dp_bot.placedb.num_terminal_NIs)
        self.n_term = self.n_term_mov

        if self.dp_top.placedb.num_terminal_NIs != self.n_term:
            logger.warning(
                "dp_tier[0].num_terminal_NIs (%d) != dp_terminal.num_movable (%d); "
                "co-place will only sync the leading min() entries.",
                self.dp_top.placedb.num_terminal_NIs, self.n_term)
        if self.dp_bot.placedb.num_terminal_NIs != self.n_term:
            logger.warning(
                "dp_tier[1].num_terminal_NIs (%d) != dp_terminal.num_movable (%d); "
                "co-place will only sync the leading min() entries.",
                self.dp_bot.placedb.num_terminal_NIs, self.n_term)

        # cached per-tier params
        self.params_top = self.params.partition_tier[0]
        self.params_bot = self.params.partition_tier[1]
        self.params_term = self.params.terminal

        # x_term_mov / y_term_mov store dp_terminal outer-box left-bottom.
        # tier terminal_NI uses terminalSize box; same bonding center requires
        # tier pos = dp_terminal pos + terminalSpacing / 2 (per partition_aux).
        self.term_shift = float(d2d_placer.die_spec.terminalSpacing) * 0.5

        # ----- precompute mov_node_pos_all slice offsets -----
        # x part layout (then mirrored for y):
        #   [0,                   nt)    -> x_top_mov
        #   [nt,                  nt+nb) -> x_bot_mov
        #   [nt+nb,               nt+nb+ne)        -> x_term_mov
        #   [nt+nb+ne,            +ftop)           -> x_top_fil
        #   [...,                 +fbot)           -> x_bot_fil
        #   [...,                 +fterm)          -> x_term_fil
        nt = self.n_top_mov
        nb = self.n_bot_mov
        ne = self.n_term
        ft = self.n_top_fil
        fb = self.n_bot_fil
        fe = self.n_term_fil
        self._half = nt + nb + ne + ft + fb + fe
        self._off_top_mov = 0
        self._off_bot_mov = nt
        self._off_term_mov = nt + nb
        self._off_top_fil = nt + nb + ne
        self._off_bot_fil = nt + nb + ne + ft
        self._off_term_fil = nt + nb + ne + ft + fb

        ignore_net_degree = int(
            getattr(self.params, "co_place_ignore_net_degree",
                    getattr(self.params_top, "ignore_net_degree", 100)))
        self._install_compact_topology(self.dp_top, "top",
                                       ignore_net_degree)
        self._install_compact_topology(self.dp_bot, "bot",
                                       ignore_net_degree)
        self._install_compact_topology(self.dp_term, "term",
                                       ignore_net_degree)

        self._build_placeobjs()

        # eval_ops dicts consumed by EvalMetrics.evaluate
        self._eval_ops_top = {
            "hpwl": self.dp_top.basic_place.op_collections.hpwl_op,
            "overflow":
            self.dp_top.basic_place.op_collections.density_overflow_op,
        }
        self._eval_ops_bot = {
            "hpwl": self.dp_bot.basic_place.op_collections.hpwl_op,
            "overflow":
            self.dp_bot.basic_place.op_collections.density_overflow_op,
        }
        self._eval_ops_term = {
            "hpwl": self.dp_term.basic_place.op_collections.hpwl_op,
            "overflow":
            self.dp_term.basic_place.op_collections.density_overflow_op,
        }

        logger.info(
            "co-place variables: movable(top/bot/term)=%d/%d/%d, "
            "fillers=%d/%d/%d, joint=%d coordinates", self.n_top_mov,
            self.n_bot_mov, self.n_term_mov, self.n_top_fil, self.n_bot_fil,
            self.n_term_fil, self._half * 2)
        self.log_memory("constructed")

    def _install_compact_topology(self, dp, name, ignore_net_degree):
        """Replace a co-place layer's GPU net data with low-fanout topology."""
        placedb = dp.placedb
        data = dp.basic_place.data_collections
        ops = dp.basic_place.op_collections
        device = data.pos[0].device
        compact = _compact_net_arrays(placedb, ignore_net_degree)

        def tensor(array, dtype=None):
            value = torch.from_numpy(np.ascontiguousarray(array)).to(device)
            return value.to(dtype=dtype) if dtype is not None else value

        data.flat_net2pin_map = tensor(compact["flat_net2pin_map"])
        data.flat_net2pin_start_map = tensor(
            compact["flat_net2pin_start_map"])
        data.pin2net_map = tensor(compact["pin2net_map"])
        data.pin2node_map = tensor(compact["pin2node_map"])
        data.pin_offset_x = tensor(compact["pin_offset_x"],
                                   data.pos[0].dtype)
        data.pin_offset_y = tensor(compact["pin_offset_y"],
                                   data.pos[0].dtype)
        data.net_weights = tensor(compact["net_weights"], data.pos[0].dtype)
        data.net_mask_all = torch.ones(compact["kept_num_nets"],
                                       dtype=torch.uint8,
                                       device=device)
        data.net_mask_ignore_large_degrees = data.net_mask_all
        data.pin_mask_ignore_fixed_macros = (
            data.pin2node_map >= placedb.num_movable_nodes)
        data.pin_weights = tensor(compact["pin_counts"], data.pos[0].dtype)
        data.num_pins_in_nodes = data.pin_weights

        compact_pin_pos = _CompactPinPos(data.pin2node_map,
                                         data.pin_offset_x,
                                         data.pin_offset_y)
        compact_hpwl = hpwl.HPWL(
            flat_netpin=data.flat_net2pin_map,
            netpin_start=data.flat_net2pin_start_map,
            pin2net_map=data.pin2net_map,
            net_weights=data.net_weights,
            net_mask=data.net_mask_all,
            algorithm="net-by-net")

        def compact_hpwl_op(pos):
            return compact_hpwl(compact_pin_pos(pos))

        ops.pin_pos_op = compact_pin_pos
        ops.hpwl_op = compact_hpwl_op
        ops.pws_op = _CachedPinWeightSum(
            tensor(compact["node_pin_weights"], data.pos[0].dtype))

        # These BasicPlace operators are not used during co-placement and keep
        # references to the original full topology.  The next die-by-die stage
        # rebuilds its BasicPlace instance, so releasing them here is safe.
        for attr in ("legality_check_op", "legalize_op",
                     "individual_legalize_op", "macro_legalize_op",
                     "detailed_place_op", "timing_op", "gift_init_op",
                     "route_utilization_map_op", "pin_utilization_map_op",
                     "nctugr_congestion_map_op", "adjust_node_area_op"):
            if hasattr(ops, attr):
                setattr(ops, attr, None)

        # pin_pos and preconditioning no longer need node-to-pin adjacency.
        data.flat_node2pin_map = torch.empty(0,
                                             dtype=torch.int32,
                                             device=device)
        data.flat_node2pin_start_map = torch.empty(0,
                                                   dtype=torch.int32,
                                                   device=device)

        removed_nets = (compact["original_num_nets"]
                        - compact["kept_num_nets"])
        removed_pins = (compact["original_num_pins"]
                        - compact["kept_num_pins"])
        logger.info(
            "co-place %s compact topology: nets %d -> %d (removed %d), "
            "pins %d -> %d (removed %d), ignore_net_degree=%d", name,
            compact["original_num_nets"], compact["kept_num_nets"],
            removed_nets, compact["original_num_pins"],
            compact["kept_num_pins"], removed_pins, ignore_net_degree)

    @staticmethod
    def log_memory(stage):
        """Log current process RSS and CUDA allocator state."""
        rss_gib = float("nan")
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as stream:
                for line in stream:
                    if line.startswith("VmRSS:"):
                        rss_gib = int(line.split()[1]) / (1024.0 * 1024.0)
                        break
        except OSError:
            pass
        if torch.cuda.is_available():
            logger.info(
                "co-place memory [%s]: RSS %.2f GiB, CUDA alloc/reserved/peak "
                "%.2f/%.2f/%.2f GiB", stage, rss_gib,
                torch.cuda.memory_allocated() / 2**30,
                torch.cuda.memory_reserved() / 2**30,
                torch.cuda.max_memory_allocated() / 2**30)
        else:
            logger.info("co-place memory [%s]: RSS %.2f GiB", stage, rss_gib)

    # ------------------------------------------------------------------
    # PlaceObj construction
    # ------------------------------------------------------------------
    def _build_placeobjs(self):
        """Construct one ``PlaceObj`` per placedb.

        Each ``PlaceObj.__init__`` rebuilds wirelength_op / density_op /
        density_overflow_op / update_gamma_op / update_density_weight_op /
        precondition_op on its placedb's ``op_collections``. After this
        call, the three placedbs each carry a self-consistent set of
        ePlace operators with dynamic gamma + adaptive density weight.
        """
        gp_top = self.params_top.global_place_stages[0]
        gp_bot = self.params_bot.global_place_stages[0]
        gp_term = self.params_term.global_place_stages[0]

        device = self.dp_top.basic_place.data_collections.pos[0].device

        self.model_top = PlaceObj.PlaceObj(
            0.0, self.params_top, self.dp_top.placedb,
            self.dp_top.basic_place.data_collections,
            self.dp_top.basic_place.op_collections, gp_top).to(device)
        self.model_top.train()

        self.model_bot = PlaceObj.PlaceObj(
            0.0, self.params_bot, self.dp_bot.placedb,
            self.dp_bot.basic_place.data_collections,
            self.dp_bot.basic_place.op_collections, gp_bot).to(device)
        self.model_bot.train()

        self.model_term = PlaceObj.PlaceObj(
            0.0, self.params_term, self.dp_term.placedb,
            self.dp_term.basic_place.data_collections,
            self.dp_term.basic_place.op_collections, gp_term).to(device)
        self.model_term.train()

    # ------------------------------------------------------------------
    # Position packing / un-packing
    # ------------------------------------------------------------------
    def pack_initial(self):
        """Build the initial mov_node_pos_all tensor from current placedb pos.

        Includes movable cells / terminal_NIs *and* fillers from each
        placedb so that all of them participate in the joint Nesterov
        optimization.
        """
        with torch.no_grad():
            top_pos = self.dp_top.basic_place.data_collections.pos[0]
            bot_pos = self.dp_bot.basic_place.data_collections.pos[0]
            term_pos = self.dp_term.basic_place.data_collections.pos[0]
            mov = top_pos.new_empty(self._half * 2)
            H = self._half

            mov[self._off_top_mov:self._off_top_mov + self.n_top_mov].copy_(
                top_pos[:self.n_top_mov])
            mov[self._off_bot_mov:self._off_bot_mov + self.n_bot_mov].copy_(
                bot_pos[:self.n_bot_mov])
            mov[self._off_term_mov:self._off_term_mov + self.n_term_mov].copy_(
                term_pos[:self.n_term_mov])
            mov[self._off_top_fil:self._off_top_fil + self.n_top_fil].copy_(
                top_pos[self.n_top_phys:self.n_top_total])
            mov[self._off_bot_fil:self._off_bot_fil + self.n_bot_fil].copy_(
                bot_pos[self.n_bot_phys:self.n_bot_total])
            mov[self._off_term_fil:self._off_term_fil + self.n_term_fil].copy_(
                term_pos[self.n_term_phys:self.n_term_total])

            mov[H + self._off_top_mov:H + self._off_top_mov
                + self.n_top_mov].copy_(
                    top_pos[self.n_top_total:self.n_top_total
                            + self.n_top_mov])
            mov[H + self._off_bot_mov:H + self._off_bot_mov
                + self.n_bot_mov].copy_(
                    bot_pos[self.n_bot_total:self.n_bot_total
                            + self.n_bot_mov])
            mov[H + self._off_term_mov:H + self._off_term_mov
                + self.n_term_mov].copy_(
                    term_pos[self.n_term_total:self.n_term_total
                             + self.n_term_mov])
            mov[H + self._off_top_fil:H + self._off_top_fil
                + self.n_top_fil].copy_(
                    top_pos[self.n_top_total
                            + self.n_top_phys:self.n_top_total * 2])
            mov[H + self._off_bot_fil:H + self._off_bot_fil
                + self.n_bot_fil].copy_(
                    bot_pos[self.n_bot_total
                            + self.n_bot_phys:self.n_bot_total * 2])
            mov[H + self._off_term_fil:H + self._off_term_fil
                + self.n_term_fil].copy_(
                    term_pos[self.n_term_total
                             + self.n_term_phys:self.n_term_total * 2])

        mov.requires_grad_(True)
        return mov

    def _split(self, mov_node_pos_all):
        """Split mov_node_pos_all into per-layer (mov, filler) views,
        for both x and y. Returns 12 tensors in fixed order:
            (x_top_mov, x_bot_mov, x_term_mov,
             x_top_fil, x_bot_fil, x_term_fil,
             y_top_mov, y_bot_mov, y_term_mov,
             y_top_fil, y_bot_fil, y_term_fil)
        """
        nt = self.n_top_mov
        nb = self.n_bot_mov
        ne = self.n_term
        ft = self.n_top_fil
        fb = self.n_bot_fil
        fe = self.n_term_fil
        H = self._half
        x = mov_node_pos_all[:H]
        y = mov_node_pos_all[H:]
        x_top_mov = x[:nt]
        x_bot_mov = x[nt:nt + nb]
        x_term_mov = x[nt + nb:nt + nb + ne]
        x_top_fil = x[nt + nb + ne:nt + nb + ne + ft]
        x_bot_fil = x[nt + nb + ne + ft:nt + nb + ne + ft + fb]
        x_term_fil = x[nt + nb + ne + ft + fb:nt + nb + ne + ft + fb + fe]
        y_top_mov = y[:nt]
        y_bot_mov = y[nt:nt + nb]
        y_term_mov = y[nt + nb:nt + nb + ne]
        y_top_fil = y[nt + nb + ne:nt + nb + ne + ft]
        y_bot_fil = y[nt + nb + ne + ft:nt + nb + ne + ft + fb]
        y_term_fil = y[nt + nb + ne + ft + fb:nt + nb + ne + ft + fb + fe]
        return (x_top_mov, x_bot_mov, x_term_mov, x_top_fil, x_bot_fil,
                x_term_fil, y_top_mov, y_bot_mov, y_term_mov, y_top_fil,
                y_bot_fil, y_term_fil)

    def _build_tier_pos(self, base, n_phys, n_total, n_mov, term_lo, term_hi,
                        x_mov, y_mov, x_term, y_term, x_fil, y_fil):
        """Assemble a tier placedb's pos tensor of length 2*n_total.

        Layout per coordinate axis:
            [movable cells]              len = n_mov
            [tier-internal fixed pieces] len = term_lo - n_mov   (e.g. fixed terminals; usually 0)
            [D2D terminal_NIs]           len = (term_hi - term_lo);
                values = x_term + terminalSpacing/2 (tier NI left-bottom)
            [tier-internal trailing fixed] len = n_phys - term_hi (usually 0)
            [fillers]                    len = n_total - n_phys
        """
        base = base.detach()
        ne_in_tier = term_hi - term_lo
        ne_used = min(ne_in_tier, x_term.shape[0])
        shift = self.term_shift

        pos = torch.empty_like(base)
        pos.copy_(base)
        pos[:n_mov].copy_(x_mov)
        pos[n_total:n_total + n_mov].copy_(y_mov)
        if ne_used > 0:
            pos[term_lo:term_lo + ne_used].copy_(x_term[:ne_used] + shift)
            pos[n_total + term_lo:n_total + term_lo + ne_used].copy_(
                y_term[:ne_used] + shift)
        pos[n_phys:n_total].copy_(x_fil)
        pos[n_total + n_phys:2 * n_total].copy_(y_fil)
        return pos

    def _build_term_pos(self, x_term_mov, y_term_mov, x_term_fil, y_term_fil):
        """Assemble dp_terminal's pos tensor with movable terminal_NIs +
        fillers; trailing fixed nodes (if any) come from current base pos.
        """
        base = self.dp_term.basic_place.data_collections.pos[0].detach()
        n_mov = self.n_term_mov
        n_phys = self.n_term_phys
        n_total = self.n_term_total

        pos = torch.empty_like(base)
        pos.copy_(base)
        pos[:n_mov].copy_(x_term_mov)
        pos[n_total:n_total + n_mov].copy_(y_term_mov)
        pos[n_phys:n_total].copy_(x_term_fil)
        pos[n_total + n_phys:2 * n_total].copy_(y_term_fil)
        return pos

    def _build_pos_top(self, x_top_mov, y_top_mov, x_term_mov, y_term_mov,
                       x_top_fil, y_top_fil):
        return self._build_tier_pos(
            self.dp_top.basic_place.data_collections.pos[0], self.n_top_phys,
            self.n_top_total, self.n_top_mov, self.top_term_lo,
            self.top_term_hi, x_top_mov, y_top_mov, x_term_mov, y_term_mov,
            x_top_fil, y_top_fil)

    def _build_pos_bot(self, x_bot_mov, y_bot_mov, x_term_mov, y_term_mov,
                       x_bot_fil, y_bot_fil):
        return self._build_tier_pos(
            self.dp_bot.basic_place.data_collections.pos[0], self.n_bot_phys,
            self.n_bot_total, self.n_bot_mov, self.bot_term_lo,
            self.bot_term_hi, x_bot_mov, y_bot_mov, x_term_mov, y_term_mov,
            x_bot_fil, y_bot_fil)

    def _build_pos_term(self, x_term_mov, y_term_mov, x_term_fil, y_term_fil):
        return self._build_term_pos(x_term_mov, y_term_mov, x_term_fil,
                                    y_term_fil)

    def _build_all_pos(self, mov_node_pos_all):
        """Return (pos_top, pos_bot, pos_term) for the current mov_node_pos_all."""
        (x_tm, x_bm, x_em, x_tf, x_bf, x_ef,
         y_tm, y_bm, y_em, y_tf, y_bf, y_ef) = self._split(mov_node_pos_all)
        pos_top = self._build_pos_top(x_tm, y_tm, x_em, y_em, x_tf, y_tf)
        pos_bot = self._build_pos_bot(x_bm, y_bm, x_em, y_em, x_bf, y_bf)
        pos_term = self._build_pos_term(x_em, y_em, x_ef, y_ef)
        return pos_top, pos_bot, pos_term

    def _sync_placedb_pos(self, mov_node_pos_all):
        """Mirror current mov_node_pos_all into each placedb's data_collections.pos[0].

        ``PlaceObj.initialize_density_weight`` reads the placedb's
        ``data_collections.pos[0]`` to estimate
        ``|grad_wl|_1 / |grad_density|_1``; making sure that vector
        carries the latest movable + terminal_NI + filler snapshot keeps
        the estimate consistent.
        """
        with torch.no_grad():
            (x_tm, x_bm, x_em, x_tf, x_bf, x_ef,
             y_tm, y_bm, y_em, y_tf, y_bf, y_ef) = self._split(
                 mov_node_pos_all)
            pos_top = self._build_pos_top(x_tm, y_tm, x_em, y_em, x_tf,
                                          y_tf)
            self.dp_top.basic_place.data_collections.pos[0].data.copy_(pos_top)
            del pos_top
            pos_bot = self._build_pos_bot(x_bm, y_bm, x_em, y_em, x_bf,
                                          y_bf)
            self.dp_bot.basic_place.data_collections.pos[0].data.copy_(pos_bot)
            del pos_bot
            pos_term = self._build_pos_term(x_em, y_em, x_ef, y_ef)
            self.dp_term.basic_place.data_collections.pos[0].data.copy_(
                pos_term)

    # ------------------------------------------------------------------
    # PlaceObj initialization (density_weight + gamma)
    # ------------------------------------------------------------------
    def initialize_models(self, mov_node_pos_all):
        """Initialize per-layer density_weight via PlaceObj's own routine."""
        self._sync_placedb_pos(mov_node_pos_all)
        for model, params, placedb, name in (
            (self.model_top, self.params_top, self.dp_top.placedb, "top"),
            (self.model_bot, self.params_bot, self.dp_bot.placedb, "bot"),
            (self.model_term, self.params_term, self.dp_term.placedb, "term"),
        ):
            try:
                model.initialize_density_weight(params, placedb)
                logger.info(
                    "co-place init density_weight (%s) = %.6E", name,
                    model.density_weight.data.mean().item())
            except Exception as e:
                logger.warning(
                    "co-place init density_weight (%s) failed: %s; "
                    "fall back to params.density_weight=%.3E", name, e,
                    params.density_weight)
                model.density_weight.data.fill_(params.density_weight)
        self.log_memory("density initialized")

    # ------------------------------------------------------------------
    # Objective + gradient
    # ------------------------------------------------------------------
    def make_obj_and_grad_fn(self):
        """Build obj_and_grad_fn(p) used by NesterovAcceleratedGradientOptimizer.

        Internally we build three leaf tensors ``pos_top/pos_bot/pos_term``
        from p (each carrying movable cells + terminal_NIs + fillers) and
        delegate per-layer wirelength + density + precondition to the
        corresponding ``PlaceObj.obj_and_grad_fn``. The three preconditioned
        gradients (which also include filler grads) are scattered back to
        ``p.grad`` so a single Nesterov step updates everything at once.
        """
        nt = self.n_top_mov
        nb = self.n_bot_mov
        ne = self.n_term
        ft = self.n_top_fil
        fb = self.n_bot_fil
        fe = self.n_term_fil
        nt_total = self.n_top_total
        nb_total = self.n_bot_total
        ne_total = self.n_term_total
        nt_phys = self.n_top_phys
        nb_phys = self.n_bot_phys
        ne_phys = self.n_term_phys
        H = self._half

        ne_top_used = min(self.top_term_hi - self.top_term_lo, ne)
        ne_bot_used = min(self.bot_term_hi - self.bot_term_lo, ne)

        def scatter_top_grad(p, g_top):
            grad = p.grad
            grad[self._off_top_mov:self._off_top_mov + nt].add_(g_top[:nt])
            if ne_top_used > 0:
                grad[self._off_term_mov:self._off_term_mov
                     + ne_top_used].add_(g_top[self.top_term_lo:
                                               self.top_term_lo + ne_top_used])
            grad[self._off_top_fil:self._off_top_fil + ft].add_(
                g_top[nt_phys:nt_total])
            grad[H + self._off_top_mov:H + self._off_top_mov + nt].add_(
                g_top[nt_total:nt_total + nt])
            if ne_top_used > 0:
                grad[H + self._off_term_mov:H + self._off_term_mov
                     + ne_top_used].add_(
                         g_top[nt_total + self.top_term_lo:nt_total
                               + self.top_term_lo + ne_top_used])
            grad[H + self._off_top_fil:H + self._off_top_fil + ft].add_(
                g_top[nt_total + nt_phys:2 * nt_total])

        def scatter_bot_grad(p, g_bot):
            grad = p.grad
            grad[self._off_bot_mov:self._off_bot_mov + nb].add_(g_bot[:nb])
            if ne_bot_used > 0:
                grad[self._off_term_mov:self._off_term_mov
                     + ne_bot_used].add_(g_bot[self.bot_term_lo:
                                               self.bot_term_lo + ne_bot_used])
            grad[self._off_bot_fil:self._off_bot_fil + fb].add_(
                g_bot[nb_phys:nb_total])
            grad[H + self._off_bot_mov:H + self._off_bot_mov + nb].add_(
                g_bot[nb_total:nb_total + nb])
            if ne_bot_used > 0:
                grad[H + self._off_term_mov:H + self._off_term_mov
                     + ne_bot_used].add_(
                         g_bot[nb_total + self.bot_term_lo:nb_total
                               + self.bot_term_lo + ne_bot_used])
            grad[H + self._off_bot_fil:H + self._off_bot_fil + fb].add_(
                g_bot[nb_total + nb_phys:2 * nb_total])

        def scatter_term_grad(p, g_term):
            grad = p.grad
            grad[self._off_term_mov:self._off_term_mov + ne].add_(g_term[:ne])
            grad[self._off_term_fil:self._off_term_fil + fe].add_(
                g_term[ne_phys:ne_total])
            grad[H + self._off_term_mov:H + self._off_term_mov + ne].add_(
                g_term[ne_total:ne_total + ne])
            grad[H + self._off_term_fil:H + self._off_term_fil + fe].add_(
                g_term[ne_total + ne_phys:2 * ne_total])

        def obj_and_grad_fn(p):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            else:
                p.grad.zero_()

            # 1) split mov_node_pos_all into per-layer pieces (views)
            (x_tm, x_bm, x_em, x_tf, x_bf, x_ef,
             y_tm, y_bm, y_em, y_tf, y_bf, y_ef) = self._split(p)

            # Build/evaluate/scatter one layer at a time.  PlaceObj.backward
            # frees that layer's graph before the next layer is constructed,
            # preventing three full pin-position and density graphs from
            # overlapping in memory.
            with torch.no_grad():
                pos_top = self._build_pos_top(x_tm, y_tm, x_em, y_em, x_tf,
                                              y_tf)
            pos_top.requires_grad_(True)
            obj_top, _ = self.model_top.obj_and_grad_fn(pos_top)
            scatter_top_grad(p, pos_top.grad)
            obj = obj_top.detach()
            del pos_top, obj_top

            with torch.no_grad():
                pos_bot = self._build_pos_bot(x_bm, y_bm, x_em, y_em, x_bf,
                                              y_bf)
            pos_bot.requires_grad_(True)
            obj_bot, _ = self.model_bot.obj_and_grad_fn(pos_bot)
            scatter_bot_grad(p, pos_bot.grad)
            obj = obj + obj_bot.detach()
            del pos_bot, obj_bot

            with torch.no_grad():
                pos_term = self._build_pos_term(x_em, y_em, x_ef, y_ef)
            pos_term.requires_grad_(True)
            obj_term, _ = self.model_term.obj_and_grad_fn(pos_term)
            scatter_term_grad(p, pos_term.grad)
            obj = obj + obj_term.detach()
            del pos_term, obj_term
            return obj, p.grad

        return obj_and_grad_fn

    # ------------------------------------------------------------------
    # Constraints / metrics
    # ------------------------------------------------------------------
    def constraint_fn(self, mov_node_pos_all):
        """Clamp pos to legal core via each placedb's move_boundary_op.

        Re-pack the clamped per-placedb positions back into
        ``mov_node_pos_all`` for movable cells, terminal_NIs, and fillers.
        """
        with torch.no_grad():
            (x_tm, x_bm, x_em, x_tf, x_bf, x_ef,
             y_tm, y_bm, y_em, y_tf, y_bf, y_ef) = self._split(mov_node_pos_all)
            data = mov_node_pos_all.data
            H = self._half
            nt = self.n_top_mov
            nb = self.n_bot_mov
            ne = self.n_term
            ft = self.n_top_fil
            fb = self.n_bot_fil
            fe = self.n_term_fil
            nt_total = self.n_top_total
            nb_total = self.n_bot_total
            ne_total = self.n_term_total
            nt_phys = self.n_top_phys
            nb_phys = self.n_bot_phys
            ne_phys = self.n_term_phys

            pos_top = self._build_pos_top(x_tm, y_tm, x_em, y_em, x_tf,
                                          y_tf)
            self.dp_top.basic_place.op_collections.move_boundary_op(pos_top)
            data[self._off_top_mov:self._off_top_mov + nt].copy_(pos_top[:nt])
            data[self._off_top_fil:self._off_top_fil + ft].copy_(
                pos_top[nt_phys:nt_total])
            data[H + self._off_top_mov:H + self._off_top_mov + nt].copy_(
                pos_top[nt_total:nt_total + nt])
            data[H + self._off_top_fil:H + self._off_top_fil + ft].copy_(
                pos_top[nt_total + nt_phys:2 * nt_total])
            del pos_top

            pos_bot = self._build_pos_bot(x_bm, y_bm, x_em, y_em, x_bf,
                                          y_bf)
            self.dp_bot.basic_place.op_collections.move_boundary_op(pos_bot)
            data[self._off_bot_mov:self._off_bot_mov + nb].copy_(pos_bot[:nb])
            data[self._off_bot_fil:self._off_bot_fil + fb].copy_(
                pos_bot[nb_phys:nb_total])
            data[H + self._off_bot_mov:H + self._off_bot_mov + nb].copy_(
                pos_bot[nb_total:nb_total + nb])
            data[H + self._off_bot_fil:H + self._off_bot_fil + fb].copy_(
                pos_bot[nb_total + nb_phys:2 * nb_total])
            del pos_bot

            pos_term = self._build_pos_term(x_em, y_em, x_ef, y_ef)
            self.dp_term.basic_place.op_collections.move_boundary_op(pos_term)
            data[self._off_term_mov:self._off_term_mov + ne].copy_(
                pos_term[:ne])
            data[self._off_term_fil:self._off_term_fil + fe].copy_(
                pos_term[ne_phys:ne_total])
            data[H + self._off_term_mov:H + self._off_term_mov + ne].copy_(
                pos_term[ne_total:ne_total + ne])
            data[H + self._off_term_fil:H + self._off_term_fil + fe].copy_(
                pos_term[ne_total + ne_phys:2 * ne_total])

    def overflow(self, mov_node_pos_all):
        """Per-layer absolute density overflow (top, bot, term) as floats."""
        with torch.no_grad():
            (x_tm, x_bm, x_em, x_tf, x_bf, x_ef,
             y_tm, y_bm, y_em, y_tf, y_bf, y_ef) = self._split(
                 mov_node_pos_all)
            pos_top = self._build_pos_top(x_tm, y_tm, x_em, y_em, x_tf,
                                          y_tf)
            ov_top, _ = self.dp_top.basic_place.op_collections.density_overflow_op(
                pos_top)
            top_value = float(ov_top.sum().item())
            del pos_top, ov_top
            pos_bot = self._build_pos_bot(x_bm, y_bm, x_em, y_em, x_bf,
                                          y_bf)
            ov_bot, _ = self.dp_bot.basic_place.op_collections.density_overflow_op(
                pos_bot)
            bot_value = float(ov_bot.sum().item())
            del pos_bot, ov_bot
            pos_term = self._build_pos_term(x_em, y_em, x_ef, y_ef)
            ov_term, _ = self.dp_term.basic_place.op_collections.density_overflow_op(
                pos_term)
            term_value = float(ov_term.sum().item())
        return top_value, bot_value, term_value

    def evaluate_metrics(self, mov_node_pos_all, iteration):
        """Build EvalMetrics for each layer (.hpwl + normalized .overflow)."""
        with torch.no_grad():
            (x_tm, x_bm, x_em, x_tf, x_bf, x_ef,
             y_tm, y_bm, y_em, y_tf, y_bf, y_ef) = self._split(
                 mov_node_pos_all)

            pos_top = self._build_pos_top(x_tm, y_tm, x_em, y_em, x_tf,
                                          y_tf)
            cur_top = EvalMetrics.EvalMetrics(iteration)
            cur_top.gamma = self.model_top.gamma.data
            cur_top.density_weight = self.model_top.density_weight.data
            cur_top.evaluate(self.dp_top.placedb, self._eval_ops_top, pos_top,
                             self.dp_top.basic_place.data_collections)
            del pos_top

            pos_bot = self._build_pos_bot(x_bm, y_bm, x_em, y_em, x_bf,
                                          y_bf)
            cur_bot = EvalMetrics.EvalMetrics(iteration)
            cur_bot.gamma = self.model_bot.gamma.data
            cur_bot.density_weight = self.model_bot.density_weight.data
            cur_bot.evaluate(self.dp_bot.placedb, self._eval_ops_bot, pos_bot,
                             self.dp_bot.basic_place.data_collections)
            del pos_bot

            pos_term = self._build_pos_term(x_em, y_em, x_ef, y_ef)
            cur_term = EvalMetrics.EvalMetrics(iteration)
            cur_term.gamma = self.model_term.gamma.data
            cur_term.density_weight = self.model_term.density_weight.data
            cur_term.evaluate(self.dp_term.placedb, self._eval_ops_term,
                              pos_term,
                              self.dp_term.basic_place.data_collections)

        return cur_top, cur_bot, cur_term

    def evaluate_d2d_hpwl(self, mov_node_pos_all):
        """Evaluate current co-place state with the final D2D HPWL metric."""
        with torch.no_grad():
            (x_tm, x_bm, x_em, x_tf, x_bf, x_ef,
             y_tm, y_bm, y_em, y_tf, y_bf, y_ef) = self._split(
                 mov_node_pos_all)
            pos_top = self._build_pos_top(x_tm, y_tm, x_em, y_em, x_tf,
                                          y_tf)
            pos_bot = self._build_pos_bot(x_bm, y_bm, x_em, y_em, x_bf,
                                          y_bf)
            flat_pos = self.dreamplace.dp_2d.pos.detach().clone()
            self.placer.op_wrapper.d2d_op_collections.pos_flattened_op(
                self.placer.tier, flat_pos, [pos_top, pos_bot])
            del pos_top, pos_bot
            pos_term = self._build_pos_term(x_em, y_em, x_ef, y_ef)
            terminal_names = self.dp_term.placedb.node_names[
                :self.num_terminal_NIs]
            return self.placer.op_wrapper.d2d_op_collections.hpwl_d2d_op(
                flat_pos, self.placer.cut_net_mask, self.placer.tier,
                pos_term, self.num_terminal_NIs, terminal_names)

    # ------------------------------------------------------------------
    # Outer-loop schedulers (gamma + density_weight)
    # ------------------------------------------------------------------
    def update_density_weights(self, cur_metrics, prev_metrics, iteration):
        """Trigger PlaceObj's adaptive density-weight update for each layer."""
        cur_top, cur_bot, cur_term = cur_metrics
        prev_top, prev_bot, prev_term = prev_metrics
        try:
            self.model_top.op_collections.update_density_weight_op(
                cur_top, prev_top, iteration)
        except Exception as e:
            logger.warning("update_density_weight (top) failed: %s", e)
        try:
            self.model_bot.op_collections.update_density_weight_op(
                cur_bot, prev_bot, iteration)
        except Exception as e:
            logger.warning("update_density_weight (bot) failed: %s", e)
        try:
            self.model_term.op_collections.update_density_weight_op(
                cur_term, prev_term, iteration)
        except Exception as e:
            logger.warning("update_density_weight (term) failed: %s", e)

    def update_gammas(self, iteration, cur_metrics):
        """Update each layer's WA gamma based on current normalized overflow."""
        cur_top, cur_bot, cur_term = cur_metrics
        try:
            self.model_top.op_collections.update_gamma_op(
                iteration, cur_top.overflow)
        except Exception as e:
            logger.warning("update_gamma (top) failed: %s", e)
        try:
            self.model_bot.op_collections.update_gamma_op(
                iteration, cur_bot.overflow)
        except Exception as e:
            logger.warning("update_gamma (bot) failed: %s", e)
        try:
            self.model_term.op_collections.update_gamma_op(
                iteration, cur_term.overflow)
        except Exception as e:
            logger.warning("update_gamma (term) failed: %s", e)

    # ------------------------------------------------------------------
    # Plot / writeback
    # ------------------------------------------------------------------
    def plot(self, mov_node_pos_all, iteration, subdir="co_place"):
        """Save layout plots for the three layers (top tier, bot tier, terminal).

        Files are written to:
            {result_dir_root}/{tag}/{subdir}/plot/iter{iteration:04d}.png
        where tag is "tier0" / "tier1" / "terminal".
        """
        with torch.no_grad():
            result_dir = self.params.result_dir_root
            (x_tm, x_bm, x_em, x_tf, x_bf, x_ef,
             y_tm, y_bm, y_em, y_tf, y_bf, y_ef) = self._split(
                 mov_node_pos_all)
            entries = (
                ("tier0", self.dp_top,
                 lambda: self._build_pos_top(x_tm, y_tm, x_em, y_em, x_tf,
                                             y_tf)),
                ("tier1", self.dp_bot,
                 lambda: self._build_pos_bot(x_bm, y_bm, x_em, y_em, x_bf,
                                             y_bf)),
                ("terminal", self.dp_term,
                 lambda: self._build_pos_term(x_em, y_em, x_ef, y_ef)),
            )
            for tag, dp, build_pos in entries:
                figdir = "%s/%s/%s/plot" % (result_dir, tag, subdir)
                os.makedirs(figdir, exist_ok=True)
                figname = "%s/iter%04d.png" % (figdir, iteration)
                try:
                    pos = build_pos().cpu()
                    dp.basic_place.op_collections.draw_place_op(pos, figname)
                    del pos
                except Exception as e:
                    logger.warning("co-place plot %s failed: %s", figname, e)

    def writeback(self, mov_node_pos_all):
        """Write final mov_node_pos_all values back to the three placedbs.

        Updates movable cells + D2D terminal_NIs + fillers in each placedb.
        """
        self._sync_placedb_pos(mov_node_pos_all)

        with torch.no_grad():
            # Keep DreamplaceBase.pos in sync (it points to placer.pos[0],
            # which is the same tensor as data_collections.pos[0])
            self.dp_top.pos = self.dp_top.basic_place.data_collections.pos[
                0].data
            self.dp_bot.pos = self.dp_bot.basic_place.data_collections.pos[
                0].data
            self.dp_term.pos = self.dp_term.basic_place.data_collections.pos[
                0].data
