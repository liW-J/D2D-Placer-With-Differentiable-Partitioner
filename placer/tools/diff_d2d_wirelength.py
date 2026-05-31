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

Note that fillers must be part of the joint optimizer state -- if they
were left static, the per-placedb density landscape would be skewed by
randomly initialized fillers and ePlace cannot converge.

The same ``mov_node_pos_all`` is shared by three placedbs:
  * dp_tier[0]: x/y_top_mov drives movable cells; x/y_term_mov drives
    terminal_NIs (fixed in dp_tier[0] view); x/y_top_fil drives fillers.
  * dp_tier[1]: same with bot.
  * dp_terminal: x/y_term_mov drives movable cells (the terminal_NIs
    themselves); x/y_term_fil drives fillers.
'''

import logging
import os
import torch

import dreamplace.PlaceObj as PlaceObj
import dreamplace.EvalMetrics as EvalMetrics

logger = logging.getLogger(__name__)


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

            # x parts
            x_top_mov = top_pos[:self.n_top_mov].clone()
            x_bot_mov = bot_pos[:self.n_bot_mov].clone()
            x_term_mov = term_pos[:self.n_term_mov].clone()
            x_top_fil = top_pos[self.n_top_phys:self.n_top_total].clone()
            x_bot_fil = bot_pos[self.n_bot_phys:self.n_bot_total].clone()
            x_term_fil = term_pos[self.n_term_phys:self.n_term_total].clone()

            # y parts
            y_top_mov = top_pos[self.n_top_total:self.n_top_total
                                + self.n_top_mov].clone()
            y_bot_mov = bot_pos[self.n_bot_total:self.n_bot_total
                                + self.n_bot_mov].clone()
            y_term_mov = term_pos[self.n_term_total:self.n_term_total
                                  + self.n_term_mov].clone()
            y_top_fil = top_pos[self.n_top_total
                                + self.n_top_phys:self.n_top_total * 2].clone()
            y_bot_fil = bot_pos[self.n_bot_total
                                + self.n_bot_phys:self.n_bot_total * 2].clone()
            y_term_fil = term_pos[self.n_term_total + self.n_term_phys:
                                  self.n_term_total * 2].clone()

        mov = torch.cat([
            x_top_mov, x_bot_mov, x_term_mov, x_top_fil, x_bot_fil, x_term_fil,
            y_top_mov, y_bot_mov, y_term_mov, y_top_fil, y_bot_fil, y_term_fil
        ])
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

        def build_axis(x_mov_, x_term_, x_fil_, base_offset):
            segs = [x_mov_]
            if term_lo > n_mov:
                segs.append(base[base_offset + n_mov:base_offset + term_lo])
            if ne_used > 0:
                segs.append(x_term_[:ne_used] + shift)
            if term_lo + ne_used < n_phys:
                segs.append(base[base_offset + term_lo + ne_used:base_offset
                                 + n_phys])
            segs.append(x_fil_)
            return torch.cat(segs)

        pos_x = build_axis(x_mov, x_term, x_fil, 0)
        pos_y = build_axis(y_mov, y_term, y_fil, n_total)
        return torch.cat([pos_x, pos_y])

    def _build_term_pos(self, x_term_mov, y_term_mov, x_term_fil, y_term_fil):
        """Assemble dp_terminal's pos tensor with movable terminal_NIs +
        fillers; trailing fixed nodes (if any) come from current base pos.
        """
        base = self.dp_term.basic_place.data_collections.pos[0].detach()
        n_mov = self.n_term_mov
        n_phys = self.n_term_phys
        n_total = self.n_term_total

        if n_phys > n_mov:
            mid_x = base[n_mov:n_phys]
            mid_y = base[n_total + n_mov:n_total + n_phys]
            pos_x = torch.cat([x_term_mov, mid_x, x_term_fil])
            pos_y = torch.cat([y_term_mov, mid_y, y_term_fil])
        else:
            pos_x = torch.cat([x_term_mov, x_term_fil])
            pos_y = torch.cat([y_term_mov, y_term_fil])
        return torch.cat([pos_x, pos_y])

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
            pos_top, pos_bot, pos_term = self._build_all_pos(mov_node_pos_all)
            self.dp_top.basic_place.data_collections.pos[0].data.copy_(pos_top)
            self.dp_bot.basic_place.data_collections.pos[0].data.copy_(pos_bot)
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

        def scatter_grads(p, g_top, g_bot, g_term):
            grad = p.grad

            # ----- movable cells in dp_top / dp_bot -----
            grad[self._off_top_mov:self._off_top_mov + nt].add_(g_top[:nt])
            grad[self._off_bot_mov:self._off_bot_mov + nb].add_(g_bot[:nb])

            # ----- D2D terminal_NIs receive grads from three sources -----
            #  * model_top: terminal_NI region [top_term_lo:top_term_hi]
            #    (typically zero because pin_mask_ignore_fixed_macros masks
            #     out fixed pins, but we still accumulate for correctness)
            #  * model_bot: same as above
            #  * model_term: terminal_NIs are *movable* in dp_term, so this
            #    is the dominant gradient source for x/y_term.
            ne_top_used = min(self.top_term_hi - self.top_term_lo, ne)
            ne_bot_used = min(self.bot_term_hi - self.bot_term_lo, ne)
            if ne_top_used > 0:
                grad[self._off_term_mov:self._off_term_mov
                     + ne_top_used].add_(g_top[self.top_term_lo:
                                               self.top_term_lo + ne_top_used])
            if ne_bot_used > 0:
                grad[self._off_term_mov:self._off_term_mov
                     + ne_bot_used].add_(g_bot[self.bot_term_lo:
                                               self.bot_term_lo + ne_bot_used])
            grad[self._off_term_mov:self._off_term_mov + ne].add_(g_term[:ne])

            # ----- fillers per-placedb -----
            grad[self._off_top_fil:self._off_top_fil + ft].add_(
                g_top[nt_phys:nt_total])
            grad[self._off_bot_fil:self._off_bot_fil + fb].add_(
                g_bot[nb_phys:nb_total])
            grad[self._off_term_fil:self._off_term_fil + fe].add_(
                g_term[ne_phys:ne_total])

            # ===== y part =====
            grad[H + self._off_top_mov:H + self._off_top_mov + nt].add_(
                g_top[nt_total:nt_total + nt])
            grad[H + self._off_bot_mov:H + self._off_bot_mov + nb].add_(
                g_bot[nb_total:nb_total + nb])
            if ne_top_used > 0:
                grad[H + self._off_term_mov:H + self._off_term_mov
                     + ne_top_used].add_(
                         g_top[nt_total + self.top_term_lo:nt_total
                               + self.top_term_lo + ne_top_used])
            if ne_bot_used > 0:
                grad[H + self._off_term_mov:H + self._off_term_mov
                     + ne_bot_used].add_(
                         g_bot[nb_total + self.bot_term_lo:nb_total
                               + self.bot_term_lo + ne_bot_used])
            grad[H + self._off_term_mov:H + self._off_term_mov + ne].add_(
                g_term[ne_total:ne_total + ne])
            grad[H + self._off_top_fil:H + self._off_top_fil + ft].add_(
                g_top[nt_total + nt_phys:2 * nt_total])
            grad[H + self._off_bot_fil:H + self._off_bot_fil + fb].add_(
                g_bot[nb_total + nb_phys:2 * nb_total])
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

            # 2) build per-layer leaf tensors. Detach + clone so PlaceObj
            #    can write its preconditioned grad directly. We then
            #    accumulate grads back to p.grad below.
            with torch.no_grad():
                pos_top = self._build_pos_top(x_tm, y_tm, x_em, y_em, x_tf,
                                              y_tf).clone()
                pos_bot = self._build_pos_bot(x_bm, y_bm, x_em, y_em, x_bf,
                                              y_bf).clone()
                pos_term = self._build_pos_term(x_em, y_em, x_ef, y_ef).clone()
            pos_top.requires_grad_(True)
            pos_bot.requires_grad_(True)
            pos_term.requires_grad_(True)

            # 3) per-layer dreamplace obj_and_grad (wl + dw*den + precondition)
            obj_top, _ = self.model_top.obj_and_grad_fn(pos_top)
            obj_bot, _ = self.model_bot.obj_and_grad_fn(pos_bot)
            obj_term, _ = self.model_term.obj_and_grad_fn(pos_term)

            # 4) scatter preconditioned grads (incl. filler) back to p.grad
            scatter_grads(p, pos_top.grad, pos_bot.grad, pos_term.grad)

            obj = obj_top.detach() + obj_bot.detach() + obj_term.detach()
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
            pos_top = self._build_pos_top(x_tm, y_tm, x_em, y_em, x_tf,
                                          y_tf).detach().clone()
            pos_bot = self._build_pos_bot(x_bm, y_bm, x_em, y_em, x_bf,
                                          y_bf).detach().clone()
            pos_term = self._build_pos_term(x_em, y_em, x_ef,
                                            y_ef).detach().clone()
            self.dp_top.basic_place.op_collections.move_boundary_op(pos_top)
            self.dp_bot.basic_place.op_collections.move_boundary_op(pos_bot)
            self.dp_term.basic_place.op_collections.move_boundary_op(pos_term)

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

            # x parts
            data[self._off_top_mov:self._off_top_mov + nt].copy_(pos_top[:nt])
            data[self._off_bot_mov:self._off_bot_mov + nb].copy_(pos_bot[:nb])
            data[self._off_term_mov:self._off_term_mov + ne].copy_(
                pos_term[:ne])
            data[self._off_top_fil:self._off_top_fil + ft].copy_(
                pos_top[nt_phys:nt_total])
            data[self._off_bot_fil:self._off_bot_fil + fb].copy_(
                pos_bot[nb_phys:nb_total])
            data[self._off_term_fil:self._off_term_fil + fe].copy_(
                pos_term[ne_phys:ne_total])
            # y parts
            data[H + self._off_top_mov:H + self._off_top_mov + nt].copy_(
                pos_top[nt_total:nt_total + nt])
            data[H + self._off_bot_mov:H + self._off_bot_mov + nb].copy_(
                pos_bot[nb_total:nb_total + nb])
            data[H + self._off_term_mov:H + self._off_term_mov + ne].copy_(
                pos_term[ne_total:ne_total + ne])
            data[H + self._off_top_fil:H + self._off_top_fil + ft].copy_(
                pos_top[nt_total + nt_phys:2 * nt_total])
            data[H + self._off_bot_fil:H + self._off_bot_fil + fb].copy_(
                pos_bot[nb_total + nb_phys:2 * nb_total])
            data[H + self._off_term_fil:H + self._off_term_fil + fe].copy_(
                pos_term[ne_total + ne_phys:2 * ne_total])

    def overflow(self, mov_node_pos_all):
        """Per-layer absolute density overflow (top, bot, term) as floats."""
        with torch.no_grad():
            pos_top, pos_bot, pos_term = self._build_all_pos(mov_node_pos_all)
            ov_top, _ = self.dp_top.basic_place.op_collections.density_overflow_op(
                pos_top)
            ov_bot, _ = self.dp_bot.basic_place.op_collections.density_overflow_op(
                pos_bot)
            ov_term, _ = self.dp_term.basic_place.op_collections.density_overflow_op(
                pos_term)
        return (float(ov_top.sum().item()), float(ov_bot.sum().item()),
                float(ov_term.sum().item()))

    def evaluate_metrics(self, mov_node_pos_all, iteration):
        """Build EvalMetrics for each layer (.hpwl + normalized .overflow)."""
        with torch.no_grad():
            pos_top, pos_bot, pos_term = self._build_all_pos(mov_node_pos_all)

            cur_top = EvalMetrics.EvalMetrics(iteration)
            cur_top.gamma = self.model_top.gamma.data
            cur_top.density_weight = self.model_top.density_weight.data
            cur_top.evaluate(self.dp_top.placedb, self._eval_ops_top, pos_top,
                             self.dp_top.basic_place.data_collections)

            cur_bot = EvalMetrics.EvalMetrics(iteration)
            cur_bot.gamma = self.model_bot.gamma.data
            cur_bot.density_weight = self.model_bot.density_weight.data
            cur_bot.evaluate(self.dp_bot.placedb, self._eval_ops_bot, pos_bot,
                             self.dp_bot.basic_place.data_collections)

            cur_term = EvalMetrics.EvalMetrics(iteration)
            cur_term.gamma = self.model_term.gamma.data
            cur_term.density_weight = self.model_term.density_weight.data
            cur_term.evaluate(self.dp_term.placedb, self._eval_ops_term,
                              pos_term,
                              self.dp_term.basic_place.data_collections)

        return cur_top, cur_bot, cur_term

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
            pos_top, pos_bot, pos_term = self._build_all_pos(mov_node_pos_all)
            pos_top = pos_top.detach().cpu()
            pos_bot = pos_bot.detach().cpu()
            pos_term = pos_term.detach().cpu()

            result_dir = self.params.result_dir_root
            entries = [
                ("tier0", self.dp_top, pos_top),
                ("tier1", self.dp_bot, pos_bot),
                ("terminal", self.dp_term, pos_term),
            ]
            for tag, dp, pos in entries:
                figdir = "%s/%s/%s/plot" % (result_dir, tag, subdir)
                os.makedirs(figdir, exist_ok=True)
                figname = "%s/iter%04d.png" % (figdir, iteration)
                try:
                    dp.basic_place.op_collections.draw_place_op(pos, figname)
                except Exception as e:
                    logger.warning("co-place plot %s failed: %s", figname, e)

    def writeback(self, mov_node_pos_all):
        """Write final mov_node_pos_all values back to the three placedbs.

        Updates movable cells + D2D terminal_NIs + fillers in each placedb.
        """
        with torch.no_grad():
            pos_top, pos_bot, pos_term = self._build_all_pos(mov_node_pos_all)
            self.dp_top.basic_place.data_collections.pos[0].data.copy_(pos_top)
            self.dp_bot.basic_place.data_collections.pos[0].data.copy_(pos_bot)
            self.dp_term.basic_place.data_collections.pos[0].data.copy_(
                pos_term)

            # Keep DreamplaceBase.pos in sync (it points to placer.pos[0],
            # which is the same tensor as data_collections.pos[0])
            self.dp_top.pos = self.dp_top.basic_place.data_collections.pos[
                0].data
            self.dp_bot.pos = self.dp_bot.basic_place.data_collections.pos[
                0].data
            self.dp_term.pos = self.dp_term.basic_place.data_collections.pos[
                0].data
