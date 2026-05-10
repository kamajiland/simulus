# FILE INFO ###################################################
# Author: Jason Liu <jasonxliu2010@gmail.com>
# Created on July 28, 2019
# Last Update: Time-stamp: <2019-08-19 19:25:26 liux>
###############################################################

from collections import defaultdict
import multiprocessing as mp
#from concurrent import futures
import time, atexit, sys, ctypes

from .simulus import *
from .simulator import *

__all__ = ["sync"]

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# MPI tags for cross-rank transport. Distinct tags let the receiver
# dispatch on type without unpacking the payload. CMB uses NULL+REAL;
# STM uses TS_UPDATE+REAL. The tags are protocol-specific (CMB and STM
# never run simultaneously, but distinct tags keep diagnostics clean).
_CMB_NULL_TAG = 4001
_CMB_REAL_TAG = 4002
_STM_TS_TAG   = 4003
_STM_REAL_TAG = 4004

# STM Phase 3 polling interval (seconds). Default 1ms; override via
# the STM_POLL_SEC env var. Tuned up from the original 100us after
# pluto SPMD-at-P=1 diagnostics (benchmarks/diag_stm_spmd.py) showed
# the 100us default thrashes when smp_ways approaches or exceeds the
# physical core count: at n=64 smp_ways=64 the wall dropped from 3.6s
# (100us) to 2.0s (1ms) to 1.3s (10ms). 1ms is a safe default --- no
# SMP regression at any scale tested, ~1.8x SPMD-mode speedup. Set
# STM_POLL_SEC=0.0001 to recover the original behavior for A/B runs.
import os as _os
try:
    _STM_POLL_SEC = float(_os.environ.get('STM_POLL_SEC', '0.001'))
except (TypeError, ValueError):
    _STM_POLL_SEC = 0.001


class _Channel(object):
    """A directed channel from a source LP to a destination mailbox.

    Used by asynchronous synchronization protocols (CMB, STM) to track
    the safe-time guarantee the source has communicated to the destination.

    The channel state lives at the destination LP: 'front' is updated by the
    destination on receipt of nulls or real messages; the source merely
    dispatches updates over the transport.

    For STM (future), this is the same channel-id key but the safe-time
    lives in shared memory (an mp.RawArray slot indexed by ts_idx) rather
    than in the front attribute.
    """

    __slots__ = ('src_name', 'dst_name', 'mailbox_name', 'min_delay',
                 'front', 'ts_idx')

    def __init__(self, src_name, dst_name, mailbox_name, min_delay):
        self.src_name = src_name
        self.dst_name = dst_name
        self.mailbox_name = mailbox_name
        self.min_delay = min_delay
        self.front = 0.0   # CMB: updated by drain on the destination side
        self.ts_idx = -1   # STM: index into shared-memory ts array (set later)


class sync(object):
    """A synchronized group of simulators whose simulation clocks will
    advance synchronously."""

    _simulus = None
    
    def __init__(self, sims, enable_smp=False, enable_spmd=False, lookahead=infinite_time, smp_ways=None, protocol='ctw'):
        """Create a synchronized group of multiple simulators. 

        Bring all simulators in the group to synchrony; that is, the
        simulation clocks of all the simulators in the group, from now
        on, will be advanced synchronously in a coordinated fashion.

        Args:
            sims (list or tuple): a list of local simulators; the
                simulators are identified either by their names or as
                direct references to instances

            enable_smp (bool): enable SMP (Symmetric Multi-Processing)
                mode, in which case each local simulator will run as a
                separate process, and communication between the
                simulators will be facilitated through inter-process
                communication (IPC) mechanisms; the default is False,
                in which case all local simulators will run
                sequentially within the same process

            enable_spmd (bool): enable SPMD (Single Program Multiple
                Data) mode, in which case multiple simulus instances,
                potentially on distributed memory machines, will run
                in parallel, where communication between the simulus
                instances will be facilitated through the Message
                Passing Interface (MPI); the default is False, in
                which case the local simulus instance will run
                standalone with all simulators running either
                sequentially as one process (when enable_smp is
                False), or in parallel as separate processes (when
                enable_smp is True)

            lookahead (float): the maximum difference in simulation
                time between the simulators in the group; the default
                is infinity; the final lookahead for parallel
                simulation will be determined by the min delays of the
                named mailboxes of the simulators

            smp_ways (int): the maximum number of processes to be
                created for shared-memory multiprocessing. This
                parameter is only used when SMP is enabled.

        Returns: 
            This function creates, initializes, and returns a
            synchronized group. The simulators will first advance
            their simulation clock (asynchronously) to the maximum
            simulation time among all simulators (including both local
            simulators and remote ones, if enable_spmd is True). When
            the function returns, the listed simulators are bound to
            the synchronized group.  That is, the simulation clocks of
            the simulators will be advanced synchronously from now on:
            all simulators will process events (including all messages
            sent between the simulators) in the proper timestamp
            order. (This is also known as the local causality
            constraint in the parallel discrete-event simulation
            literature.)

        """

        # the simulus instance is a class variable
        if not sync._simulus:
            sync._simulus = _Simulus()

        if lookahead <= 0:
            errmsg = "sync(looahead=%r) expects a positive lookahead" % lookahead
            log.error(errmsg)
            raise ValueError(errmsg)
            
        if smp_ways is not None and \
           (not isinstance(smp_ways, int) or smp_ways <= 0):
            errmsg = "sync(smp_ways=%r) expects a positive integer" % smp_ways
            log.error(errmsg)
            raise ValueError(errmsg)

        # Accept legacy 'yawns' and 'soft_tm' as aliases for 'ctw' and 'stm'.
        protocol_aliases = {'yawns': 'ctw', 'soft_tm': 'stm'}
        protocol = protocol_aliases.get(protocol, protocol)
        if protocol not in ('ctw', 'cmb', 'stm'):
            errmsg = "sync(protocol=%r) expects 'ctw', 'cmb', or 'stm'" % protocol
            log.error(errmsg)
            raise ValueError(errmsg)

        self._activated = False  # keep it false until we are done with creating the sync group
        self._smp = enable_smp
        self._smp_ways = smp_ways
        self._spmd = enable_spmd
        self._protocol = protocol
        # CMB and STM channel graph (populated below for asynchronous protocols).
        self._channels = {}            # (src_name, mb_name) -> _Channel
        self._lp_inputs = defaultdict(list)   # lp_name -> list of (src_name, mb_name)
        self._lp_outputs = defaultdict(list)  # lp_name -> list of (src_name, mb_name)
        # CMB transport: per-pid data queues + final-exit barrier + done-pid
        # counter (allocated in run() before fork when SMP is enabled and
        # protocol == 'cmb').
        self._local_data_queues = None
        self._cmb_done_barrier = None
        self._cmb_done_count = None
        # Per-worker pid context, set on entry to _smp_run_cmb / _smp_run_stm
        # so sync.send() (called from inside sim._run) knows which pid is
        # dispatching. -1 means "not currently running an async protocol";
        # the legacy CTW path falls through.
        self._cmb_my_pid = -1
        self._stm_my_pid = -1
        # STM shared-memory channel-timestamp array (the ts[] of the
        # paper). One double per channel; populated in _build_channel_graph.
        self._channel_ts = None
        if self._spmd and not sync._simulus.args.mpi:
            errmsg = "sync(enable_spmd=True) requires MPI support (use --mpi or -x command-line option)"
            log.error(errmsg)
            raise ValueError(errmsg)

        # the local simulators are provided either by names or as
        # direct references
        self._local_sims = {} # a map from names to simulator instances
        self._all_sims = {} # a map from names to mpi ranks (identifying simulator's location)
        now_max = minus_infinite_time # to find out the max simulation time of all simulators
        if not isinstance(sims, (list, tuple)):
            errmsg = "sync(sims=%r) expects a list of simulators" % sims
            log.error(errmsg)
            raise TypeError(errmsg)
        for s in sims:
            if isinstance(s, str):
                # if simulator name is provided, turn it into instance
                ss = sync._simulus.get_simulator(s)
                if ss is None:
                    errmsg = "sync() expects a list of simulators, but '%s' is not" % s
                    log.error(errmsg)
                    raise ValueError(errmsg)
                else: s = ss

            # the item must be a simulator instance
            if isinstance(s, simulator):
                if s._insync:
                    # the simulator's already in a sync group
                    if s._insync != self:
                        errmsg = "sync() simulator '%s' belongs to another group" % s.name
                    else:
                        errmsg = "sync() duplicate simulator '%s' listed" % s.name
                    log.error(errmsg)
                    raise ValueError(errmsg)
                else:
                    s._insync = self
                    self._local_sims[s.name] = s
                    self._all_sims[s.name] = sync._simulus.comm_rank
                    if s.now > now_max: now_max = s.now
            else:
                errmsg = "sync() expects a list of simulators, but %r is not" % s
                log.error(errmsg)
                raise ValueError(errmsg)

        # a synchronized group cannot be empty
        if len(self._local_sims) < 1:
            errmsg = "sync() sims should not be empty"
            log.error(errmsg)
            raise ValueError(errmsg)

        # if this is a global synchronization group (i.e., when
        # enable_spmd is true), we need to learn about the remote
        # simulators (e.g., the ranks at which they reside), and get
        # the maximum simulation time of all simulators in the group
        if self._spmd:
            self._all_sims = sync._simulus.allgather(self._all_sims)
            now_max = sync._simulus.allreduce(now_max, max)

        # find all mailboxes attached to local simulators.
        # _all_mboxes layout: mbname -> (owner_sname, min_delay, nparts, source_names)
        # source_names is a tuple of simulator names allowed to send to this mailbox,
        # or None for legacy many-writer (CTW only).
        self._lookahead = lookahead
        self._local_mboxes = {}
        self._all_mboxes = {}
        for sname, sim in self._local_sims.items():
            for mbname, mb in sim._mailboxes.items():
                if mbname in self._local_mboxes:
                    if sim == mb._sim:
                        errmsg = "sync() duplicate mailbox named '%s' in simulator '%s'" % \
                                 (mbname, sname)
                    else:
                        errmsg = "sync() duplicate mailbox name '%s' in simulators '%s' and '%s'" % \
                                 (mbname, sname, mb._sim.name)
                    log.error(errmsg)
                    raise ValueError(errmsg)
                else:
                    self._local_mboxes[mbname] = mb
                    src_names = self._normalize_source(mb.source)
                    self._all_mboxes[mbname] = (sname, mb.min_delay, mb.nparts, src_names)
                    if mb.min_delay < self._lookahead:
                        self._lookahead = mb.min_delay

        # if this is a global synchronization group (i.e., when
        # enable_spmd is true) , we need to learn about the remote
        # mailboxes and the min delays of all mailboxes
        if self._spmd:
            self._all_mboxes = sync._simulus.allgather(self._all_mboxes)
            self._lookahead = sync._simulus.allreduce(self._lookahead, min)

        # lookahead must be strictly positive
        if self._lookahead <= 0:
            errmsg = "sync() expects positive lookahead; " + \
                   "check min_delay of mailboxes in simulators"
            log.error(errmsg)
            raise ValueError(errmsg)

        # bring all local simulators' time to the max now
        for sname, sim in self._local_sims.items():
            if sim.now < now_max:
                sim._run(now_max, True)
        self.now = now_max

        # Build channel graph for asynchronous protocols (cmb, stm).
        # Must come after self.now is set, since channel fronts are
        # initialized to (now + min_delay).
        if self._protocol in ('cmb', 'stm'):
            self._build_channel_graph()

        log.info("[r%d] creating sync (enable_smp=%r, enable_spmd=%r): now=%g, lookahead=%g" %
                 (sync._simulus.comm_rank, self._smp, self._spmd, self.now, self._lookahead))
        for sname, simrank in self._all_sims.items():
            log.info("[r%d] >> simulator '%s' => r%d" %
                     (sync._simulus.comm_rank, sname, simrank))
        for mbname, (sname, mbdly, mbparts, _src) in self._all_mboxes.items():
            log.info("[r%d] >> mailbox '%s' => sim='%s', min_delay=%g, nparts=%d" %
                     (sync._simulus.comm_rank, mbname, sname, mbdly, mbparts))

        # ready for next window
        self._remote_msgbuf = defaultdict(list) # a map from rank to list of remote messages
        self._remote_future = infinite_time
        self._local_partitions = None
        self._activated = True

    @staticmethod
    def _normalize_source(src):
        """Convert a Mailbox.source value into a tuple of simulator names,
        or None for the legacy many-writer behavior."""
        if src is None:
            return None
        if isinstance(src, simulator):
            return (src.name,)
        if isinstance(src, (list, tuple)):
            names = []
            for s in src:
                if isinstance(s, simulator):
                    names.append(s.name)
                elif isinstance(s, str):
                    names.append(s)
                else:
                    raise TypeError(
                        "mailbox(source=%r): each source must be a simulator "
                        "or simulator name" % (src,))
            return tuple(names)
        if isinstance(src, str):
            return (src,)
        raise TypeError(
            "mailbox(source=%r): expected simulator, simulator name, list, "
            "or tuple" % (src,))

    def _build_channel_graph(self):
        """Build the (src_name, mb_name) -> _Channel registry plus per-LP
        input/output indexes. Called only when protocol in {'cmb', 'stm'}.

        Each (source, mailbox) pair is one directed channel. Mailboxes that
        did not declare a source are an error: asynchronous protocols
        require explicit channel-graph declaration.

        For STM, also allocate a shared-memory array of channel timestamps
        (one slot per channel). The array is sized once and inherited by
        forked SMP children. Each channel's safe-time guarantee is read
        and written through its `ts_idx` slot in this array; this is the
        ts[c] of Algorithm 1 / Algorithm 2 in the paper.
        """
        for mbname, (dst_sname, min_delay, _nparts, src_names) in \
                self._all_mboxes.items():
            if src_names is None:
                errmsg = ("sync(protocol=%r) requires every mailbox to "
                          "declare a source; mailbox '%s' has none" %
                          (self._protocol, mbname))
                log.error(errmsg)
                raise ValueError(errmsg)
            for src_name in src_names:
                if src_name not in self._all_sims:
                    errmsg = ("mailbox '%s' source '%s' is not a simulator "
                              "in this sync group" % (mbname, src_name))
                    log.error(errmsg)
                    raise ValueError(errmsg)
                # Skip self-loops: an LP doesn't need a synchronization
                # channel with itself. A self-send still goes through the
                # local mailbox machinery, but the source has no synchronization
                # commitment to track for itself.
                if src_name == dst_sname:
                    continue
                ch_id = (src_name, mbname)
                ch = _Channel(src_name, dst_sname, mbname, min_delay)
                # Initialize front to self.now + min_delay so that LPs can
                # advance to (self.now + min_delay) without a null arriving.
                ch.front = self.now + min_delay
                self._channels[ch_id] = ch
                self._lp_inputs[dst_sname].append(ch_id)
                self._lp_outputs[src_name].append(ch_id)

        # STM: allocate the shared-memory ts[] array. One slot per channel,
        # double precision. Allocated as mp.RawArray so SMP children
        # inherit a single backing buffer through fork; works equivalently
        # in single-process mode. ts_idx records each channel's slot.
        #
        # Also allocate per-channel send/drain counters for the
        # snapshot-consistency check in _stm_compute_horizon. SPSC by
        # construction: send_count[c] is written only by the source pid
        # of channel c (after each REAL queue.put); drain_count[c] is
        # written only by the destination pid (after each REAL drain).
        # Plain RawArray writes suffice on x86 TSO.
        if self._protocol == 'stm':
            n_ch = len(self._channels)
            self._channel_ts = mp.RawArray(ctypes.c_double, n_ch)
            self._stm_send_count = mp.RawArray(ctypes.c_uint64, n_ch)
            self._stm_drain_count = mp.RawArray(ctypes.c_uint64, n_ch)
            for i, (ch_id, ch) in enumerate(self._channels.items()):
                ch.ts_idx = i
                self._channel_ts[i] = ch.front

        log.info("[r%d] sync built channel graph: %d channels, "
                 "protocol=%s" %
                 (sync._simulus.comm_rank, len(self._channels), self._protocol))

    def run(self, offset=None, until=None, show_runtime_report=False):
        """Process events of all simulators in the synchronized group each in
        timestamp order and advances the simulation time of all simulators 
        synchronously.

        Args:
            offset (float): relative time from now until which each of
                the simulators should advance its simulation time; if
                provided, it must be a non-negative value

            until (float): the absolute time until which each of the
                simulators should advance its simulation time; if
                provided, it must not be earlier than the current time

        The user can specify either 'offset' or 'until', but not both;
        if both 'offset' and 'until' are ignored, the simulator will
        run as long as there are events on the event lists of the
        simulators. Be careful, in this case, the simulation may run
        forever as for some models there may always be future events.

        Each simulator will process their events in timestamp order.
        Synchronization is provided so that messages sent between the
        simulators may not produce causality errors. When this method
        returns, the simulation time of the simulators will advance to
        the designated time, if either 'offset' or 'until' has been
        specified.  All events with timestamps smaller than the
        designated time will be processed. If neither 'offset' nor
        'until' is provided, the simulators will advance to the time
        of the last processed event among all simulators.

        If SPMD is enabled, at most one simulus instance (at rank 0)
        is allowed to specify the time (using 'offset' or 'until').
        All the other simulators must not specify the time.

        """

        # figure out the time, up to which all events will be processed
        upper_specified = 1
        if until == None and offset == None:
            upper = infinite_time
            upper_specified = 0
        elif until != None and offset != None:
            errmsg = "sync.run(until=%r, offset=%r) duplicate specification" % (until, offset)
            log.error(errmsg)
            raise ValueError(errmsg)
        elif offset != None:
            if offset < 0:
                errmsg = "sync.run(offset=%r) negative offset" % offset
                log.error(errmsg)
                raise ValueError(errmsg)
            upper = self.now + offset
        elif until < self.now:
            errmsg = "sync.run(until=%r) earlier than now (%r)" % (until, self.now)
            log.error(errmsg)
            raise ValueError(errmsg)
        else: upper = until

        if self._spmd:
            # only rank 0 can specify the upper for global synchronization
            if upper_specified > 0 and sync._simulus.comm_rank > 0:
                errmsg = "sync.run() 'offset' or 'until' allowed only on rank 0"
                log.error(errmsg)
                raise ValueError(errmsg)

            # we conduct a global synchronization to get the upper
            # time for all
            sync._simulus.bcast(0) # run command
            upper = sync._simulus.allreduce(upper, min)
            upper_specified = sync._simulus.allreduce(upper_specified, max)

        if self._local_partitions is None:
            if self._smp:
                # divide the local simulators among the CPU/cores
                sims = list(self._local_sims.keys())
                if self._smp_ways is None:
                    self._smp_ways = mp.cpu_count()
                k, m = divmod(len(sims), self._smp_ways)
                self._local_partitions = list(filter(lambda x: len(x)>0, \
                        (sims[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(self._smp_ways))))
                
                self._local_queues = {} # a map from pid to queue
                self._local_pids = {} # a map from simulator name to pid
                for pid, snames in enumerate(self._local_partitions):
                    self._local_queues[pid] = mp.Queue()
                    for s in snames: self._local_pids[s] = pid

                # On Python 3.8+ macOS uses "spawn" by default; mp.Process
                # instances can't be pickled under "spawn" (CPython issue #91090).
                # TODO: refactor _child_run so sync is not passed as an argument,
                # which would make this workaround unnecessary.
                if sys.platform != 'win32':
                    try:
                        mp.set_start_method("fork")
                    except RuntimeError:
                        pass  # start method already set; assume fork or acceptable alternative

                # Asynchronous-protocol transport: per-pid data queues +
                # done counter + done barrier. Both CMB and STM use the
                # same intra-rank termination dance (each pid bumps the
                # counter when its LPs are all done; everyone keeps
                # draining until all pids done; final barrier for clean
                # exit). For CMB the queues carry NULL+REAL; for STM
                # only REAL (timestamp updates go through the
                # shared-memory _channel_ts array set up in
                # _build_channel_graph). Allocated before fork so
                # children inherit the queue handles.
                if self._protocol in ('cmb', 'stm'):
                    n_pids = len(self._local_partitions)
                    self._local_data_queues = {
                        i: mp.Queue() for i in range(n_pids)
                    }
                    self._cmb_done_count = mp.Value('i', 0)
                    self._cmb_done_barrier = mp.Barrier(n_pids)

                # start the child processes
                self._child_procs = [mp.Process(target=sync._child_run, args=(self, i)) \
                                     for i in range(1, len(self._local_partitions))]
                for p in self._child_procs: p.start()
            else:
                self._local_partitions = [self._local_sims.keys()]
                self._local_pids = {} # a map from simulator name to pid
                for s in self._local_sims.keys():
                    self._local_pids[s] = 0

        atexit.register(self._run_finish)
        self._smp_run(0, upper, upper_specified)

        if self._simulus.comm_rank > 0:
            while True:
                cmd = self._simulus.bcast(None)
                if cmd == 0: # run command
                    upper = sync._simulus.allreduce(infinite_time, min)
                    upper_specified = sync._simulus.allreduce(0, max)
                    self._smp_run(0, upper, upper_specified)
                elif cmd == 1: # report command
                    self._smp_report(0)
                else: # stop command
                    assert cmd == 2
                    break
        else:
            if show_runtime_report:
                self.show_runtime_report()
                
    def _child_run(self, pid):
        """The child processes running in SMP mode."""

        # When sync is pickled and sent to the child process, pickle's
        # object graph traversal creates independent copies of each
        # simulator — so self._local_sims and sync._simulus.named_simulators
        # end up pointing to different objects with the same content.
        # Re-register from _local_sims so the rest of the child code uses
        # a single consistent set of simulator references.
        for name, sim in self._local_sims.items():
            sync._simulus.register_simulator(name, sim)

        log.info("[r%d] sync._child_run(pid=%d): partitions=%r" %
                 (sync._simulus.comm_rank, pid, self._local_partitions))
        assert self._smp and pid>0

        # if smp is enabled and for all child processes, we need
        # to clear up remote message buffer so that events don't
        # get duplicated on different processes
        self._remote_msgbuf.clear()
        self._remote_future = infinite_time

        while True:
            try:
                cmd = self._local_queues[pid].get()
            except KeyboardInterrupt:
                # we handle the keyboard interrupt here, since Jupyter
                # notebook seems to raise this exception when the
                # kernel is interrupted
                continue
            log.info("[r%d] sync._child_run(pid=%d): recv command %d" %
                     (sync._simulus.comm_rank, pid, cmd))
            if cmd == 0: # run command
                upper, upper_specified = self._local_queues[pid].get()
                self._smp_run(pid, upper, upper_specified)
            elif cmd == 1: # report command
                self._smp_report(pid)
            else: # stop command
                assert cmd == 2
                break

    def _smp_run(self, pid, upper, upper_specified):
        """Dispatch to the protocol-specific run loop.

        Each protocol owns its own worker-process loop because the
        synchronization mechanics differ fundamentally:
          - 'ctw': YAWNS-style synchronous barrier reduce (lockstep)
          - 'cmb': fully asynchronous null-message protocol
          - 'stm': fully asynchronous channel-scanning over a
                   shared-memory ts[] array (the paper's headline).
        """
        if self._protocol == 'cmb':
            self._smp_run_cmb(pid, upper, upper_specified)
        elif self._protocol == 'stm':
            self._smp_run_stm(pid, upper, upper_specified)
        else:
            self._smp_run_lockstep(pid, upper, upper_specified)

    def _smp_run_cmb(self, pid, upper, upper_specified):
        """CMB asynchronous run loop.

        Each worker process owns a subset of LPs and runs them
        independently. An LP advances when the minimum of its input
        channel fronts permits; after advancing it dispatches a null
        message on each output channel carrying its new safe-time
        guarantee. Real messages travel through the same per-channel
        transport (one mp.Queue per pid) and update the destination
        mailbox; channel fronts are updated only by null messages
        (see _cmb_drain_one_message).

        Termination: requires upper to be specified. Each LP marks itself
        done on first reaching upper. When all LPs on this pid are done,
        the pid hits a final mp.Barrier so all pids exit together.
        """
        import queue as _queue_mod  # for queue.Empty

        log.info("[r%d] sync._smp_run_cmb(pid=%d): begins upper=%g, "
                 "upper_specified=%r" %
                 (sync._simulus.comm_rank, pid, upper, upper_specified))

        if not upper_specified:
            raise RuntimeError(
                "CMB protocol requires sync.run(until=...) to be specified")

        run_sims = self._local_partitions[pid]
        self._cmb_my_pid = pid
        self._queue_empty = _queue_mod.Empty
        self._cmb_pending_sends = []
        multi_pid = len(self._local_partitions) > 1
        multi_rank = self._spmd and sync._simulus.comm_size > 1

        # Bootstrap order matters: drain initial messages from
        # _remote_msgbuf BEFORE signaling children to start. Otherwise a
        # child can enter its main loop, advance past the time of an
        # initial event before pid 0 has pushed it, then sched-in-the-
        # past when the late-arriving REAL is finally drained.
        # The drain must run for any pid==0, including multi_pid==False
        # (single pid per rank, SPMD-only); otherwise initial cross-rank
        # messages stay buffered in _remote_msgbuf forever and downstream
        # LPs starve.
        if pid == 0:
            for _rank, msgs in self._remote_msgbuf.items():
                for (until, mb_name, part, msg) in msgs:
                    target_sname, _md, _np, _src = self._all_mboxes[mb_name]
                    target_pid = self._local_pids.get(target_sname)
                    if target_pid == 0:
                        mb = self._local_mboxes[mb_name]
                        mb._sim.sched(mb._mailbox_event, msg, part, until=until)
                    elif target_pid is not None:
                        self._local_data_queues[target_pid].put(
                            ('REAL', '<init>', mb_name, part, msg, until))
                    else:
                        # Cross-rank initial send (only meaningful in SPMD).
                        target_rank = self._all_sims[target_sname]
                        self._cmb_mpi_isend_real(
                            target_rank, '<init>', mb_name, part, msg, until)
            self._remote_msgbuf.clear()
            self._remote_future = infinite_time

        # Now signal children to start their main loops.
        if pid == 0 and multi_pid:
            for s in range(1, len(self._local_partitions)):
                self._local_queues[s].put(0)               # run command
                self._local_queues[s].put((upper, upper_specified))

        # Termination tracking
        upper_reached = {sname: False for sname in run_sims}

        # pid 0 maintains the MPI drain cadence: every K iterations of the
        # main loop, prune completed isends so the request list does not grow.
        prune_counter = 0

        while not all(upper_reached.values()):
            # Phase 1: drain incoming.
            #   - intra-rank: every pid drains its own mp.Queue
            #   - inter-rank: only pid 0 polls MPI; messages are forwarded
            #     to the appropriate local pid's mp.Queue if not for pid 0
            if multi_pid:
                self._cmb_drain_incoming(pid)
            if multi_rank and pid == 0:
                self._cmb_mpi_drain()
                prune_counter += 1
                if prune_counter >= 64:
                    self._cmb_mpi_prune_pending()
                    prune_counter = 0

            # Phase 2: try to advance each LP that's not yet done.
            any_advanced = False
            for sname in run_sims:
                if upper_reached[sname]:
                    continue
                sim = self._local_sims[sname]

                horizon = self._cmb_compute_horizon(sname, upper)
                if horizon > sim.now:
                    self._cmb_advance_lp(sname, horizon)
                    any_advanced = True
                    if sim.now >= upper:
                        upper_reached[sname] = True
                        # Final null dispatch already happened in _cmb_advance_lp.

            # Block on incoming if we made no progress.
            if not any_advanced and not all(upper_reached.values()):
                if multi_pid and not (multi_rank and pid == 0):
                    # Non-pid-0 workers (or pid 0 in non-MPI runs) can block
                    # on the mp.Queue. pid 0 in MPI mode must keep polling
                    # MPI so we sleep briefly instead.
                    self._cmb_wait_for_incoming(pid)
                elif multi_rank and pid == 0:
                    # pid 0 in MPI mode: can't block on mp.Queue (would
                    # starve MPI polling). Tiny sleep + continue loop.
                    time.sleep(0.0001)
                else:
                    # Single-process, no IPC, no progress: zero-aggregate-
                    # lookahead cycle or upper unreachable.
                    log.warning("[r%d] sync._smp_run_cmb(pid=%d): no LP can "
                                "advance and no inter-pid/inter-rank IPC; "
                                "aborting" %
                                (sync._simulus.comm_rank, pid))
                    break

        # Termination phase. Three kinds of in-flight messages can still
        # be in motion:
        #   (a) intra-pid scheduled events — already handled, those LPs
        #       reached upper_reached[sname]
        #   (b) intra-rank queued messages — drain via _cmb_done_count
        #   (c) inter-rank MPI messages — coordinated via Iallreduce + final
        #       MPI_Barrier on pid 0

        # First: intra-rank done counter. Each pid bumps the counter once
        # its local LPs are all done; everyone keeps draining their mp.Queue
        # until the counter reaches n_pids_local.
        if multi_pid:
            n_pids_local = len(self._local_partitions)
            with self._cmb_done_count.get_lock():
                self._cmb_done_count.value += 1
            while self._cmb_done_count.value < n_pids_local:
                self._cmb_drain_incoming(pid)
                if multi_rank and pid == 0:
                    self._cmb_mpi_drain()
                time.sleep(0.0001)
            # All local pids signaled done. Drain residual mp.Queue traffic.
            self._cmb_drain_incoming(pid)
            if multi_rank and pid == 0:
                self._cmb_mpi_drain()

        # Second: cross-rank Iallreduce so all ranks know "everyone done".
        # Only pid 0 of each rank participates in MPI. Non-pid-0 workers
        # keep draining their mp.Queue until pid 0 sends a GLOBAL_DONE
        # signal — pid 0 may forward MPI-incoming messages to those queues
        # right up until it issues GLOBAL_DONE.
        if multi_rank:
            from mpi4py import MPI
            if pid == 0:
                local_done = bytearray([1])
                global_done = bytearray([1])
                req = MPI.COMM_WORLD.Iallreduce(
                    [local_done, MPI.BYTE],
                    [global_done, MPI.BYTE],
                    op=MPI.MIN)
                while not req.Test():
                    self._cmb_mpi_drain()
                    if multi_pid:
                        self._cmb_drain_incoming(pid)
                    time.sleep(0.0001)
                # Tail-drain MPI a few times to catch any final messages
                # that arrived after Iallreduce completed.
                for _ in range(8):
                    self._cmb_mpi_drain()
                    time.sleep(0.0001)
                # MPI barrier so no rank exits while another is still draining.
                MPI.COMM_WORLD.Barrier()
                self._cmb_mpi_drain()
                if self._cmb_pending_sends:
                    MPI.Request.Waitall(self._cmb_pending_sends)
                    self._cmb_pending_sends = []
                # Now signal non-pid-0 workers that global termination is done.
                if multi_pid:
                    for q_pid in range(1, len(self._local_partitions)):
                        self._local_data_queues[q_pid].put(('GLOBAL_DONE',))
            else:
                # Non-pid-0 in multi-rank: keep draining mp.Queue until
                # GLOBAL_DONE arrives. pid 0 forwards MPI traffic to our
                # queue throughout the Iallreduce + Barrier window.
                while True:
                    try:
                        msg = self._local_data_queues[pid].get(timeout=0.001)
                    except self._queue_empty:
                        continue
                    if msg[0] == 'GLOBAL_DONE':
                        break
                    self._cmb_drain_one_message(msg)

        # Final intra-rank barrier so all pids exit together.
        if multi_pid:
            self._cmb_done_barrier.wait()
            self._cmb_drain_incoming(pid)

        self._cmb_my_pid = -1
        log.info("[r%d] sync._smp_run_cmb(pid=%d): ends" %
                 (sync._simulus.comm_rank, pid))

    def _cmb_compute_horizon(self, sname, upper):
        """Return the maximum time this LP can safely advance to:
        min(channel.front for ch in inputs(sname)), bounded by upper."""
        inputs = self._lp_inputs.get(sname, [])
        if not inputs:
            # No inputs -> may advance freely up to upper.
            return upper
        h = min(self._channels[ch_id].front for ch_id in inputs)
        if h > upper:
            h = upper
        return h

    def _cmb_advance_lp(self, sname, horizon):
        """Advance LP sname to horizon, then dispatch nulls on each output
        channel with the new safe-time guarantee.

        Correctness for the SMP transport rests on FIFO ordering of the
        per-pid mp.Queue: any REAL enqueued by sim._run during this
        advance is enqueued *before* the NULL we publish below, so the
        destination drains the REAL (sched at `until`) before applying
        the NULL (which raises ch.front). Since `until <= sim.now` at
        any sim._run send and `sim.now + min_delay >= sim.now`, the
        destination never advances past `until` before scheduling the
        event. SPMD/MPI ordering between REAL and NULL tags is a
        separate concern; see _cmb_mpi_isend_*.
        """
        sim = self._local_sims[sname]
        sim._run(horizon, True)

        for ch_id in self._lp_outputs.get(sname, []):
            ch = self._channels[ch_id]
            self._cmb_publish_safe_time(ch_id, sim.now + ch.min_delay)

    def _cmb_publish_safe_time(self, ch_id, safe_time):
        """Publish a new safe-time guarantee to the destination of ch_id.

        Intra-pid: write directly to channel.front.
        Inter-pid (same rank): enqueue NULL on dst pid's data queue.
        Inter-rank: pid 0 issues MPI isend; other pids forward to pid 0
                    via the MPI_OUT_NULL trampoline.
        """
        src_name, mb_name = ch_id
        ch = self._channels[ch_id]
        dst_pid = self._local_pids.get(ch.dst_name)
        if dst_pid is None:
            # Destination is on a different MPI rank.
            target_rank = self._all_sims[ch.dst_name]
            if self._cmb_my_pid == 0:
                self._cmb_mpi_isend_null(target_rank, src_name, mb_name, safe_time)
            else:
                self._local_data_queues[0].put(
                    ('MPI_OUT_NULL', target_rank, src_name, mb_name, safe_time))
            return
        if dst_pid == self._cmb_my_pid:
            if safe_time > ch.front:
                ch.front = safe_time
        else:
            self._local_data_queues[dst_pid].put(
                ('NULL', src_name, mb_name, safe_time))

    def _cmb_drain_incoming(self, pid):
        """Non-blocking drain of this pid's data queue."""
        q = self._local_data_queues[pid]
        while True:
            try:
                msg = q.get_nowait()
            except self._queue_empty:
                break
            self._cmb_drain_one_message(msg)

    def _cmb_wait_for_incoming(self, pid):
        """Block until at least one message arrives, then return."""
        q = self._local_data_queues[pid]
        msg = q.get()
        self._cmb_drain_one_message(msg)

    def _cmb_drain_one_message(self, msg):
        """Process a single message from the data queue.

        NULL/REAL messages target a local LP; MPI_OUT_* messages are
        cross-rank send requests that non-pid-0 workers forward to pid 0
        via the queue (only pid 0 may call MPI in the SPMD layout).
        """
        kind = msg[0]
        if kind == 'NULL':
            _, src_name, mb_name, safe_time = msg
            ch = self._channels.get((src_name, mb_name))
            if ch is not None and safe_time > ch.front:
                ch.front = safe_time
        elif kind == 'REAL':
            _, src_name, mb_name, part, payload, until = msg
            mb = self._local_mboxes[mb_name]
            mb._sim.sched(mb._mailbox_event, payload, part, until=until)
        elif kind == 'MPI_OUT_NULL':
            _, target_rank, src_name, mb_name, safe_time = msg
            self._cmb_mpi_isend_null(target_rank, src_name, mb_name, safe_time)
        elif kind == 'MPI_OUT_REAL':
            _, target_rank, src_name, mb_name, part, payload, until = msg
            self._cmb_mpi_isend_real(
                target_rank, src_name, mb_name, part, payload, until)
        elif kind == 'GLOBAL_DONE':
            # Sent by pid 0 to non-pid-0 workers in multi-rank termination.
            # Non-pid-0 workers see this and know they can stop draining;
            # the final intra-rank barrier handles the actual rendezvous.
            pass
        else:
            raise RuntimeError("unknown CMB message kind: %r" % (kind,))

    # ---------- MPI transport (pid 0 only) ----------

    def _cmb_mpi_isend_null(self, target_rank, src_name, mb_name, safe_time):
        """Non-blocking MPI send of a NULL message. Queues the request for
        later pruning."""
        from mpi4py import MPI
        payload = (src_name, mb_name, safe_time)
        req = MPI.COMM_WORLD.isend(
            payload, dest=target_rank, tag=_CMB_NULL_TAG)
        self._cmb_pending_sends.append(req)

    def _cmb_mpi_isend_real(self, target_rank, src_name, mb_name,
                             part, msg, until):
        """Non-blocking MPI send of a REAL message."""
        from mpi4py import MPI
        payload = (src_name, mb_name, part, msg, until)
        req = MPI.COMM_WORLD.isend(
            payload, dest=target_rank, tag=_CMB_REAL_TAG)
        self._cmb_pending_sends.append(req)

    def _cmb_mpi_prune_pending(self):
        """Remove completed isend requests so the list does not grow without
        bound. Called periodically by pid 0."""
        if not self._cmb_pending_sends:
            return
        self._cmb_pending_sends = [
            r for r in self._cmb_pending_sends if not r.Test()
        ]

    def _cmb_mpi_drain(self):
        """pid 0 only: drain all pending incoming MPI messages, routing
        NULL/REAL to the appropriate local pid (or applying directly if
        the destination LP is on pid 0). Non-blocking; returns when no
        more incoming messages are pending."""
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        while comm.iprobe(source=MPI.ANY_SOURCE,
                          tag=MPI.ANY_TAG,
                          status=status):
            tag = status.Get_tag()
            source = status.Get_source()
            if tag == _CMB_NULL_TAG:
                payload = comm.recv(source=source, tag=tag)
                src_name, mb_name, safe_time = payload
                target_sname = self._all_mboxes[mb_name][0]
                target_pid = self._local_pids.get(target_sname)
                if target_pid == 0:
                    ch = self._channels.get((src_name, mb_name))
                    if ch is not None and safe_time > ch.front:
                        ch.front = safe_time
                elif target_pid is not None:
                    self._local_data_queues[target_pid].put(
                        ('NULL', src_name, mb_name, safe_time))
                # If target_pid is None we received a stray; drop it.
            elif tag == _CMB_REAL_TAG:
                payload = comm.recv(source=source, tag=tag)
                src_name, mb_name, part, msg, until = payload
                target_sname = self._all_mboxes[mb_name][0]
                target_pid = self._local_pids.get(target_sname)
                if target_pid == 0:
                    mb = self._local_mboxes[mb_name]
                    mb._sim.sched(mb._mailbox_event, msg, part, until=until)
                elif target_pid is not None:
                    self._local_data_queues[target_pid].put(
                        ('REAL', src_name, mb_name, part, msg, until))
            else:
                # Unknown tag — drain via recv to keep buffer clean.
                comm.recv(source=source, tag=tag)
                log.warning("[r%d] CMB: unknown MPI tag %d, dropped" %
                            (sync._simulus.comm_rank, tag))

    def _cmb_route_send(self, sim, mbox_name, msg, part, until):
        """CMB-specific send routing, called from sync.send when
        protocol == 'cmb'. Returns True if handled, False to fall through
        to the legacy CTW send path.
        """
        sname, _min_delay, _nparts, _src = self._all_mboxes[mbox_name]
        target_pid = self._local_pids.get(sname)
        if target_pid is None:
            # Cross-rank.
            target_rank = self._all_sims[sname]
            if self._cmb_my_pid == 0:
                self._cmb_mpi_isend_real(
                    target_rank, sim.name, mbox_name, part, msg, until)
            else:
                self._local_data_queues[0].put(
                    ('MPI_OUT_REAL', target_rank, sim.name, mbox_name,
                     part, msg, until))
            return True
        if target_pid == self._cmb_my_pid:
            mb = self._local_mboxes[mbox_name]
            mb._sim.sched(mb._mailbox_event, msg, part, until=until)
        else:
            self._local_data_queues[target_pid].put(
                ('REAL', sim.name, mbox_name, part, msg, until))
        return True

    # ============================================================
    # STM (Soft-TM) asynchronous channel-scanning protocol.
    #
    # Each LP advances independently to its own safe-time horizon, computed
    # by a read-only transaction over its input channels' ts[] slots in
    # shared memory:
    #     h_i = min over inputs c of ts[c]
    # After advancing, the LP performs a single-write transaction on each
    # output channel's ts[] slot:
    #     ts[c] := max(ts[c], sim.now + min_delay_c)
    # Real messages travel through per-pid mp.Queue (intra-rank); inter-rank
    # transport is left as future work in this stage --- attempting STM
    # under enable_spmd raises a NotImplementedError at the dispatcher.
    # ============================================================

    def _smp_run_stm(self, pid, upper, upper_specified):
        """STM asynchronous channel-scanning run loop.

        Mirrors _smp_run_cmb in structure (intra-rank drain + try-advance
        + done counter + barrier; multi-rank Iallreduce + GLOBAL_DONE).
        The protocol differences are isolated to the helpers:
        _stm_compute_horizon reads input fronts; _stm_publish_safe_time
        publishes via TS_UPDATE on the FIFO transport (intra-rank
        mp.Queue or inter-rank MPI on _STM_TS_TAG). REAL messages share
        the same transport ordering, so any TS_UPDATE on a channel
        arrives at the destination AFTER all REALs sent on that channel
        in the same advance.
        """
        import queue as _queue_mod

        log.info("[r%d] sync._smp_run_stm(pid=%d): begins upper=%g, "
                 "upper_specified=%r" %
                 (sync._simulus.comm_rank, pid, upper, upper_specified))

        if not upper_specified:
            raise RuntimeError(
                "STM protocol requires sync.run(until=...) to be specified")

        run_sims = self._local_partitions[pid]
        self._stm_my_pid = pid
        self._queue_empty = _queue_mod.Empty
        self._stm_pending_sends = []
        # Inter-rank message counters used by the termination
        # quiescence collective. _stm_mpi_sent_count is bumped on each
        # MPI_Isend posting (real or ts); _stm_mpi_recvd_count is
        # bumped on each iprobe-matched recv. Both are pid-0-only;
        # fork makes each pid see its own copy, but only pid 0 ever
        # writes them.
        self._stm_mpi_sent_count = 0
        self._stm_mpi_recvd_count = 0
        multi_pid = len(self._local_partitions) > 1
        multi_rank = self._spmd and sync._simulus.comm_size > 1

        # Drain initial messages from _remote_msgbuf BEFORE signaling
        # children to start (same race fix as _smp_run_cmb): otherwise
        # a child can advance past an initial event before pid 0 has
        # pushed it. The drain must run for any pid==0 (including
        # multi_pid==False, single pid per rank).
        if pid == 0:
            for _rank, msgs in self._remote_msgbuf.items():
                for (until, mb_name, part, msg) in msgs:
                    target_sname, _md, _np, _src = self._all_mboxes[mb_name]
                    target_pid = self._local_pids.get(target_sname)
                    if target_pid == 0:
                        mb = self._local_mboxes[mb_name]
                        mb._sim.sched(mb._mailbox_event, msg, part, until=until)
                    elif target_pid is not None:
                        self._local_data_queues[target_pid].put(
                            ('REAL', '<init>', mb_name, part, msg, until))
                    else:
                        target_rank = self._all_sims[target_sname]
                        self._stm_mpi_isend_real(
                            target_rank, '<init>', mb_name, part, msg, until)
            self._remote_msgbuf.clear()
            self._remote_future = infinite_time

        # Now signal children to start their main loops.
        if pid == 0 and multi_pid:
            for s in range(1, len(self._local_partitions)):
                self._local_queues[s].put(0)               # run command
                self._local_queues[s].put((upper, upper_specified))

        upper_reached = {sname: False for sname in run_sims}
        prune_counter = 0

        while not all(upper_reached.values()):
            # Phase 1: drain incoming.
            if multi_pid:
                self._stm_drain_incoming(pid)
            if multi_rank and pid == 0:
                self._stm_mpi_drain()
                prune_counter += 1
                if prune_counter >= 64:
                    self._stm_mpi_prune_pending()
                    prune_counter = 0

            # Phase 2: try to advance each LP not yet done.
            any_advanced = False
            for sname in run_sims:
                if upper_reached[sname]:
                    continue
                sim = self._local_sims[sname]
                horizon = self._stm_compute_horizon(sname, upper)
                if horizon > sim.now:
                    self._stm_advance_lp(sname, horizon)
                    any_advanced = True
                    if sim.now >= upper:
                        upper_reached[sname] = True

            # Phase 3: no LP advanced this round. STM cannot block on
            # the data queue (a blocked q.get() would miss intra-pid
            # publishes that go through _channel_ts directly), so we
            # poll on a short sleep --- the paper's `retry` semantics
            # under mp.Queue + RawArray. Cost bounded by sleep interval.
            # STM_POLL_SEC env var overrides the default 100us; used by
            # benchmarks/diag_stm_spmd.py to A/B test the polling cost.
            if not any_advanced and not all(upper_reached.values()):
                if multi_pid or multi_rank:
                    time.sleep(_STM_POLL_SEC)
                else:
                    log.warning("[r%d] sync._smp_run_stm(pid=%d): no LP can "
                                "advance and no inter-pid/inter-rank IPC; "
                                "aborting" %
                                (sync._simulus.comm_rank, pid))
                    break

        # Termination phase. Three kinds of in-flight messages can still
        # be in motion (mirroring CMB Stage 4):
        #   (a) intra-pid scheduled events --- already handled
        #   (b) intra-rank queued messages --- drain via _cmb_done_count
        #   (c) inter-rank MPI messages --- Iallreduce + Barrier + GLOBAL_DONE

        # First: intra-rank done counter.
        if multi_pid:
            n_pids_local = len(self._local_partitions)
            with self._cmb_done_count.get_lock():
                self._cmb_done_count.value += 1
            while self._cmb_done_count.value < n_pids_local:
                self._stm_drain_incoming(pid)
                if multi_rank and pid == 0:
                    self._stm_mpi_drain()
                time.sleep(0.0001)
            self._stm_drain_incoming(pid)
            if multi_rank and pid == 0:
                self._stm_mpi_drain()

        # Second: cross-rank termination handshake.
        #
        # Phase A: Iallreduce(local_done, MIN) -- every rank confirms it
        #          has exited the main loop and the intra-rank done
        #          counter (so no rank is still generating new MPI
        #          sends).
        #
        # Phase B (the "final round of collective"): quiescence loop.
        #          Each rank tracks the number of inter-rank MPI
        #          messages it has issued (_stm_mpi_sent_count) and
        #          drained (_stm_mpi_recvd_count). We drain once, then
        #          Iallreduce the (sent, recvd) pair. When global_sent
        #          == global_recvd, every cross-rank message ever
        #          issued has been consumed at its destination and
        #          there are no in-flight messages anywhere.
        #          Convergence: in the termination phase neither
        #          counter can decrease and sent_count is fixed (no
        #          new sends), so global_recvd monotonically rises to
        #          global_sent in O(1) iterations under any MPI
        #          implementation. This is more robust than relying on
        #          Waitall + Barrier ordering, which depends on
        #          implementation-specific isend completion semantics.
        if multi_rank:
            from mpi4py import MPI
            import array
            if pid == 0:
                local_done = bytearray([1])
                global_done = bytearray([1])
                req = MPI.COMM_WORLD.Iallreduce(
                    [local_done, MPI.BYTE],
                    [global_done, MPI.BYTE],
                    op=MPI.MIN)
                while not req.Test():
                    self._stm_mpi_drain()
                    if multi_pid:
                        self._stm_drain_incoming(pid)
                    time.sleep(0.0001)

                # Phase B: drain to global quiescence.
                while True:
                    self._stm_mpi_drain()
                    if multi_pid:
                        self._stm_drain_incoming(pid)
                    local_sr = array.array(
                        'Q',
                        [self._stm_mpi_sent_count,
                         self._stm_mpi_recvd_count])
                    global_sr = array.array('Q', [0, 0])
                    req2 = MPI.COMM_WORLD.Iallreduce(
                        [local_sr, MPI.UINT64_T],
                        [global_sr, MPI.UINT64_T],
                        op=MPI.SUM)
                    while not req2.Test():
                        self._stm_mpi_drain()
                        if multi_pid:
                            self._stm_drain_incoming(pid)
                        time.sleep(0.0001)
                    if global_sr[0] == global_sr[1]:
                        break

                # Pending isends are guaranteed matched at this point
                # (every send was counted, every recv was counted, and
                # the counters agree globally). Waitall is a paranoia
                # cleanup of the request list; should be effectively a
                # no-op.
                if self._stm_pending_sends:
                    MPI.Request.Waitall(self._stm_pending_sends)
                    self._stm_pending_sends = []
                if multi_pid:
                    for q_pid in range(1, len(self._local_partitions)):
                        self._local_data_queues[q_pid].put(('GLOBAL_DONE',))
            else:
                # Non-pid-0 in multi-rank: keep draining mp.Queue until
                # GLOBAL_DONE arrives. (pid 0's quiescence loop still
                # forwards inter-rank arrivals to non-zero pids via
                # this queue, so we must keep draining throughout.)
                while True:
                    try:
                        msg = self._local_data_queues[pid].get(timeout=0.001)
                    except self._queue_empty:
                        continue
                    if msg[0] == 'GLOBAL_DONE':
                        break
                    self._stm_drain_one_message(msg)

        # Final intra-rank barrier.
        if multi_pid:
            self._cmb_done_barrier.wait()
            self._stm_drain_incoming(pid)

        self._stm_my_pid = -1
        log.info("[r%d] sync._smp_run_stm(pid=%d): ends" %
                 (sync._simulus.comm_rank, pid))

    def _stm_compute_horizon(self, sname, upper):
        """STM advance check (paper Algorithm 1): atomic snapshot of all
        input channel timestamps from shared memory.

        Correctness uses per-channel send/drain counters. When the
        check finds send_count[c] != drain_count[c] for any input c,
        a REAL is in flight on c -- either truly queued, or visible
        as send_count++ but not yet flushed by mp.Queue's feeder
        thread. Either way we drain and retry. The retry is bounded:
        each iteration's drain monotonically raises drain_count, so
        the loop exits as soon as the feeder has caught up. When the
        check passes, every REAL ever sent on every input channel is
        scheduled in this LP's heap, so advancing to min(ts[c]) is
        safe (events fire in time order during sim._run).
        """
        inputs = self._lp_inputs.get(sname, [])
        if not inputs:
            return upper
        ts_arr = self._channel_ts
        send_arr = self._stm_send_count
        drain_arr = self._stm_drain_count
        while True:
            h = upper
            for ch_id in inputs:
                slot = self._channels[ch_id].ts_idx
                t = ts_arr[slot]
                if t < h:
                    h = t
            any_pending = False
            for ch_id in inputs:
                slot = self._channels[ch_id].ts_idx
                if send_arr[slot] != drain_arr[slot]:
                    any_pending = True
                    break
            if not any_pending:
                return h
            self._stm_drain_incoming(self._stm_my_pid)

    def _stm_advance_lp(self, sname, horizon):
        """Advance LP sname to horizon; then publish new safe times on
        each output channel.

        Correctness rests on the per-channel send/drain counter
        consistency check in _stm_compute_horizon (see its docstring).
        Each REAL queued during sim._run bumps send_count[c]; the
        destination's snapshot-with-retry ensures it cannot read a
        stale ts[] without first seeing the matching drain_count++.
        """
        sim = self._local_sims[sname]
        sim._run(horizon, True)
        end_now = sim.now
        for ch_id in self._lp_outputs.get(sname, []):
            ch = self._channels[ch_id]
            self._stm_publish_safe_time(ch_id, end_now + ch.min_delay)

    def _stm_publish_safe_time(self, ch_id, safe_time):
        """Publish a new safe-time guarantee to the destination of ch_id.

        v3: intra-rank publish is a single shared-memory write. There
        is no per-channel queue message accompanying it (compare CMB,
        which dispatches a NULL on every output channel per advance).
        This is the structural change that makes STM's per-advance
        publish cost O(N_out) memory writes instead of O(N_out) queue
        puts --- and the corresponding read O(N_in) memory loads
        instead of O(N_in) queue gets. The cost-model crossover at
        high fan-in (paper Section 5) is realized here.

        Inter-rank: shared memory does not span ranks, so we still
        issue MPI TS_UPDATE messages. pid 0 issues the isend directly;
        other pids trampoline via pid 0's queue.
        """
        ch = self._channels[ch_id]
        slot = ch.ts_idx
        ts_arr = self._channel_ts
        # Single shared-memory write. Monotonic guard: only raise the
        # slot, never lower it. Source pid is the sole writer of this
        # slot (channel is directed: one source LP -> one ts[] slot).
        if safe_time > ts_arr[slot]:
            ts_arr[slot] = safe_time
        dst_pid = self._local_pids.get(ch.dst_name)
        if dst_pid is None:
            # Inter-rank: still need an MPI message (no shared memory
            # across nodes).
            target_rank = self._all_sims[ch.dst_name]
            if self._stm_my_pid == 0:
                self._stm_mpi_isend_ts(target_rank, ch_id, safe_time)
            else:
                self._local_data_queues[0].put(
                    ('MPI_OUT_TS_UPDATE', target_rank, ch_id, safe_time))

    def _stm_drain_incoming(self, pid):
        """Non-blocking drain of this pid's mp.Queue for REAL messages."""
        q = self._local_data_queues[pid]
        while True:
            try:
                msg = q.get_nowait()
            except self._queue_empty:
                break
            self._stm_drain_one_message(msg)

    def _stm_wait_for_incoming(self, pid):
        """Brief poll for a REAL or any ts[] update.

        v3 sends intra-rank ts updates through shared memory, not the
        data queue --- so a blocking q.get() would miss them. We poll
        the queue with a tiny timeout, returning either when a REAL
        arrives (drained immediately) or when the timeout expires
        (the outer loop then re-snapshots ts[] for new fronts).
        """
        try:
            msg = self._local_data_queues[pid].get(timeout=0.0001)
            self._stm_drain_one_message(msg)
        except self._queue_empty:
            pass

    def _stm_drain_one_message(self, msg):
        """Process a single message from the data queue.

        Intra-rank kinds:
          REAL       a forwarded inter-LP message (-> sched, drain++)
          TS_UPDATE  cross-rank ts publication forwarded by pid 0 to
                     a non-pid-0 worker; v3 does not use TS_UPDATE
                     intra-rank (those go via shared memory).
        MPI trampoline kinds (non-pid-0 pushes; pid 0 fires the MPI):
          MPI_OUT_REAL       -> _stm_mpi_isend_real
          MPI_OUT_TS_UPDATE  -> _stm_mpi_isend_ts
          GLOBAL_DONE        sentinel from pid 0 closing termination

        v3 correctness rests on (1) shared-memory ts[] reads in
        _stm_compute_horizon and (2) the per-channel sent-min cap on
        ts[c] in _stm_advance_lp. The cap guarantees ts[c] is no
        higher than the smallest in-flight REAL.until on c, so
        destinations advancing to ts[c] cannot pass any in-flight
        message --- the eventual drain always sched(until) with
        until >= sim.now.
        """
        kind = msg[0]
        if kind == 'REAL':
            _, src_name, mb_name, part, payload, until = msg
            mb = self._local_mboxes[mb_name]
            mb._sim.sched(mb._mailbox_event, payload, part, until=until)
            ch_id = (src_name, mb_name)
            ch = self._channels.get(ch_id)
            if ch is not None:
                # Only bump drain_count for intra-rank inter-pid REALs.
                # _stm_route_send bumps send_count[slot] only on the
                # intra-rank inter-pid path; inter-rank sends skip it
                # (and the bootstrap '<init>' inserts skip it too).
                # Bumping drain_count here for an inter-rank-delivered
                # REAL would leave send_count==0 != drain_count>0 and
                # cause _stm_compute_horizon's consistency check to
                # spin forever on every input from a remote rank.
                if src_name in self._local_pids:
                    self._stm_drain_count[ch.ts_idx] += 1
        elif kind == 'TS_UPDATE':
            # Reached only via MPI trampoline (pid 0 forwarding to a
            # non-pid-0 worker after _stm_mpi_drain receives a TS_TAG).
            # Apply directly to shared memory.
            _, ch_id, new_safe = msg
            ch = self._channels.get(ch_id)
            if ch is not None:
                slot = ch.ts_idx
                if new_safe > self._channel_ts[slot]:
                    self._channel_ts[slot] = new_safe
        elif kind == 'MPI_OUT_REAL':
            _, target_rank, src_name, mb_name, part, payload, until = msg
            self._stm_mpi_isend_real(
                target_rank, src_name, mb_name, part, payload, until)
        elif kind == 'MPI_OUT_TS_UPDATE':
            _, target_rank, ch_id, safe_time = msg
            self._stm_mpi_isend_ts(target_rank, ch_id, safe_time)
        elif kind == 'GLOBAL_DONE':
            # Sent by pid 0 to non-pid-0 workers in multi-rank
            # termination; the post-Iallreduce barrier handles
            # the actual rendezvous, this just exits the wait loop.
            pass
        else:
            raise RuntimeError("unknown STM message kind: %r" % (kind,))

    # ---------- MPI transport (pid 0 only) ----------

    def _stm_mpi_isend_ts(self, target_rank, ch_id, safe_time):
        """Non-blocking MPI send of a TS_UPDATE. Mirrors
        _cmb_mpi_isend_null but on _STM_TS_TAG and with channel-id
        payload."""
        from mpi4py import MPI
        payload = (ch_id, safe_time)
        req = MPI.COMM_WORLD.isend(
            payload, dest=target_rank, tag=_STM_TS_TAG)
        self._stm_pending_sends.append(req)
        self._stm_mpi_sent_count += 1

    def _stm_mpi_isend_real(self, target_rank, src_name, mb_name,
                             part, msg, until):
        """Non-blocking MPI send of a REAL message. Mirrors
        _cmb_mpi_isend_real but on _STM_REAL_TAG."""
        from mpi4py import MPI
        payload = (src_name, mb_name, part, msg, until)
        req = MPI.COMM_WORLD.isend(
            payload, dest=target_rank, tag=_STM_REAL_TAG)
        self._stm_pending_sends.append(req)
        self._stm_mpi_sent_count += 1

    def _stm_mpi_prune_pending(self):
        """Periodic prune of completed isend requests. pid 0 only."""
        if not self._stm_pending_sends:
            return
        self._stm_pending_sends = [
            r for r in self._stm_pending_sends if not r.Test()
        ]

    def _stm_mpi_drain(self):
        """pid 0 only: drain all pending incoming MPI messages, routing
        TS_UPDATE / REAL to the appropriate local pid (or applying
        directly if the destination LP is on pid 0). Non-blocking;
        returns when no more incoming messages are pending."""
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        while comm.iprobe(source=MPI.ANY_SOURCE,
                          tag=MPI.ANY_TAG,
                          status=status):
            tag = status.Get_tag()
            source = status.Get_source()
            # Count one inter-rank message received per iprobe match.
            # Used by the termination quiescence collective to detect
            # when global_sent == global_recvd (= no in-flight messages).
            self._stm_mpi_recvd_count += 1
            if tag == _STM_TS_TAG:
                payload = comm.recv(source=source, tag=tag)
                ch_id, safe_time = payload
                ch = self._channels.get(ch_id)
                if ch is None:
                    continue
                target_pid = self._local_pids.get(ch.dst_name)
                if target_pid == 0:
                    if safe_time > ch.front:
                        ch.front = safe_time
                    slot = ch.ts_idx
                    if safe_time > self._channel_ts[slot]:
                        self._channel_ts[slot] = safe_time
                elif target_pid is not None:
                    self._local_data_queues[target_pid].put(
                        ('TS_UPDATE', ch_id, safe_time))
            elif tag == _STM_REAL_TAG:
                payload = comm.recv(source=source, tag=tag)
                src_name, mb_name, part, msg, until = payload
                target_sname = self._all_mboxes[mb_name][0]
                target_pid = self._local_pids.get(target_sname)
                if target_pid == 0:
                    mb = self._local_mboxes[mb_name]
                    mb._sim.sched(mb._mailbox_event, msg, part, until=until)
                elif target_pid is not None:
                    self._local_data_queues[target_pid].put(
                        ('REAL', src_name, mb_name, part, msg, until))
            else:
                comm.recv(source=source, tag=tag)
                log.warning("[r%d] STM: unknown MPI tag %d, dropped" %
                            (sync._simulus.comm_rank, tag))

    def _stm_route_send(self, sim, mbox_name, msg, part, until):
        """STM-specific send routing. Mirrors _cmb_route_send.

        Routing tiers:
          intra-pid: schedule directly on receiver's mailbox
          inter-pid (same rank): mp.Queue on target pid + send_count++
                                 (consistency-check signal for the
                                 destination's snapshot retry)
          inter-rank: pid 0 issues MPI isend; other pids forward via
                      the MPI_OUT_REAL trampoline on pid 0's queue
        """
        sname, _min_delay, _nparts, _src = self._all_mboxes[mbox_name]
        target_pid = self._local_pids.get(sname)
        if target_pid is None:
            target_rank = self._all_sims[sname]
            if self._stm_my_pid == 0:
                self._stm_mpi_isend_real(
                    target_rank, sim.name, mbox_name, part, msg, until)
            else:
                self._local_data_queues[0].put(
                    ('MPI_OUT_REAL', target_rank, sim.name, mbox_name,
                     part, msg, until))
            return True
        if target_pid == self._stm_my_pid:
            mb = self._local_mboxes[mbox_name]
            mb._sim.sched(mb._mailbox_event, msg, part, until=until)
        else:
            self._local_data_queues[target_pid].put(
                ('REAL', sim.name, mbox_name, part, msg, until))
            # Increment send_count AFTER queue.put. Source pid is the
            # only writer for this channel's send_count, so a plain
            # RawArray write suffices. Destination's retry loop reads
            # send_count and waits until drain_count catches up.
            ch_id = (sim.name, mbox_name)
            ch = self._channels.get(ch_id)
            if ch is not None:
                self._stm_send_count[ch.ts_idx] += 1
        return True

    def _smp_run_lockstep(self, pid, upper, upper_specified):
        """Lockstep run loop, used by 'ctw' (YAWNS).

        The horizon-reduce step uses an mp.Queue allreduce intra-rank
        plus an MPI allreduce on rank 0 in SPMD mode."""

        log.info("[r%d] sync._smp_run(pid=%d): begins with upper=%g, upper_specified=%r" %
                 (sync._simulus.comm_rank, pid, upper, upper_specified))
        run_sims = self._local_partitions[pid]
                
        if pid == 0:
            for s in range(1, len(self._local_partitions)):
                self._local_queues[s].put(0) # run command
                self._local_queues[s].put((upper, upper_specified))
        
        while True:
            # figure out the start time of the next window (a.k.a.,
            # lower bound on timestamp): it's the minimum of three
            # values: (1) the timestamp of the first event plus the
            # lookahead, (2) the smallest timestamp of messages to be
            # sent to a remote simulator, and (3) the upper time
            horizon = infinite_time
            for s in run_sims:
                t = self._local_sims[s].peek()
                if horizon > t: horizon = t
            if horizon < infinite_time:
                horizon += self._lookahead
            if horizon > self._remote_future:
                horizon = self._remote_future
            if horizon > upper:
                horizon = upper

            # YAWNS allreduce: mp.Queue intra-rank, MPI on rank 0.
            if len(self._local_partitions) > 1:
                if pid > 0:
                    self._local_queues[0].put(horizon)
                else:
                    for s in range(1, len(self._local_partitions)):
                        x = self._local_queues[0].get()
                        if x < horizon: horizon = x
            if self._spmd and pid == 0:
                horizon = sync._simulus.allreduce(horizon, min)
            if len(self._local_partitions) > 1:
                if pid > 0:
                    horizon = self._local_queues[pid].get()
                else:
                    for s in range(1, len(self._local_partitions)):
                        self._local_queues[s].put(horizon)
            #log.debug("[r%d] sync._run(pid='%d'): sync window [%g:%g]" %
            #          (sync._simulus.comm_rank, pid, self.now, horizon))

            # if there's no more event anywhere, and the upper was not
            # specified, it means we can simply stop by now, the
            # previous iteration should have updated the current time
            # to the horizon for the last event
            if horizon == infinite_time and upper_specified == 0:
                break

            # bring all local simulators' time to horizon
            for s in run_sims:
                #log.debug("[r%d] sync._run(): simulator '%s' execute [%g:%g]" %
                #          (sync._simulus.comm_rank, s[-4:], self._local_sims[s].now, horizon))
                self._local_sims[s]._run(horizon, True)
            self.now = horizon

            # distribute remote messages:
            
            # first, gather remote messages from processes
            if len(self._local_partitions) > 1:
                if pid > 0:
                    #log.debug("[r%d] sync._run(pid=%d): put %r to pid 0" %
                    #          (sync._simulus.comm_rank, pid, self._remote_msgbuf))
                    self._local_queues[0].put(self._remote_msgbuf)
                else:
                    for s in range(1, len(self._local_partitions)):
                        x = self._local_queues[0].get()
                        #log.debug("[r%d] sync._run(pid=0): get %r" %
                        #          (sync._simulus.comm_rank, x))
                        for r in x.keys():
                            self._remote_msgbuf[r].extend(x[r])
                        
            # second, distribute via all to all
            if pid == 0:
                if self._spmd:
                    incoming = sync._simulus.alltoall(self._remote_msgbuf)
                else:
                    incoming = self._remote_msgbuf[0]
                #log.debug("[r%d] sync._run(pid=0): all-to-all incoming=%r" %
                #          (sync._simulus.comm_rank, incoming))
            
            # third, scatter messages to target processes
            if len(self._local_partitions) > 1:
                if pid > 0:
                    incoming = self._local_queues[pid].get()
                    #log.debug("[r%d] sync._run(pid=%d): get %r" %
                    #              (sync._simulus.comm_rank, pid, incoming))
                else:
                    pmsgs = defaultdict(list)
                    if incoming is not None:
                        for m in incoming:
                            _, mbname, *_ = m # find destination mailbox name
                            s, *_ = self._all_mboxes[mbname] # find destination simulator name
                            # find destination pid
                            pmsgs[self._local_pids[s]].append(m)
                    for s in range(1, len(self._local_partitions)):
                        self._local_queues[s].put(pmsgs[s])
                        #log.debug("[r%d] sync._run(pid=%d): put %r to pid %d" %
                        #          (sync._simulus.comm_rank, pid, pmsgs[s], s))
                    incoming = pmsgs[0]
                    #log.debug("[r%d] sync._run(pid=%d): keep %r" %
                    #          (sync._simulus.comm_rank, pid, incoming))
            
            if incoming is not None:
                for until, mbname, part, msg in incoming:
                    mbox = self._local_mboxes[mbname]
                    mbox._sim.sched(mbox._mailbox_event, msg, part, until=until)

            # now we can remove the old messages and get ready for next window
            self._remote_msgbuf.clear()
            self._remote_future = infinite_time

            if horizon >= upper: break

        log.info("[r%d] sync._smp_run(pid=%d): finishes with upper=%g, upper_specified=%r" %
                 (sync._simulus.comm_rank, pid, upper, upper_specified))

    def _run_finish(self):
        log.info("[r%d] sync._run_finish() at exit" % sync._simulus.comm_rank)

        if self._simulus.comm_rank ==0:
            self._simulus.bcast(2) # stop command

        if len(self._local_partitions) > 1:
            for s in range(1, len(self._local_partitions)):
                self._local_queues[s].put(2) # stop command
            for p in self._child_procs: p.join()

    def send(self, sim, mbox_name, msg, delay=None, part=0):
        """Send a messsage from a simulator to a named mailbox.

        Args:
            sim (simulator): the simulator from which the message will
                be sent

            name (str): the name of the mailbox to which the message
                is expected to be delivered

            msg (object): a message can be any Python object; however,
                a message needs to be pickle-able as it may be
                transferred between different simulators located on
                separate processes (with different Python interpreter)
                or even on different machines; a message also cannot
                be None
        
            delay (float): the delay with which the message is
                expected to be delivered to the mailbox; if it is
                ignored, the delay will be set to be the min_delay of
                the mailbox; if it is set, the delay value must not be
                smaller than the min_delay of the mailbox
        
            part (int): the partition number of the mailbox to which
                the message will be delivered; the default is zero

        Returns:
            This method returns nothing (as opposed to the mailbox
            send() method); once sent, it's sent, as it cannot be
            cancelled or rescheduled.

        """

        if not self._activated:
            errmsg = "sync.send() called before the synchronized is created"
            log.error(errmsg)
            raise RuntimeError(errmsg)
        if sim is None or not isinstance(sim, simulator):
            errmsg = "sync.send(sim=%r) requires a simulator" % sim
            log.error(errmsg)
            raise ValueError(errmsg)
        if sim.name not in self._local_sims:
            errmsg = "sync.send(sim='%s') simulator not in synchronized group" % sim.name
            log.error(errmsg)
            raise ValueError(errmsg)
        if msg is None:
            errmsg = "sync.send() message cannot be None"
            log.error(errmsg)
            raise ValueError(errmsg)

        if mbox_name in self._all_mboxes:
            sname, min_delay, nparts, _src = self._all_mboxes[mbox_name]
            if delay is None:
                delay = min_delay
            elif delay < min_delay:
                errmsg = "sync.send() delay (%g) less than min_delay (%r)" % \
                         (delay, min_delay)
                log.error(errmsg)
                raise ValueError(errmsg)
            if part < 0 or part >= nparts:
                errmsg = "sync.send(part=%r) out of range (target mailbox '%s' has %d partitions)" % \
                         (part, mbox_name, nparts)
                log.error(errmsg)
                raise IndexError(errmsg)

            # Asynchronous-protocol send: route through the per-pid data
            # queue immediately, rather than buffering for window-boundary
            # distribution as in CTW. The _cmb_my_pid / _stm_my_pid guard
            # is non-negative only while the corresponding async run loop
            # is active --- pre-run() g.send() falls through to the
            # legacy CTW path, which buffers into _remote_msgbuf for the
            # async loops to drain on entry.
            if self._protocol == 'cmb' and self._cmb_my_pid >= 0:
                until = sim.now + delay
                self._cmb_route_send(sim, mbox_name, msg, part, until)
                return
            if self._protocol == 'stm' and self._stm_my_pid >= 0:
                until = sim.now + delay
                self._stm_route_send(sim, mbox_name, msg, part, until)
                return

            # if it's local delivery, send to the target mailbox
            # directly; a local delivery can be one of the two cases:
            # 1) if SMP is disabled (that is, all local simulators are
            # executed on the same process), the target mailbox
            # belongs to one of the local simulators; or 2) if SMP is
            # enabled (that is, all local simulators are executed on
            # separate processes), the target mailbox belongs to the
            # same sender simulator
            if not self._smp and mbox_name in self._local_mboxes or \
               mbox_name in sim._mailboxes:
                mbox = self._local_mboxes[mbox_name]
                until = sim.now+delay
                mbox._sim.sched(mbox._mailbox_event, msg, part, until=until)
                #log.debug("[r%d] sync.send(sim='%s') to local mailbox '%s': msg=%r, delay=%g (until=%g), part=%d" %
                #          (sync._simulus.comm_rank, sim.name[-4:], mbox_name, msg, delay, until, part))
            else:
                until = sim.now+delay
                self._remote_msgbuf[self._all_sims[sname]].append((until, mbox_name, part, msg))
                if self._remote_future > until:
                    self._remote_future = until
                #log.debug("[r%d] sync.send(sim='%s') to remote mailbox '%s' on simulator '%s': msg=%r, delay=%g, part=%d" %
                #          (sync._simulus.comm_rank, sim.name[-4:], mbox_name, sname[-4:], msg, delay, part))
        else:
            errmsg = "sync.send() to mailbox named '%s' not found" % mbox_name
            log.error(errmsg)
            raise ValueError(errmsg)

    @classmethod
    def comm_rank(cls):
        """Return the process rank of this simulus instance."""
        
        # the simulus instance is a class variable
        if not sync._simulus:
            sync._simulus = _Simulus()
        return sync._simulus.comm_rank

    @classmethod
    def comm_size(cls):
        """Return the total number processes."""
        
        # the simulus instance is a class variable
        if not sync._simulus:
            sync._simulus = _Simulus()
        return sync._simulus.comm_size

    def show_runtime_report(self, show_partition=True, prefix=''):
        """Print a report on the runtime performance of running the
        synchronized group. 

        Args:
            show_partition (bool): if it's True (the default), the
                print-out report also contains the processor
                assignment of the simulators

            prefix (str): all print-out lines will be prefixed by this
                string (the default is empty); this would help if one
                wants to find the report in a large amount of output

        """

        if self._spmd and sync._simulus.comm_rank > 0:
            errmsg = "sync.show_runtime_report() allowed only on rank 0"
            log.error(errmsg)
            raise RuntimeError(errmsg)

        if self._local_partitions is None:
            errmsg = "sync.show_runtime_report() called before sync.run()"
            log.error(errmsg)
            raise ValueError(errmsg)

        if self._spmd:
            cmd = sync._simulus.bcast(1) # report command
        self._smp_report(0, show_partition, prefix)

    def _smp_report(self, pid, show_partition=None, prefix=None):
        """Collect statistics and report them."""

        if pid == 0:
            for s in range(1, len(self._local_partitions)):
                self._local_queues[s].put(1) # report command
        
        t1 = time.time()
        run_sims = self._local_partitions[pid]
        sync_rt = {
            "start_clock": time.time(),
            "sims": self._local_pids.copy(),
            "scheduled_events": 0,
            "cancelled_events": 0,
            "executed_events": 0,
            "initiated_processes": 0,
            "cancelled_processes": 0,
            "process_contexts": 0,
            "terminated_processes": 0,
        }
        for s in run_sims:
            rt = self._local_sims[s]._runtime
            if rt["start_clock"] < sync_rt["start_clock"]:
                sync_rt["start_clock"] = rt["start_clock"]
            sync_rt["scheduled_events"] += rt["scheduled_events"]
            sync_rt["cancelled_events"] += rt["cancelled_events"]
            sync_rt["executed_events"] += rt["executed_events"]
            sync_rt["initiated_processes"] += rt["initiated_processes"]
            sync_rt["cancelled_processes"] += rt["cancelled_processes"]
            sync_rt["process_contexts"] += rt["process_contexts"]
            sync_rt["terminated_processes"] += rt["terminated_processes"]
            
        if len(self._local_partitions) > 1:
            if pid > 0:
                self._local_queues[0].put(sync_rt)
            else:
                for s in range(1, len(self._local_partitions)):
                    rt = self._local_queues[0].get()
                    if rt["start_clock"] < sync_rt["start_clock"]:
                        sync_rt["start_clock"] = rt["start_clock"]
                    sync_rt["scheduled_events"] += rt["scheduled_events"]
                    sync_rt["cancelled_events"] += rt["cancelled_events"]
                    sync_rt["executed_events"] += rt["executed_events"]
                    sync_rt["initiated_processes"] += rt["initiated_processes"]
                    sync_rt["cancelled_processes"] += rt["cancelled_processes"]
                    sync_rt["process_contexts"] += rt["process_contexts"]
                    sync_rt["terminated_processes"] += rt["terminated_processes"]

        if pid == 0 and self._spmd:
            all_rts = sync._simulus.gather(sync_rt)
            if self._simulus.comm_rank == 0:
                sync_rt = all_rts[0]
                for rt in all_rts[1:]:
                    if rt["start_clock"] < sync_rt["start_clock"]:
                        sync_rt["start_clock"] = rt["start_clock"]
                    sync_rt["sims"].update(rt["sims"])
                    sync_rt["scheduled_events"] += rt["scheduled_events"]
                    sync_rt["cancelled_events"] += rt["cancelled_events"]
                    sync_rt["executed_events"] += rt["executed_events"]
                    sync_rt["initiated_processes"] += rt["initiated_processes"]
                    sync_rt["cancelled_processes"] += rt["cancelled_processes"]
                    sync_rt["process_contexts"] += rt["process_contexts"]
                    sync_rt["terminated_processes"] += rt["terminated_processes"]

        if pid == 0 and self._simulus.comm_rank == 0:
            print('%s*********** sync group performance metrics ***********' % prefix)
            if show_partition:
                print('%spartitioning information (simulator assignment):' % prefix)
                for sname, simrank in self._all_sims.items():
                    print("%s  '%s' on rank %d proc %d" % (prefix, sname, simrank, sync_rt["sims"][sname]))
            t = t1-sync_rt["start_clock"]
            print('%sexecution time: %g' % (prefix,t))
            print('%sscheduled events: %d (rate=%g)' %
                  (prefix, sync_rt["scheduled_events"], sync_rt["scheduled_events"]/t))
            print('%sexecuted events: %d (rate=%g)' %
                  (prefix, sync_rt["executed_events"], sync_rt["executed_events"]/t))
            print('%scancelled events: %d' % (prefix, sync_rt["cancelled_events"]))
            print('%screated processes: %d' % (prefix, sync_rt["initiated_processes"]))
            print('%sfinished processes: %d' % (prefix, sync_rt["terminated_processes"]))
            print('%scancelled processes: %d' % (prefix, sync_rt["cancelled_processes"]))
            print('%sprocess context switches: %d' % (prefix, sync_rt["process_contexts"]))
