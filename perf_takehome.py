"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

import random
import unittest
from ast import Add
from collections import defaultdict
from typing import Optional, TypeAlias

from problem import (
    HASH_STAGES,
    N_CORES,
    SCRATCH_SIZE,
    SLOT_LIMITS,
    VLEN,
    DebugInfo,
    Engine,
    Input,
    Instruction,
    Machine,
    Slot,
    Tree,
    build_mem_image,
    my_reference_kernel,
    reference_kernel,
)

Address: TypeAlias = int


def vec_at(base_addr: Address, idx: int) -> Address:
    return base_addr + idx


class KernelBuilder:
    def __init__(self):
        self.instrs: list[Instruction] = []
        self.scratch: dict[str, Address] = {}
        self.scratch_debug: dict[Address, tuple[str, int]] = {}
        self.scratch_ptr: Address = 0
        self.const_map: dict[int, Address] = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def bundle(self, instr: Instruction):
        return self.instrs.append(instr)

    def add_single(self, engine: Engine, slot: Slot):
        self.instrs.append({engine: [slot]})

    def add(self, engine: Engine, slots: list[Slot]):
        self.instrs.append({engine: slots})

    def alloc_scratch(self, name: str | None = None, length: int = 1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_vec(self, name: str | None = None):
        return self.alloc_scratch(name, 8)

    def scratch_const(self, val: int, name: str | None = None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add_single("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def vectorize_hash(self, hash_stages: list[tuple[str, int, str, str, int]]):
        vectorized_hash_stages: list[tuple[str, Address, str, str, Address]] = []

        for i, (op1, a, op3, op2, b) in enumerate(hash_stages):
            v_a = self.alloc_vec(f"hash_a_{i}")
            v_b = self.alloc_vec(f"hash_b_{i}")
            vectorized_hash_stages.append((op1, v_a, op3, op2, v_b))
            self.add(
                "valu",
                [
                    ("vbroadcast", v_a, self.scratch_const(a)),
                    ("vbroadcast", v_b, self.scratch_const(b)),
                ],
            )

        return vectorized_hash_stages

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(
                ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
            )

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Only using simd, no parallel slots
        """

        # Intermediate variables we'll need
        v_curr_tree_vals = self.alloc_scratch("v_curr_tree_vals", length=n_nodes)
        v_curr_node_vals = self.alloc_scratch("v_curr_node_vals", length=n_nodes)
        v_curr_idx = self.alloc_scratch("v_curr_idx", length=n_nodes)
        v_tmp1 = self.alloc_scratch("v_tmp1", length=n_nodes)
        v_tmp2 = self.alloc_scratch("v_tmp2", length=n_nodes)

        # input is only 7 variables, but allocate 8 so vload doesn't stomp on 8th entry
        v_mem_input = self.alloc_vec("v_mem_input")
        mem_tree_vals = v_mem_input + 4
        v_mem_tree_vals = self.alloc_vec("v_mem_tree_vals")
        mem_node_vals = v_mem_input + 6

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        v_two_const = self.alloc_vec("v_two_const")
        v_hash_stages = self.vectorize_hash(HASH_STAGES)

        # Initialize variables
        self.add_single("load", ("vload", v_mem_input, zero_const))
        self.add_single(
            "alu", ("-", mem_tree_vals, mem_tree_vals, one_const)
        )  # So we can 1-index
        self.add(
            "valu",
            [
                ("vbroadcast", v_two_const, two_const),
                ("vbroadcast", v_mem_tree_vals, mem_tree_vals),
            ],
        )

        self.add_single("flow", ("pause",))
        assert batch_size % 8 == 0, "Batch size not in even chunks of 8"

        # Initialize v_curr_idx and v_curr_node_vals
        self.add_single(
            "valu", ("vbroadcast", v_tmp1, mem_node_vals)
        )  # Use as temp idx
        for offset in range(0, batch_size, 8):
            self.bundle(
                {
                    "valu": [("vbroadcast", vec_at(v_curr_idx, offset), one_const)],
                    "load": [("vload", vec_at(v_curr_node_vals, offset), v_tmp1)],
                    "flow": [("add_imm", v_tmp1, v_tmp1, 8)],
                }
            )

        for batch in range(0, batch_size, 8):
            for round in range(rounds):
                is_last_layer = (round + 1) % (forest_height + 1) == 0

                # Load current tree values
                bundle: list[Slot] = [("+", v_tmp1, v_mem_tree_vals, v_curr_idx)]
                if is_last_layer:
                    # Bundle this here to save one instruction
                    bundle.append(("vbroadcast", v_curr_idx, one_const))
                self.add("valu", bundle)
                for j in range(4):
                    # Can do 2 loads in parallel
                    self.add(
                        "load",
                        [
                            ("load_offset", v_curr_tree_vals, v_tmp1, 2 * j),
                            ("load_offset", v_curr_tree_vals, v_tmp1, 2 * j + 1),
                        ],
                    )

                self.add(
                    "debug",
                    [
                        (
                            "vcompare",
                            v_curr_idx,
                            [(batch, round, "v_curr_idx", i) for i in range(8)],
                        ),
                        (
                            "vcompare",
                            v_curr_node_vals,
                            [(batch, round, "v_curr_node_vals", i) for i in range(8)],
                        ),
                        (
                            "vcompare",
                            v_curr_tree_vals,
                            [(batch, round, "v_curr_tree_vals", i) for i in range(8)],
                        ),
                    ],
                )

                # node = hash(node ^ tree)
                self.add_single(
                    "valu", ("^", v_curr_node_vals, v_curr_node_vals, v_curr_tree_vals)
                )
                self.add_single(
                    "debug",
                    (
                        "vcompare",
                        v_curr_node_vals,
                        [(batch, round, "hash_input", i) for i in range(8)],
                    ),
                )
                for hi, (op1, v_a, op3, op2, v_b) in enumerate(v_hash_stages):
                    self.add(
                        "valu",
                        [
                            (op1, v_tmp1, v_curr_node_vals, v_a),
                            (op2, v_tmp2, v_curr_node_vals, v_b),
                        ],
                    )
                    self.add_single("valu", (op3, v_curr_node_vals, v_tmp1, v_tmp2))
                    self.add_single(
                        "debug",
                        (
                            "vcompare",
                            v_curr_node_vals,
                            [(batch, round, "hash_stage", hi, i) for i in range(8)],
                        ),
                    )

                if not is_last_layer:
                    self.add_single(
                        "valu", ("%", v_tmp1, v_curr_node_vals, v_two_const)
                    )
                    self.add_single(
                        "valu",
                        ("multiply_add", v_curr_idx, v_two_const, v_curr_idx, v_tmp1),
                    )

            self.add_single(
                "debug",
                (
                    "vcompare",
                    v_curr_node_vals,
                    [(batch, "final_values", i) for i in range(8)],
                ),
            )
            self.add_single("store", ("vstore", mem_node_vals, v_curr_node_vals))
            self.add_single("flow", ("add_imm", mem_node_vals, mem_node_vals, 8))
        self.add_single("flow", ("pause",))


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(my_reference_kernel(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(10)
            inp = Input.generate(f, 256, 16)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in my_reference_kernel(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
