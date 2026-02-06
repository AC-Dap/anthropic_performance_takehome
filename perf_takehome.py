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
from typing import TypeAlias

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
        self.named_scratch: dict[str, Address] = {}
        self.named_scratch_debug: dict[Address, tuple[str, int]] = {}
        self.scratch_ptr: Address = 0
        self.const_map: dict[int, Address] = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.named_scratch_debug)

    def bundle(self, instr: Instruction):
        return self.instrs.append(instr)

    def add_single(self, engine: Engine, slot: Slot):
        self.instrs.append({engine: [slot]})

    def add(self, engine: Engine, slots: list[Slot]):
        self.instrs.append({engine: slots})

    def scratch(self, name: str | None = None, length: int = 1):
        if name in self.named_scratch:
            return self.named_scratch[name]

        addr = self.scratch_ptr
        if name is not None:
            self.named_scratch[name] = addr
            self.named_scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space:\n{self.named_scratch_debug}"
        return addr

    def scratch_vec(self, name: str | None = None):
        return self.scratch(name, 8)

    def build_load_tree_vals(self, level: int, start: int, end: int):
        if level == 0:
            # We've already loaded the root node
            return

        def build_alu_step(node_idx: int, batch_idx: int):
            slots: list[Slot] = []
            tmp1 = vec_at(self.v_tmp1, batch_idx)
            curr_idx = vec_at(self.v_curr_idx, node_idx)
            curr_tree_val = vec_at(self.v_curr_tree_vals, node_idx)
            for i in range(1, 2**level):
                tree_idx = 2**level + i
                slots.append(("==", tmp1, curr_idx, self.consts[tree_idx]))
                slots.append(("*", tmp1, tmp1, self.v_first_tree_vals[tree_idx - 1]))
                base = self.v_first_tree_vals[2**level - 1] if i == 0 else curr_tree_val
                slots.append(("+", curr_tree_val, tmp1, base))
            return slots

        def build_valu_step(node_idx: int, batch_idx: int):
            slots: list[Slot] = []
            tmp1 = vec_at(self.v_tmp1, batch_idx)
            curr_idx = vec_at(self.v_curr_idx, node_idx)
            curr_tree_val = vec_at(self.v_curr_tree_vals, node_idx)
            for i in range(1, 2**level):
                tree_idx = 2**level + i
                slots.append(("==", tmp1, curr_idx, self.consts[tree_idx]))
                base = self.v_first_tree_vals[2**level - 1] if i == 0 else curr_tree_val
                slots.append(("multiply_add", curr_tree_val, tmp1, self.v_first_tree_vals[tree_idx - 1], base))
            return slots

        def build_load_step(node_idx: int, batch_idx: int, num: int):
            alu1_slots: list[Slot] = []
            alu2_slots: list[Slot] = []
            load1_slots: list[Slot] = []
            load2_slots: list[Slot] = []

            tmp1 = vec_at(self.v_tmp1, batch_idx)
            tmp2 = vec_at(self.v_tmp2, batch_idx)
            alu1_slots.append(("+", tmp1, self.mem_tree_vals, vec_at(self.v_curr_idx, node_idx)))
            alu2_slots.append(("+", tmp2, self.mem_tree_vals, vec_at(self.v_curr_idx, node_idx + 1)))
            for j in range(0, num, 2):
                if j < num - 2:
                    alu1_slots.append(("+", tmp1, self.mem_tree_vals, vec_at(self.v_curr_idx, node_idx + 2 + j)))
                    alu2_slots.append(("+", tmp2, self.mem_tree_vals, vec_at(self.v_curr_idx, node_idx + 3 + j)))
                load1_slots.append(("load", vec_at(self.v_curr_tree_vals, node_idx + j), tmp1))
                load2_slots.append(("load", vec_at(self.v_curr_tree_vals, node_idx + 1 + j), tmp2))
            return alu1_slots, alu2_slots, load1_slots, load2_slots

        n_valu = min(SLOT_LIMITS["valu"], (end - start) // VLEN)
        n_alu = (end - start) - n_valu * VLEN

        if level == 1:
            valu_slots = [build_valu_step(start + i, i) for i in range(0, n_valu * VLEN, VLEN)]
            alu_slots = [build_alu_step(start + n_valu * VLEN + i, n_valu * VLEN + i) for i in range(n_alu)]
            if n_alu > 0:
                self.bundle({"valu": [slot[0] for slot in valu_slots], "alu": [slot[0] for slot in alu_slots]})
                self.bundle({"valu": [slot[1] for slot in valu_slots], "alu": [slot[1] for slot in alu_slots]})
                self.bundle({"alu": [slot[2] for slot in alu_slots]})
            else:
                self.add("valu", [slot[0] for slot in valu_slots])
                self.add("valu", [slot[1] for slot in valu_slots])
        elif level == 2:
            valu_slots = [build_valu_step(start + i, i) for i in range(0, n_valu * VLEN, VLEN)]
            alu1_slots, alu2_slots, load1_slots, load2_slots = build_load_step(
                start + n_valu * VLEN, n_valu * VLEN, n_alu
            )
            if n_alu > 0:
                self.bundle({"valu": [slot[0] for slot in valu_slots], "alu": [alu1_slots[0], alu2_slots[0]]})
                for i in range(1, len(valu_slots)):
                    self.bundle(
                        {
                            "valu": [slot[i] for slot in valu_slots],
                            "alu": [alu1_slots[i], alu2_slots[i]],
                            "load": [load1_slots[i - 1], load2_slots[i - 1]],
                        }
                    )
                self.add("load", [load1_slots[-1], load2_slots[-1]])
            else:
                for i in range(len(valu_slots)):
                    self.add("valu", [slot[i] for slot in valu_slots])
        elif level == 3 or level == 4:
            valu_slots = [build_valu_step(start + i, i) for i in range(0, n_valu * VLEN, VLEN)]
            alu1_slots, alu2_slots, load1_slots, load2_slots = build_load_step(
                start + n_valu * VLEN, n_valu * VLEN, n_alu
            )
            if n_alu > 0:
                self.bundle({"valu": [slot[0] for slot in valu_slots], "alu": [alu1_slots[0], alu2_slots[0]]})
                for i in range(1, len(valu_slots)):
                    bundle: Instruction = {"valu": [slot[i] for slot in valu_slots]}
                    if i < len(alu1_slots):
                        bundle["alu"] = [alu1_slots[i], alu2_slots[i]]
                    if i - 1 < len(load1_slots):
                        bundle["load"] = [load1_slots[i - 1], load2_slots[i - 1]]
                    self.bundle(bundle)
            else:
                for i in range(len(valu_slots)):
                    self.add("valu", [slot[i] for slot in valu_slots])
        else:
            alu1_slots, alu2_slots, load1_slots, load2_slots = build_load_step(start, 0, end - start)
            self.add("alu", [alu1_slots[0], alu2_slots[0]])
            for i in range(1, len(alu1_slots)):
                self.bundle({"alu": [alu1_slots[i], alu2_slots[i]], "load": [load1_slots[i - 1], load2_slots[i - 1]]})
            self.add("load", [load1_slots[-1], load2_slots[-1]])

    def build_alu_step(self, node_idx: int, batch_idx: int, level: int, forest_height: int):
        curr_tree_val = self.v_first_tree_vals[0] if level == 0 else vec_at(self.v_curr_tree_vals, node_idx)
        curr_node_val = vec_at(self.v_curr_node_vals, node_idx)
        curr_idx = vec_at(self.v_curr_idx, node_idx)
        tmp1 = vec_at(self.v_tmp1, batch_idx)
        tmp2 = vec_at(self.v_tmp2, batch_idx)

        steps: list[Slot] = []
        steps.append(("^", curr_node_val, curr_node_val, curr_tree_val))
        for hi, (op1, _, op3, op2, _) in enumerate(HASH_STAGES):
            steps.append((op1, tmp1, curr_node_val, self.hash_a[hi]))
            steps.append((op2, tmp2, curr_node_val, self.hash_b[hi]))
            steps.append((op3, curr_node_val, tmp1, tmp2))

        is_last_layer = level == forest_height
        if not is_last_layer:
            steps.append(("%", tmp1, curr_node_val, self.consts[2]))
            if level == 0:
                steps.append(("+", curr_idx, self.consts[2], tmp1))
            elif not is_last_layer:
                steps.append(("*", curr_idx, self.consts[2], curr_idx))
                steps.append(("+", curr_idx, curr_idx, tmp1))

        return steps

    def build_valu_step(self, node_idx: int, batch_idx: int, level: int, forest_height: int):
        v_curr_tree_vals = self.v_first_tree_vals[0] if level == 0 else vec_at(self.v_curr_tree_vals, node_idx)
        v_curr_node_vals = vec_at(self.v_curr_node_vals, node_idx)
        v_curr_idx = vec_at(self.v_curr_idx, node_idx)
        v_tmp1 = vec_at(self.v_tmp1, batch_idx)
        v_tmp2 = vec_at(self.v_tmp2, batch_idx)

        steps: list[Slot] = []
        steps.append(("^", v_curr_node_vals, v_curr_node_vals, v_curr_tree_vals))
        for hi, (op1, _, op3, op2, _) in enumerate(HASH_STAGES):
            steps.append((op1, v_tmp1, v_curr_node_vals, self.hash_a[hi]))
            steps.append((op2, v_tmp2, v_curr_node_vals, self.hash_b[hi]))
            steps.append((op3, v_curr_node_vals, v_tmp1, v_tmp2))

        is_last_layer = level == forest_height
        if not is_last_layer:
            steps.append(("%", v_tmp1, v_curr_node_vals, self.consts[2]))
            if level == 0:
                steps.append(("+", v_curr_idx, self.consts[2], v_tmp1))
            else:
                steps.append(("multiply_add", v_curr_idx, self.consts[2], v_curr_idx, v_tmp1))

        return steps

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Only using simd, no parallel slots
        """
        num_parallel = SLOT_LIMITS["valu"] * VLEN + SLOT_LIMITS["alu"]

        # Intermediate variables we'll need
        self.v_curr_tree_vals = self.scratch("v_curr_tree_vals", length=batch_size)
        self.v_curr_node_vals = self.scratch("v_curr_node_vals", length=batch_size)
        self.v_curr_idx = self.scratch("v_curr_idx", length=batch_size)
        self.v_tmp1 = self.scratch("v_tmp1", length=num_parallel)
        self.v_tmp2 = self.scratch("v_tmp2", length=num_parallel)

        # input is only 7 variables, but allocate 8 so vload doesn't stomp on 8th entry
        self.v_mem_input = self.scratch_vec("v_mem_input")
        self.mem_tree_vals = self.v_mem_input + 4
        self.mem_node_vals = self.v_mem_input + 6

        self.v_first_tree_vals = [self.scratch_vec(f"v_first_tree_vals_{i + 1}") for i in range(32)]

        self.consts = [self.scratch("const_0")] + [self.scratch_vec(f"v_const_{i + 1}") for i in range(32)]

        self.hash_a = [self.scratch_vec(f"v_hash_a_{i}") for i in range(len(HASH_STAGES))]
        self.hash_b = [self.scratch_vec(f"v_hash_b_{i}") for i in range(len(HASH_STAGES))]

        # Initialize consts
        self.add_single("load", ("const", self.consts[0], 0))
        for i in range(1, 33, 2):
            if i == 1:
                self.add("load", [("const", self.consts[i], i), ("const", self.consts[i + 1], i + 1)])
            else:
                self.bundle(
                    {
                        "valu": [
                            ("vbroadcast", self.consts[i - 2], self.consts[i - 2]),
                            ("vbroadcast", self.consts[i - 1], self.consts[i - 1]),
                        ],
                        "load": [("const", self.consts[i], i), ("const", self.consts[i + 1], i + 1)],
                    }
                )
        self.add(
            "valu", [("vbroadcast", self.consts[31], self.consts[31]), ("vbroadcast", self.consts[32], self.consts[32])]
        )

        # Initialize hash constants
        for i, (_, a, _, _, b) in enumerate(HASH_STAGES):
            if i == 0:
                self.add("load", [("const", self.hash_a[i], a), ("const", self.hash_b[i], b)])
            else:
                self.bundle(
                    {
                        "valu": [
                            ("vbroadcast", self.hash_a[i - 1], self.hash_a[i - 1]),
                            ("vbroadcast", self.hash_b[i - 1], self.hash_b[i - 1]),
                        ],
                        "load": [("const", self.hash_a[i], a), ("const", self.hash_b[i], b)],
                    }
                )
        self.add(
            "valu", [("vbroadcast", self.hash_a[-1], self.hash_a[-1]), ("vbroadcast", self.hash_b[-1], self.hash_b[-1])]
        )

        # Initialize first_tree_vals
        # We load scalar values into tmp2, then broadcast each into the vectors
        self.add(
            "alu",
            [
                ("+", self.v_tmp1, self.mem_tree_vals, self.consts[0]),
                ("+", self.v_tmp1 + 1, self.mem_tree_vals, self.consts[VLEN]),
            ],
        )  # Loop idx
        for offset in range(0, 32, 2 * VLEN):
            self.bundle(
                {
                    "load": [
                        ("vload", vec_at(self.v_tmp2, offset), self.v_tmp1),
                        ("vload", vec_at(self.v_tmp2, offset + VLEN), self.v_tmp1 + 1),
                    ],
                    "alu": [
                        ("+", self.v_tmp1, self.v_tmp1, self.consts[2 * VLEN]),
                        ("+", self.v_tmp1 + 1, self.v_tmp1 + 1, self.consts[2 * VLEN]),
                    ],
                }
            )
        for offset in range(0, 32, SLOT_LIMITS["valu"]):
            self.add(
                "valu",
                [
                    ("vbroadcast", self.v_first_tree_vals[i], vec_at(self.v_tmp2, i))
                    for i in range(min(32 - offset, SLOT_LIMITS["valu"]))
                ],
            )

        # Store difference between node + first node in each layer
        slots = (
            [("-", self.v_first_tree_vals[2], self.v_first_tree_vals[2], self.v_first_tree_vals[1])]  # Layer 2
            + [
                ("-", self.v_first_tree_vals[i], self.v_first_tree_vals[i], self.v_first_tree_vals[3])
                for i in range(4, 7)
            ]  # Layer 3
            + [
                ("-", self.v_first_tree_vals[i], self.v_first_tree_vals[i], self.v_first_tree_vals[7])
                for i in range(8, 15)
            ]  # Layer 4
            + [
                ("-", self.v_first_tree_vals[i], self.v_first_tree_vals[i], self.v_first_tree_vals[15])
                for i in range(16, 32)
            ]  # Layer 5
        )
        self.add("valu", slots[:6])
        self.add("valu", slots[6:12])
        self.add("valu", slots[12:18])
        self.add("valu", slots[18:24])
        self.add("valu", slots[24:])

        # Initialize v_curr_node_vals
        self.add(
            "alu",
            [
                ("+", self.v_tmp1, self.mem_node_vals, self.consts[0]),
                ("+", self.v_tmp2, self.mem_node_vals, self.consts[VLEN]),
            ],
        )  # Loop idx
        for offset in range(0, batch_size, 2 * VLEN):
            self.bundle(
                {
                    "load": [
                        ("vload", vec_at(self.v_curr_node_vals, offset), self.v_tmp1),
                        ("vload", vec_at(self.v_curr_node_vals, offset + VLEN), self.v_tmp2),
                    ],
                    "alu": [
                        ("+", self.v_tmp1, self.v_tmp1, self.consts[2 * VLEN]),
                        ("+", self.v_tmp2, self.v_tmp2, self.consts[2 * VLEN]),
                    ],
                }
            )

        # Initialize others
        self.add_single("alu", ("-", self.mem_tree_vals, self.mem_tree_vals, self.consts[1]))  # So we can 1-index

        self.add_single("flow", ("pause",))
        assert batch_size % VLEN == 0, "Batch size not in even chunks of VLEN"

        batches = [(start, min(batch_size, start + num_parallel)) for start in range(0, batch_size, num_parallel)]
        for round in range(rounds):
            for start, end in batches:
                level = round % (forest_height + 1)

                # Load current tree values
                self.build_load_tree_vals(level, start, end)

                # Assemble hashing steps in parallel
                # Greedily take valu first
                idx = start
                alu_slots = []
                valu_slots = []
                for _ in range(SLOT_LIMITS["valu"]):
                    if end - idx < SLOT_LIMITS["valu"]:
                        break
                    valu_slots.append(self.build_valu_step(idx, idx - start, level, forest_height))
                    idx += VLEN

                for _ in range(SLOT_LIMITS["alu"]):
                    if idx >= end:
                        break
                    alu_slots.append(self.build_alu_step(idx, idx - start, level, forest_height))
                    idx += 1

                for step in range(len(valu_slots[0])):
                    self.bundle(
                        {
                            "alu": [alu_slot[step] for alu_slot in alu_slots],
                            "valu": [valu_slot[step] for valu_slot in valu_slots],
                        }
                    )

                # ALU is sometimes one instruction longer
                if len(alu_slots) > 0 and len(alu_slots[0]) > len(valu_slots[0]):
                    self.add("alu", [alu_slot[-1] for alu_slot in alu_slots])

        # Write out v_curr_node_vals
        self.add(
            "alu",
            [
                ("+", self.v_tmp1, self.mem_node_vals, self.consts[0]),
                ("+", self.v_tmp2, self.mem_node_vals, self.consts[VLEN]),
            ],
        )  # Loop idx
        for offset in range(0, batch_size, 2 * VLEN):
            self.bundle(
                {
                    "store": [
                        ("vstore", self.v_tmp1, vec_at(self.v_curr_node_vals, offset)),
                        ("vstore", self.v_tmp2, vec_at(self.v_curr_node_vals, offset + VLEN)),
                    ],
                    "alu": [
                        ("+", self.v_tmp1, self.v_tmp1, self.consts[2 * VLEN]),
                        ("+", self.v_tmp2, self.v_tmp2, self.consts[2 * VLEN]),
                    ],
                }
            )
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
