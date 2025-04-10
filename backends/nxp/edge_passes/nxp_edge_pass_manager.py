# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

from executorch.backends.nxp.edge_passes.move_auxiliary_operator_into_separate_qdq_cluster_pass import \
    MoveTrailingAuxiliaryOperatorIntoSeparateQDQClusterPass, MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass
from torch.export import ExportedProgram
from torch.fx.passes.infra.pass_manager import PassManager

from executorch.backends.nxp.edge_passes.nxp_edge_pass import NXPEdgePass
from executorch.exir.program._program import _get_updated_graph_signature, _get_updated_range_constraints, \
    EdgeProgramManager
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier


class NXPEdgePassManager:
    def __init__(self, edge_program_manager: EdgeProgramManager, passes: list[type[NXPEdgePass]] | None = None):
        self.edge_program_manager = edge_program_manager

        self.passes = passes or [
            # Run all NXP edge passes by default.
        ]

    def transform(self) -> EdgeProgramManager:
        """ Apply all NXPEdgePasses in `self.passes` as long as they are making changes. """

        iteration_limit = 10  # Empirical value.
        for _ in range(iteration_limit):
            self.edge_program_manager, modified = self._run_one_cycle_of_passes()
            if not modified:
                break

        return self.edge_program_manager

    # noinspection PyProtectedMember
    def _run_one_cycle_of_passes(self) -> (EdgeProgramManager, bool):
        """ Apply all NXPEdgePasses in `self.passes` to modify the edge program. """
        new_programs: dict[str, ExportedProgram] = {}
        overall_modified = False

        # Go through all the edge programs in the model. There is usually just the "forward" program.
        for name, program in self.edge_program_manager._edge_programs.items():
            new_program = program

            # Cycle through all passes.
            for pass_class in self.passes:
                try:
                    new_program, modified = run_pass(new_program, pass_class())
                    EXIREdgeDialectVerifier(edge_compile_config=self.edge_program_manager.compile_config)(
                        new_program.graph_module
                    )
                    overall_modified = overall_modified or modified

                except Exception as e:
                    logging.warning(f'An exception occurred during the pre-processing pass `{pass_class}`. '
                                    'Please report this issue.\n' + str(e))

                finally:
                    new_programs[name] = new_program

        return EdgeProgramManager(
            new_programs, copy.deepcopy(self.edge_program_manager._config_methods),
            self.edge_program_manager.compile_config
        ), overall_modified


def run_pass(exported_program: ExportedProgram, pass_: NXPEdgePass) -> (ExportedProgram, bool):
    """ Run a given NXPEdgePass and return a tuple with the modified graph and a bool indicating if changes were made.

        The code is a modified version of `executorch/exir/program/_program.py:_transform()`
    """
    res = PassManager([pass_])(exported_program.graph_module)
    transformed_module = res.graph_module if res is not None else exported_program.graph_module
    assert transformed_module is not None

    if transformed_module is exported_program.graph_module and not res.modified:
        # No changes have been made
        return exported_program, False

    transformed_ep = ExportedProgram(
        root=transformed_module,
        graph=transformed_module.graph,
        graph_signature=_get_updated_graph_signature(
            exported_program.graph_signature, transformed_module
        ),
        state_dict=exported_program.state_dict,
        range_constraints=_get_updated_range_constraints(transformed_module),
        module_call_graph=copy.deepcopy(exported_program._module_call_graph),
        example_inputs=exported_program.example_inputs,
        constants=exported_program.constants,
        verifiers=[exported_program.verifier],
    )
    transformed_ep.graph_module.meta.update(exported_program.graph_module.meta)
    transformed_ep.graph_module.meta.update(res.graph_module.meta)
    return transformed_ep, res.modified
