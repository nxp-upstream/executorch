/*
 * Copyright 2024 NXP
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Implementation of the backend for the NXP Neutron NPU.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include "NeutronDriver.h"
#include "NeutronErrors.h"

using namespace std;

namespace torch {
namespace executor {

// Aggregate neutron model handle and data structures into one.
typedef struct {
    int numInputs = 0;
    int numOutputs = 0;
    NeutronModelConfig mcfg;
    NeutronDataConfig dcfg;
    NeutronModelHandle nmh = NULL;
} NeutronConfig;

class NeutronBackend final : public PyTorchBackendInterface {
 public:
  NeutronBackend() {}

  ~NeutronBackend() = default;

  virtual bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {

    MemoryAllocator* allocator = context.get_runtime_allocator();
    
    auto *cfg = allocator->allocateInstance<NeutronConfig>();

    // The following data is read from the "processed" data blob.
    //    cfg->numInputs
    //    cfg->numoutputs
    //    cfg->mcfg.microcode
    //    cfg->mcfg.weights
    //    cfg->mcfg.kernels
    const uint32_t* buffer = static_cast<const uint32_t*>(processed->data());
    uint32_t magicWord = buffer[0];
    // Check valid microcode.
    if (magicWord != 0x64434D6E) {
      ET_LOG(Error, "Preprocessed buffer does not contain a valid Neutron microcode");
      return Error::InvalidProgram;
    }
    uint32_t microcodeSize = buffer[6];
    uint32_t weightsSize = buffer[7];
    cfg->numInputs = buffer[9];
    cfg->numOutputs = buffer[10];
    cfg->mcfg.microcode = static_cast<const uint8_t*>(processed->data());
    cfg->mcfg.weights = static_cast<const uint8_t*>(cfg->mcfg.microcode) + microcodeSize;
    cfg->mcfg.kernels = static_cast<const uint8_t*>(cfg->mcfg.weights) + weightsSize;

    // Allocate place for input and output pointers.
    cfg->dcfg.inputs = static_cast<const void**>(allocator->allocate(cfg->numInputs * sizeof(void*)));
    cfg->dcfg.outputs = static_cast<void**>(allocator->allocate(cfg->numOutputs * sizeof(void*)));
    
    // Prepare data for through neutron driver.
    NeutronError neutronRC = neutronModelPrepare((const NeutronModelConfig *)&cfg->mcfg, &cfg->nmh);
    if (neutronRC != ENONE) {
      ET_LOG(Error, "Neutron model preparation failed with error code %d", neutronRC);
      return Error::InvalidProgram;
    }

    return cfg;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* input_handle,
      EValue** args) const override {
    
    NeutronConfig *cfg = static_cast<NeutronConfig *>(input_handle);

    // Set inputs and outputs from args.    
    for (int i = 0; i < cfg->numInputs; i++) {
      cfg->dcfg.inputs[i] = args[i]->toTensor().const_data_ptr();
    }
    for (int i = 0; i < cfg->numOutputs; i++) {
      cfg->dcfg.outputs[i] = args[cfg->numInputs + i]->toTensor().mutable_data_ptr();
    }

    // TODO: Use trace from BackendExecutionContext.
    NeutronTraceConfig trace_config{.traceConfig = 0};
    neutronSetTrace(cfg->nmh, &trace_config);

    // Run neutron compute.
    NeutronError neutronRC = neutronRunBlocking(cfg->nmh, &cfg->dcfg);
    if (neutronRC != ENONE) {
      ET_LOG(Error, "Neutron model evaluation failed with error code %d", neutronRC);
      return Error::InvalidProgram;
    }

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    NeutronConfig *cfg = reinterpret_cast<NeutronConfig *>(handle);

    // Unprepare to free resources in neutron driver.
    NeutronError neutronRC = neutronModelUnprepare(cfg->nmh);
    (void)neutronRC;

    // Deallocation is done automatically.
    /*
    delete[] cfg->dcfg.inputs;
    delete[] cfg->dcfg.outputs;
    delete cfg;
    */
    return;
  }
};

namespace {
auto backend = NeutronBackend();
Backend backend_id{"NeutronBackend", &backend};
static auto registered = register_backend(backend_id);
} // namespace

} // namespace executor
} // namespace torch
