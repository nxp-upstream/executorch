/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpContextCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

class QnnDlcManager;
class HtpContext : public QnnContext {
 public:
  HtpContext(
      const QnnImplementation& implementation,
      QnnBackend* backend,
      QnnDevice* device,
      QnnBackendCache* cache,
      const QnnExecuTorchHtpBackendOptions* htp_options,
      QnnDlcManager* qnn_dlc_manager)
      : QnnContext(implementation, backend, device, cache, qnn_dlc_manager) {
    htp_context_custom_config_ =
        std::make_unique<HtpContextCustomConfig>(this, htp_options);
  }
  ~HtpContext() {}

  Qnn_ContextHandle_t GetSpillFillHandle() const {
    return sf_handle_;
  }

 protected:
  executorch::runtime::Error MakeConfig(
      std::vector<const QnnContext_Config_t*>& config) override;
  executorch::runtime::Error AfterConfigure() override;

 private:
  std::vector<QnnContext_Config_t> context_config_;
  std::unique_ptr<HtpContextCustomConfig> htp_context_custom_config_;
  // this is shared between all partitions inside one pte
  static inline Qnn_ContextHandle_t sf_handle_ = 0x0;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
