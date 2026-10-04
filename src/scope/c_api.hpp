#ifndef HALO_SCOPE_C_API_HPP_
#define HALO_SCOPE_C_API_HPP_

// C++ side of include/halo_scope.h: a handle over an existing recorder, for
// the engine (which owns its recorder) and for tests (memory sinks, fake
// clocks). Free it with halo_scope_close, which leaves the recorder running.

#include <string>

#include "halo_scope.h"
#include "scope/annotate.hpp"

namespace halo::scope {

halo_scope* WrapRecorder(Recorder& rec, std::string project, Capture capture, Annotator::Clock clock = {});

}  // namespace halo::scope

#endif  // HALO_SCOPE_C_API_HPP_
