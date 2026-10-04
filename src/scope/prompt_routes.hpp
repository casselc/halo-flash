#ifndef HALO_SCOPE_PROMPT_ROUTES_HPP_
#define HALO_SCOPE_PROMPT_ROUTES_HPP_

// Langfuse-style prompt management over HTTP (PLAN.md P12), on the SQLite
// metadata store: immutable versions, movable labels ("latest" follows the
// newest version; "production" or any other label is set explicitly).
//   POST /v1/scope/prompts                   {name, template, config?}
//                                            -> {name, version, labels}
//   GET  /v1/scope/prompts/{name}[?label=L | ?version=N]   (default: latest)
//                                            -> {name, version, template, config, labels}
//   POST /v1/scope/prompts/{name}/labels     {label, version}

#include "scope/metadata.hpp"
#include "server/http.hpp"

namespace halo::scope {

/// `store` must outlive the server's Stop().
void RegisterPromptRoutes(server::HttpServer& server, MetadataStore& store);

}  // namespace halo::scope

#endif  // HALO_SCOPE_PROMPT_ROUTES_HPP_
