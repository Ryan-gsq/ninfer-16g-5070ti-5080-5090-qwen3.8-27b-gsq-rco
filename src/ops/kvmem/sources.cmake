# KVMem component: the ring that lets the Device KV pool be smaller than the logical context by
# demoting pages to the Host tier and bringing back the ones retrieval needs.
#
# WHY THE LIST IS EXPLICIT
#   Sources are declared per directory in these manifests -- src/ops/CMakeLists.txt includes each
#   subdirectory's sources.cmake -- so this file names the twelve translation units of the component
#   that compile into ninfer_ops. Headers here are header-only and compile through their includers,
#   so they are not listed: kvmem_score.h (inline accumulator), kvmem_window.h, kvmem_window_plan.h,
#   kvmem_select_actkv.h (ActKV LRFU + semantic protection, inline), kvmem_shadow.h,
#   raw_k_block_arena.h and the qw3 .hpp policy layer.
#
# REQUIRED COMPANION LINE (in ops/CMakeLists.txt, beside this include):
#   target_include_directories(ninfer_ops PRIVATE ${PROJECT_SOURCE_DIR}/src/ops/kvmem)
#   -- the qw3 policy layer keeps upstream's own include spelling ("qw3/kvmem_store.hpp"), and this
#      include root is what resolves it without editing the ported files. Removing the line must fail
#      the build with "qw3/kvmem_store.hpp not found"; that failure is the intended negative control.
#
# OPT-IN
#   The component is inert unless the ring is configured (the NINFER_KV_* switches in the repo
#   README): with the ring off it allocates nothing and the engine runs as it does without it.

target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/raw_k_shadow.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/raw_k_shadow_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/raw_k_harvest.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mean_k_index.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mean_k_index_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kvmem_select.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kvmem_window_assembly.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kvmem_retrieve.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kvmem_retrieve_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kvmem_port_bridge.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/qw3/kvmem_store.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/qw3/kvmem_gc.cpp"
)
