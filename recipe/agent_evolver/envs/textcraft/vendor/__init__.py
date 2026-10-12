# Vendored from AgentGym (commit 3ef9235). See VENDOR.md.
#
# Intentionally a bare stub: environment.py / crafting_tree.py / utils.py use
# relative imports (from .utils import ...) so `vendor` must be a package, but
# we do NOT re-export here — importing `vendor.environment` pulls in gymnasium,
# and build_manifest only needs `vendor.crafting_tree`.  Import the submodules
# you need directly.
