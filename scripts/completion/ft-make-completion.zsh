# Zsh completion for FineTune Makefile
#compdef make
_make_targets() {
    local targets
    targets=($(make -qp | awk -F':' '/^[a-zA-Z0-9][^0\/	=]*:([^=]|$)/ {split($1,A,/ /);for(i in A)print A[i]}' | sort -u))
    _describe 'make targets' targets
}
_make_targets

