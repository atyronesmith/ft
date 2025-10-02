# Zsh completion for FineTune Makefile
#compdef make

_ft_make_completion() {
    # Only run if we're in the completion context and in the ft directory
    if [[ -f "Makefile" && "$PWD" == */ft ]]; then
        local targets
        targets=($(make -qp 2>/dev/null | awk -F':' '/^[a-zA-Z0-9][^0\/	=]*:([^=]|$)/ {split($1,A,/ /);for(i in A)print A[i]}' | sort -u))
        _describe 'make targets' targets
    fi
}

# Only register completion if we're actually in zsh completion context
if [[ -n "$ZSH_VERSION" && -n "$_comp_name" ]]; then
    compdef _ft_make_completion make
fi

