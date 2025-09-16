# Bash completion for FineTune Makefile
_ft_make_completion() {
    local cur targets
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    targets=$(make -qp | awk -F':' '/^[a-zA-Z0-9][^0\/	=]*:([^=]|$)/ {split($1,A,/ /);for(i in A)print A[i]}' | sort -u)
    COMPREPLY=( $(compgen -W "$targets" -- $cur) )
    return 0
}
complete -F _ft_make_completion make

