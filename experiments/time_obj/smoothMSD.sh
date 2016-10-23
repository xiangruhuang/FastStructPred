cat $1 | grep 'best_primal_obj' | sed 's/.* best_primal_obj=//' | sed 's/,.*time=/ /' | awk '{print $2 " " $1}' > $2
